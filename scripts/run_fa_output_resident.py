"""Two-way output-resident FA: independent math gates, then T4096 comparison.

Scoped SPM correctness repairs. Revalidate historical cache and V-SPM;
run new O-SPM and a same-source/tile coherent control, at most two sims.
"""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import numpy as np
from build_access_timing import ROOT,sha,save
from run_access_timing import stats
from test_micro_timing import check_global_guard
from fa_headroom_profiles import leaves
from run_fa_headroom import prior,validate as validate_old,diagnostics,ENV,REF
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
from single_head_reference import check

BASE=ROOT/'results/cacheflex_flash_attention'
OUT=BASE/os.environ.get('CACHEFLEX_OUTPUT_EXPERIMENT','output_resident_v8_20260924')
GUESTS=BASE/'output_resident_20260924'
BUILD=GUESTS/os.environ.get('CACHEFLEX_OUTPUT_BUILD','build_model_v6')

def command(kind,t,path):
    backend='spm' if kind in ('output_spm','recheck_spm') else 'cache'
    old=prior(backend,t)
    cmd=json.loads((old/'command.json').read_text())
    cmd[0]=str(BUILD/'gem5.opt')
    for i,arg in enumerate(cmd):
        if arg.startswith('--outdir='):cmd[i]=f'--outdir={path}'
        if arg.endswith('/micro_timing_se.py'):cmd[i]=str(OUT/'configs/deprecated/example/micro_timing_se.py')
        if not kind.startswith('recheck_') and arg.startswith('system.cpu[0].workload[0].cmd='):
            cmd[i]=f"system.cpu[0].workload[0].cmd={['cacheflex-fa',str(t),'64','16','128']!r}"
    if not kind.startswith('recheck_'):
        cmd[cmd.index('--cmd')+1]=str(GUESTS/f'build_{backend}/guest.final')
    return cmd

def config_audit(path,kind,t):
    backend='spm' if kind=='output_spm' else 'cache';old=prior(backend,t)
    a=dict(leaves(json.loads((old/'config.json').read_text().replace(str(old),'<OUT>'))))
    b=dict(leaves(json.loads((path/'config.json').read_text().replace(str(path),'<OUT>'))))
    assert a.keys()==b.keys(),'Configuration shape changed'
    changes=[]
    for k,v in a.items():
        if b[k]==v:continue
        if k.endswith('.executable'):
            assert b[k]==str(GUESTS/f'build_{backend}/guest.final')
        elif re.fullmatch(r'system.cpu\[\d+\].workload\[0\].cmd\[3\]',k):
            assert b[k]=='16'
        else:raise AssertionError((k,v,b[k]))
        changes.append(dict(path=k,old=v,new=b[k]))
    r=json.loads((path/'config.json').read_text())['system']['ruby']
    for i in range(8):
        c=r[f'l1_cntrl{i}']['cache']
        assert c['size']==65536 and c['assoc']==8
        assert c['num_spm_ways']==(2 if backend=='spm' else 0)
    return changes

def validate(path,kind,t):
    if kind.startswith('recheck_'):
        backend=kind.removeprefix('recheck_')
        row=validate_old(path,backend,t,'control')
        oldstats=stats(prior(backend,t));newstats=stats(path)
        delta=[{k:[a.get(k),b.get(k)] for k in a.keys()|b.keys()
                if a.get(k)!=b.get(k)} for a,b in zip(oldstats,newstats)]
        assert len(newstats)==len(oldstats)
        if newstats!=oldstats:
            # V7 audit: core6 is awake for nine more empty cycles, with
            # identical ROI, instructions and ALL memory
            # counters. Permit only this exact recorded difference, not a
            # general exclusion of counters or a timing tolerance.
            allowed=json.loads((GUESTS/'preserved/t512_idle_counter_delta.json').read_text())
            assert backend=='spm' and t==512 and delta==allowed, \
                'Unaudited old-kernel counter change after repair'
        row.update(kind=kind,reused=False,exact_all_stats=newstats==oldstats,
                   counter_changes=delta)
        return row
    changes=config_audit(path,kind,t)
    numerical=check(path/'output.fp16',np.load(REF/f't{t}_bc128.npy'))
    # Preserve arithmetic exactly, not just tolerance agreement.
    assert sha(path/'output.fp16')==sha(prior('cache',t)/'output.fp16'),'Output bits changed'
    h=1469598103934665603
    for x in np.fromfile(path/'output.fp16',dtype=np.uint16):h=((h^int(x))*1099511628211)&((1<<64)-1)
    match=re.search(r'FLASH_RESULT.*?hash=([0-9a-f]+)',(path/'run.log').read_text())
    assert match and match[1]==f'{h:016x}','Printed/export hash mismatch'
    blocks=stats(path);s=blocks[0]
    for b in blocks:check_global_guard(b)
    copies=2048 if kind=='output_spm' else 0
    for name in ('m_spmcpRequestsIssued','m_spmcpRequestsCompleted'):
        assert sum(s[f'system.ruby.l0_cntrl{i}.sequencer.{name}'] for i in range(8))==copies
    assert sum(s[f'system.ruby.l0_cntrl{i}.sequencer.m_idealStagingCopies'] for i in range(8))==0
    stores=s.get('system.ruby.L1Cache_Controller.X.SPMST_direct::total',0)
    releases=s.get('system.ruby.L1Cache_Controller.X.SPM_release::total',0)
    if kind=='output_spm':
        # Two 128B stores/row on init, rescale, accumulation. Speculative loads
        # may add events, but stores only issue non-speculatively.
        assert stores==2*t*(1+2*(t//128)),(stores,t)
        assert releases==2048,releases
    return dict(kind=kind,T=t,path=str(path),roi_us=s['simTicks']/1e6,
                arithmetic=numerical,output_hash=match[1],bitwise_reference_equal=True,
                config_changes=changes,copy_count=copies,spm_stores=stores,
                spm_releases=releases,diagnostics=diagnostics(s),reused=False)

def report(state):
    lines=['# Output-resident FA: two existing SPM ways','',
           'Single head, 8 cores, VL16, d64, 64KiB private L2, 4MiB shared LLC. '
           'IFETCH-on MICRO-timing model with wide-SPM-store acknowledgment repair; '
           'no added SRAM or changed resource/timing parameters.', '',
           '| T | Kernel | End-to-end ROI (us) |','|---|---|---:|']
    for row in state['completed']:lines.append(f"| {row['T']} | {row['kind']} | {row['roi_us']:.4f} |")
    for t in (512,4096):
        rows={r['kind']:r for r in state['completed'] if r['T']==t}
        if 'output_spm' not in rows:continue
        for name in ('matched_cache','historical_cache','historical_v_spm'):
            if name in rows:
                ratio=rows[name]['roi_us']/rows['output_spm']['roi_us']
                lines.append(f'\nT{t}: {name}/output-SPM = {ratio:.6f}x.')
    lines+=['','T512 is a correctness gate, not the target performance claim. '
            'O-SPM and matched cache use Br16/Bc128/G4 and the same generated source. '
            'Historical best cache uses Br32/G2. FP16 GEMMs, FP32 online state, FP16 output. '
            'The 16KiB O tile is retained for all K/V panels. Existing SPMCP bootstrap '
            '(2048 lines total), zeroing, final normalization/store and release are inside ROI. '
            '128B widening/narrowing transfers plus ZIP/UZP avoid claiming four ways. '
            'No claim of newly tuned best-to-best or architectural novelty.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    OUT.mkdir()
    (OUT/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='freeze_and_revalidate',active={},completed=[])
    lock=threading.RLock();halt=threading.Event();manifest={}
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state);report(state)
    def frozen():
        for p,h in manifest.items():assert sha(p)==h,('Frozen input changed',p)
    def run(kind,t):
        frozen();path=OUT/'runs'/f'{kind}_t{t}';path.mkdir()
        cmd=command(kind,t,path);save(path/'command.json',cmd)
        proc=None;begin=time.monotonic()
        try:
            with (path/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][path.name]=dict(pid=proc.pid,path=str(path));status()
                while proc.poll() is None:
                    assert not halt.is_set(),'Peer failed'
                    assert time.monotonic()-begin<10800,'Three-hour limit'
                    assert shutil.disk_usage(OUT).free>50*2**30,'Disk guard'
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            assert proc.returncode==0,(kind,proc.returncode)
            row=validate(path,kind,t);frozen();row['host_seconds']=time.monotonic()-begin
            save(path/'validated.json',row)
            with lock:state['active'].pop(path.name);state['completed'].append(row);status()
        except BaseException as exc:
            halt.set();save(path/'failed.json',dict(error=repr(exc)))
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            with lock:state['active'].pop(path.name,None);status()
            raise
    try:
        status('waiting_for_store_fix_build')
        deadline=time.monotonic()+7200
        while not (BUILD/'frozen.json').exists():
            if (BUILD/'status.json').exists():
                assert json.loads((BUILD/'status.json').read_text())['stage']!='failed','Model build failed'
            assert time.monotonic()<deadline,'Build wait timeout'
            time.sleep(5)
        status('freeze_and_revalidate');shutil.copytree(BUILD/'configs',OUT/'configs')
        shutil.copy2(__file__,OUT/'runner_snapshot.py')
        manifest.update(json.loads((BUILD/'frozen.json').read_text()))
        paths=list((OUT/'configs').rglob('*.py'))+[Path(__file__).resolve()]
        paths.append(GUESTS/'roundtrip/guest')
        paths.append(GUESTS/'preserved/t512_idle_counter_delta.json')
        for b in ('cache','spm'):paths+=list(p for p in (GUESTS/f'build_{b}').iterdir() if p.is_file())
        for t in (512,4096):
            paths.append(REF/f't{t}_bc128.npy')
            for b in ('cache','spm'):
                old=prior(b,t)
                paths += [old/n for n in ('command.json','config.json','stats.txt','output.fp16','run.log')]
                row=validate_old(old,b,t,'control')
                row.update(kind='historical_cache' if b=='cache' else 'historical_v_spm',reused=True)
                state['completed'].append(row)
        manifest.update({str(p):sha(p) for p in paths});save(OUT/'inputs.json',manifest);frozen()
        status('wide_store_roundtrip')
        smoke=OUT/'runs/roundtrip';smoke.mkdir()
        cmd=json.loads((GUESTS/'roundtrip/command.json').read_text());cmd[0]=str(BUILD/'gem5.opt')
        for i,a in enumerate(cmd):
            if a.startswith('--outdir='):cmd[i]=f'--outdir={smoke}'
            if a.endswith('/micro_timing_se.py'):cmd[i]=str(OUT/'configs/deprecated/example/micro_timing_se.py')
        save(smoke/'command.json',cmd)
        with (smoke/'run.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
            state['active']['roundtrip']=dict(pid=proc.pid,path=str(smoke));status()
            try:rc=proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait();raise
        state['active'].pop('roundtrip');status()
        assert rc==0,('wide-store roundtrip failed',rc)
        assert 'OUTPUT_RESIDENT_ROUNDTRIP errors=0 elements=4096' in (smoke/'run.log').read_text()
        save(smoke/'validated.json',dict(passed=True,bit_patterns=4096,iterations=64))
        for t in (512,4096):
            status(f't{t}_old_kernel_revalidation')
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(run,k,t) for k in ('recheck_spm','recheck_cache')]
                for f in cf.as_completed(futures):f.result()
            status(f't{t}')
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(run,k,t) for k in ('output_spm','matched_cache')]
                for f in cf.as_completed(futures):f.result()
            if t==512:save(OUT/'gates_passed.json',dict(full_math=True,bitwise_match=True))
        status('completed_validated')
    except BaseException as exc:
        halt.set();state['error']=repr(exc);status('failed');raise

if __name__=='__main__':main()
