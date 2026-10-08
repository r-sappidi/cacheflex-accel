"""Isolated, correctness-gated FA interventions. Existing queues stay untouched.

All service/capacity interventions are diagnostics, not area-matched designs.
Two simulators maximum; no automatic acceptance after a failed check.
"""
import concurrent.futures as cf
import configparser
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import threading
import time

from run_fa_sequencer_sweep import ROOT, HERE, BASE, prior, reference, sha, save, read, check, verify_dump, np
from analyze_fa_retention_pcs import a, phase, instruction_bytes
from run_causal import parameters, config_key

SWEEP = ROOT/'results/cacheflex_flash_attention/sequencer_sweep_20260917'
BUILDS = ROOT/'results/cacheflex_flash_attention/pv_improvements_20260913/builds'
DEBUG = BUILDS/'ea1c7516d839b578/guest.debug'
GUEST = ROOT/'results/cacheflex_flash_attention/long_context_pv_20260914/spm.guest'

def patch_stride(src, dst):
    """d=64 only: replace one stride reload by mov x3,#128, preserving layout."""
    data=bytearray(src.read_bytes()); va=0x401e14
    assert instruction_bytes(data,va)==bytes.fromhex('e3f740f9')
    ph=struct.unpack_from('<Q',data,32)[0]; size,n=struct.unpack_from('<HH',data,54)
    offsets=[]
    for i in range(n):
        typ,_,off,addr,_,filesz,_,_=struct.unpack_from('<IIQQQQQQ',data,ph+i*size)
        if typ==1 and addr<=va and va+4<=addr+filesz: offsets.append(off+va-addr)
    assert len(offsets)==1
    off=offsets[0]; data[off:off+4]=bytes.fromhex('031080d2')
    dst.write_bytes(data); shutil.copymode(src,dst)
    original=src.read_bytes()
    assert original[:off]==data[:off] and original[off+4:]==data[off+4:]
    return dict(pc=hex(va),offset=off,old='ldr x3,[sp,#488]',new='mov x3,#128',
                source_sha256=sha(src),patched_sha256=sha(dst))

def audit(oldpath,newpath,params,patched):
    old=configparser.ConfigParser(interpolation=None,strict=False)
    new=configparser.ConfigParser(interpolation=None,strict=False)
    old.read(oldpath/'config.ini');new.read(newpath/'config.ini')
    assert old.sections()==new.sections()
    expected={config_key(k):str(v).lower() for k,v in params.items()}; seen=set(); changes=[]
    for s in old.sections():
        assert set(old[s])==set(new[s]),s
        for k,v in old[s].items():
            actual=new[s][k]
            if (s,k) in expected:
                assert actual.lower()==expected[s,k],(s,k,actual,expected[s,k]);seen.add((s,k))
            else:
                actual=actual.replace(str(newpath),'<OUT>')
                if patched:actual=actual.replace(str(patched),str(GUEST))
                assert v.replace(str(oldpath),'<OUT>')==actual,(s,k,v,actual)
            if v!=new[s][k]:changes.append([s,k,v,new[s][k]])
    assert seen==set(expected)
    return changes

def analyze(run,debug):
    a.phase=phase; result=a.analyze(run,debug)
    cmd=json.loads((run/'command.json').read_text())
    gd=Path(cmd[cmd.index('--cmd')+1]).read_bytes();dd=debug.read_bytes()
    for pc in result['pcs']:
        addr=int(pc['pc'],16)
        assert instruction_bytes(gd,addr)==instruction_bytes(dd,addr),pc
    save(run/'phase_pc_analysis.json',result)
    return result

def snapshot(out):
    snap=out/'preserved';snap.mkdir()
    for args,name in [(['git','rev-parse','HEAD'],'HEAD.txt'),(['git','status','--porcelain=v1','-uall'],'git-status.txt'),
                      (['git','diff','--binary','HEAD'],'working-tree.patch'),(['ps','-eo','pid,ppid,etime,args'],'processes.txt')]:
        (snap/name).write_bytes(subprocess.check_output(args,cwd=ROOT))
    with tarfile.open(snap/'source-state.tar.gz','w:gz') as tar:
        for name in ('gem5/src','gem5/configs','scripts','benchmarks','AGENTS.md'):
            tar.add(ROOT/name,arcname=name,filter=lambda info: None if '/__pycache__/' in info.name else info)
    paths=[BASE/'build/gem5.opt',GUEST,DEBUG,Path(__file__).resolve()]
    for t in (1024,16384):
        paths.append(reference(t))
        for arch in ('cache','spm'):
            p=prior(arch,t)
            paths += [p/n for n in ('accepted.json','command.json','config.ini','stats.txt')]
            cmd=json.loads((p/'command.json').read_text());paths.append(Path(cmd[cmd.index('--cmd')+1]))
    for p in set(paths):
        dest=snap/'artifacts'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    shutil.copy2(SWEEP/'status.json',snap/'sequencer-status-at-start.json')
    manifest={str(p.relative_to(snap)):sha(p) for p in snap.rglob('*') if p.is_file()}
    save(snap/'manifest.json',manifest)
    assert all(sha(snap/p)==h for p,h in manifest.items())

def report(out):
    rows=[]
    for p in sorted((out/'runs').glob('*/accepted.json')):
        row=json.loads(p.read_text());t,n,m,patch=row['case']
        control=SWEEP/'runs'/f'spm_t{t}_seq{n}'/'accepted.json'
        if not control.exists():
            if n!=16:continue
            control=prior('spm',t)/'accepted.json'
        base=json.loads(control.read_text())['counters']['roi_us'];us=row['counters']['roi_us']
        rows.append(dict(T=t,entries=n,mask=m,patch=patch,roi_us=us,control_us=base,
                         saved_us=base-us,saved_percent=100*(base-us)/base))
    save(out/'effects.json',rows)
    lines=['# FA diagnostic interventions','',
      'Positive saved time means faster than the same-capacity unmodified SPM control.',
      'Effects overlap: do not add single-factor savings. Separate backing adds storage; service changes are composite.',
      'Only accepted, independently checked outputs appear below. This is not a claim of complete causal closure.','',
      '| T | Entries | Mask | Stride patch | Runtime us | Saved us | Saved % |',
      '|---:|---:|---:|:---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['T']} | {r['entries']} | {r['mask']} | {r['patch']} | {r['roi_us']:.4f} | {r['saved_us']:.4f} | {r['saved_percent']:.3f} |")
    lines += ['', 'Mask bits: 1 = copy service/ports; 2 = SPM-load service/ports; 8 = separate backing.',
              'Detailed instruction-verified phase/PC and message/route accounting: each run/phase_pc_analysis.json.',
              'ROB-head attribution identifies where waiting appears, not which hardware resource independently caused each cycle.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    out=Path(sys.argv[1]).resolve();out.mkdir();(out/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='snapshot',completed=[]); lock=threading.Lock();stop=threading.Event()
    def status(stage):
        with lock:
            state.update(stage=stage,updated=time.time());save(out/'status.json',state)
        print(stage,flush=True)
    status('snapshot')
    try:
        snapshot(out)
        patched=out/'spm.stride128.guest';patched_debug=out/'spm.stride128.debug'
        proof=patch_stride(GUEST,patched);patch_stride(DEBUG,patched_debug)
        # Archived assembly computes the stack value as d*2; target fixes d=64.
        asm=(DEBUG.parent/'guest.enc.s').read_text()
        assert '\tlsl\tx1, x28, 1\n\tstr\tx1, [sp, 488]' in asm
        proof['scope']='Only d64; independent full-output checks required before timing acceptance.'
        save(out/'stride_patch.json',proof)
        frozen_paths=[Path(__file__),GUEST,DEBUG,patched,patched_debug,BASE/'build/gem5.opt']
        frozen_paths+=list((ROOT/'gem5/configs').rglob('*.py'))
        frozen_paths+=list((ROOT/'scripts').glob('*fa*py'))
        frozen_paths += [HERE/n for n in ('run_causal.py','analyze_attribution.py','analyze_traffic.py','validate_attribution.py','fa_independent_reference.py','fa_export_patch.py')]
        frozen_paths += [reference(t) for t in (1024,16384)]
        hashes={str(p):sha(p) for p in frozen_paths};save(out/'inputs.json',hashes)
        def frozen():
            for p,h in hashes.items():assert sha(p)==h,('changed input',p)
        gates=[(1024,n,m,False) for n in (16,128) for m in (0,1,2,8,3,9,10,11)]
        gates += [(1024,n,0,True) for n in (16,128)]
        # Fixed target plan avoids selecting only small-T winners. Single factors
        # first, then their combined effect and admission interaction.
        targets=[(16384,16,0,True),(16384,16,8,False),(16384,16,1,False),
                 (16384,16,2,False),(16384,16,11,False),(16384,128,11,False),
                 (16384,128,0,True)]
        save(out/'plan.json',dict(gates=gates,targets=targets,max_simulators=2,
          masks={'1':'copy request/response delay and store ports','2':'SPMLD request delay and load ports',
                 '8':'separate SPM backing: adds capacity and avoids claim eviction, NOT area matched'},
          shared_service_detail='Any service intervention also enables existing FIFO arrival clamp; effects are composite.',
          controls='Reuse independently accepted original and ongoing sequencer controls; rerun T1024 no-op endpoints exactly.',
          limits='Multi-head traced setup only. Finite interventions do not remove all delays. PC head cycles are attribution, not independent causal stall fractions. No claim of exhaustive bottleneck identification.',
          original_sweep=str(SWEEP),timeout_per_simulator_hours=30))
        status('analyzing_existing_controls')
        (out/'control_analysis').mkdir()
        for t in (1024,16384):
            for arch in ('cache','spm'):
                for n in ((16,32,64,128) if t==1024 else (16,)):
                    run=SWEEP/'runs'/f'{arch}_t{t}_seq{n}'
                    if not (run/'accepted.json').exists():run=prior(arch,t)
                    debug=DEBUG if arch=='spm' else BUILDS/'fce06cc784b2725b/guest.debug'
                    # Existing results remain read-only; save derived analysis here.
                    a.phase=phase;result=a.analyze(run,debug)
                    cmd=json.loads((run/'command.json').read_text())
                    gd=Path(cmd[cmd.index('--cmd')+1]).read_bytes();dd=debug.read_bytes()
                    for pc in result['pcs']:
                        addr=int(pc['pc'],16)
                        assert instruction_bytes(gd,addr)==instruction_bytes(dd,addr)
                    save(out/'control_analysis'/f'{arch}_t{t}_seq{n}.json',result)
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        def run(case):
            t,n,m,patch=case; frozen()
            p=out/'runs'/f'spm_t{t}_seq{n}_m{m}_patch{int(patch)}';p.mkdir()
            old=prior('spm',t);cmd=json.loads((old/'command.json').read_text())
            cmd=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in cmd]
            if patch:cmd[cmd.index('--cmd')+1]=str(patched)
            params=parameters(m)
            params.update({f'system.ruby.l0_cntrl{i}.sequencer.max_outstanding_requests':str(n) for i in range(8)})
            for k,v in params.items():cmd+=['-P',f'{k}={v}']
            save(p/'command.json',cmd);start=time.monotonic();proc=None
            try:
                with (p/'run.log').open('w') as log:
                    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                    while True:
                        elapsed=time.monotonic()-start
                        save(p/'progress.json',dict(pid=proc.pid,updated=time.time(),host_elapsed_s=elapsed,kernel_percentage=None))
                        if stop.is_set() or elapsed>30*3600:raise RuntimeError('Peer failed or 30h timeout')
                        if shutil.disk_usage(out).free<50*1024**3:raise RuntimeError('Less than 50GiB disk free')
                        try:rc=proc.wait(timeout=10);break
                        except subprocess.TimeoutExpired:pass
                save(p/'exit.json',dict(returncode=rc,host_elapsed_s=time.monotonic()-start));assert rc==0,(p,rc)
                frozen();changes=audit(old,p,params,patched if patch else None)
                output,counters,_=read(p,16)
                for k,v in dict(T=t,Hq=8,Hkv=1,d=64,cores=8,Br=16,Bc=512).items():assert output[k]==str(v)
                copies=8*math.ceil(t/256)*t*2
                assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==copies
                row=dict(path=str(p),case=case,output=output,counters=counters,config_changes=changes)
                verify_dump(row);row['validation']=check(p/'output.fp16',t,expected=np.load(reference(t)))
                original=json.loads((old/'accepted.json').read_text())
                assert output==original['output'],'Guest output changed'
                if m==0 and not patch:
                    control=json.loads((SWEEP/'runs'/f'spm_t{t}_seq{n}'/'accepted.json').read_text())
                    assert {k:v for k,v in counters.items() if k!='path'}=={k:v for k,v in control['counters'].items() if k!='path'}
                save(p/'accepted.json',row)
                # Serialize analysis because its module installs a phase callback.
                with lock:
                    analyze(p,patched_debug if patch else DEBUG)
                    state['completed'].append(dict(case=case,path=str(p),roi_us=counters['roi_us']))
                    save(out/'status.json',state)
                    report(out)
                print('accepted',case,counters['roi_us'],flush=True)
                return row
            except BaseException:
                stop.set()
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise
        for label,cases in [('gates',gates),('targets',targets)]:
            status(label)
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                # Bounded batches: failure never leaves queued jobs launching.
                for i in range(0,len(cases),2):
                    jobs=[pool.submit(run,c) for c in cases[i:i+2]]
                    for job in jobs:job.result()
            save(out/(label+'_complete.json'),state['completed'])
        status('complete')
    except BaseException as exc:
        stop.set();state['error']=repr(exc);status('failed');raise

if __name__=='__main__':main()
