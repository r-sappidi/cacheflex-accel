"""Frozen, gated FA timing interventions; never changes shared runners.

Usage: run_causal.py NEW_OUT NEW_SIM OLD_SIM
Old simulator must be the LSQ-repaired, pre-ablation build. Default no-op
equivalence is checked before the factorial. Full factorial at T1024,
single-factor/all-on smoke at T128, then endpoint repetitions.
"""
import concurrent.futures as cf
import configparser
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from analyze_causal import FACTORS, report
from validate_attribution import read

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = ROOT/'results/cacheflex_flash_attention/spm_capacity_probe_20260909'

def parameters(mask):
    p = {}
    for i in range(8):
        if mask & 3:
            p[f'system.ruby.l0_cntrl{i}.bufferToL1.clamp_fifo_arrival'] = 'True'
        if mask & 1:
            p[f'system.ruby.l0_cntrl{i}.spm_copy_request_latency'] = '1'
            p[f'system.ruby.l1_cntrl{i}.spm_response_latency'] = '1'
            p[f'system.cpu[{i}].spmStorePorts'] = '8'
        if mask & 2:
            p[f'system.ruby.l0_cntrl{i}.spm_load_request_latency'] = '1'
            p[f'system.cpu[{i}].spmLoadPorts'] = '8'
        if mask & 4:
            for key in ('l1_request_latency','l1_response_latency'):
                p[f'system.ruby.l1_cntrl{i}.{key}'] = '1'
            for key in ('l2_request_latency','l2_response_latency'):
                p[f'system.ruby.l2_cntrl{i}.{key}'] = '1'
            p[f'system.ruby.dir_cntrl{i}.directory_latency'] = '1'
        if mask & 8:
            p[f'system.ruby.l1_cntrl{i}.cache.spm_separate_backing'] = 'True'
    return p

def config_key(param):
    import re
    section,key = re.sub(r'\[(\d+)\]', r'\1', param).rsplit('.',1)
    return section,key.lower()

def audit(reference, actual, mask, old=False):
    expected = {config_key(k):v.lower() for k,v in parameters(mask).items()}
    assert reference.sections() == actual.sections()
    seen = set()
    for section in reference.sections():
        keys = set(reference[section]) | set(actual[section])
        for key in keys:
            if key not in reference[section]:
                if old and key=='clamp_fifo_arrival' and actual[section][key]=='false':
                    continue
                assert old and section.startswith('system.ruby.l0_cntrl') and key in (
                    'spm_load_request_latency','spm_copy_request_latency') and actual[section][key]=='0'
                continue
            assert key in actual[section], (section,key,'missing')
            a,b = reference[section][key],actual[section][key]
            if section=='system.cpu0.workload' and key in ('cmd','executable'):
                continue
            if (section,key) in expected:
                assert b==expected[section,key],(section,key,b,expected[section,key])
                seen.add((section,key))
                continue
            if mask & 4 and section.startswith('system.ruby.network') and key in ('ni_flit_size','width'):
                assert b=='128',(section,key,b)
                continue
            if key=='num_spm_ways' and section.startswith('system.ruby.l1_cntrl') and section.endswith('.cache'):
                assert a in ('0','4') and b in ('0','4')
                continue
            assert a==b,(section,key,a,b)
    assert seen==set(expected),('unverified parameters',set(expected)-seen)

def main():
    out = Path(sys.argv[1]).resolve();out.mkdir()
    for src,name in ((sys.argv[2],'gem5.opt'),(sys.argv[3],'gem5.old.opt')):
        shutil.copy2(src,out/name)
    for v in ('cache','spm'):
        shutil.copy2(ARCHIVE/v,out/v)
    sources = [HERE/n for n in ('run_causal.py','analyze_causal.py','validate_attribution.py','analyze_traffic.py')]
    sources += [ROOT/'gem5/src/mem/ruby/protocol/MESI_Three_Level_SPM-L0cache.sm']
    sources += [ROOT/f'gem5/src/mem/ruby/network/MessageBuffer.{ext}' for ext in ('cc','hh','py')]
    for src in sources:
        shutil.copy2(src,out/src.name)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {str(p):digest(p) for p in sources+[out/'gem5.opt',out/'gem5.old.opt',out/'cache',out/'spm']}
    (out/'hashes.json').write_text(json.dumps(hashes,indent=2))
    experiment = dict(factors=list(FACTORS),tokens=1024,cores=8,vl=16,br=64,bc=128,
        copy_service='SPMCP request delay 2->1 cycles; bridge ack 6->1; SPM store ports 1->8. Does not remove copy instructions or coherent source fetches.',
        spm_load_service='SPMLD request delay 2->1 cycles; SPM load ports 2->8. Response remains 1 cycle. Does not bypass FIFO/controller queues or dependencies.',
        coherence_transport='Garnet width 128->1024 bits; private-L2 and LLC request/response enqueue delay 2->1; directory delay 6->1. Same intervention for cache/SPM; physical hop and arbitration latency remain.',
        capacity='Add four SPM-only ways (256 KiB/core), preserving all 512 KiB coherent capacity. Not area matched. No extra capacity when num_spm_ways=0.',
        limits='Bounded causal interventions, not complete removal of mechanisms. Shapley allocations depend on the chosen interventions. Residual remains unexplained; negative effects retained.',
        parameters={str(m):parameters(m) for m in range(16)})
    (out/'experiment.json').write_text(json.dumps(experiment,indent=2))
    rows=[];state=dict(stage='starting',completed=[],failed=None)
    def status(stage):
        state['stage']=stage;(out/'status.json').write_text(json.dumps(state,indent=2));print(stage,flush=True)
    def unchanged():
        for p,h in hashes.items():
            assert digest(Path(p))==h,('source or executable changed',p)
    def run(stage,v,t,mask,rep=0,old=False):
        name=f'{stage}_{v}_t{t}_m{mask:02d}_r{rep}'+('_old' if old else '')
        path=out/name;path.mkdir()
        log=ARCHIVE/f'{v}_t{t}_separate0_trace0/run.log'
        line=next(x for x in log.read_text().splitlines() if x.startswith('command line: '))
        args=shlex.split(line.removeprefix('command line: '))
        args[0]=str(out/('gem5.old.opt' if old else 'gem5.opt'))
        args=[f'--outdir={path}' if x.startswith('--outdir=') else x for x in args]
        args[args.index('--cmd')+1]=str(out/v)
        if mask & 4:
            args=[('--link-width-bits=1024' if x.startswith('--link-width-bits=') else x) for x in args]
        for key,value in parameters(mask).items():args+=['-P',key+'='+value]
        (path/'command.json').write_text(json.dumps(args,indent=2))
        with (path/'run.log').open('w') as stream:
            subprocess.run(args,env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'),
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
        output,counters,cfg=read(path,16)
        copies=8*(t//128)*64*4 if v=='spm' else 0
        assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==copies
        assert cfg['system.ruby.l1_cntrl0.cache']['num_spm_ways']==('4' if v=='spm' else '0')
        return dict(stage=stage,variant=v,t=t,mask=mask,rep=rep,old=old,path=str(path),output=output,counters=counters),cfg
    def batch(stage,cases,reference=None):
        unchanged();state['active_cases']=cases;status(stage)
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda c:run(stage,*c),cases))
        if reference:
            refrow,refcfg=reference
            for row,cfg in results:
                assert row['output']['hash']==refrow['output']['hash'],(row['path'],'output mismatch')
                audit(refcfg,cfg,row['mask'])
        rows.extend(r for r,_ in results)
        (out/'summary.json').write_text(json.dumps(rows,indent=2))
        state['completed'].append(stage);status(stage+'_complete')
        return results
    try:
        # Same guest bytes/paths on both simulators: new knobs default to no-op.
        controls=batch('default_equivalence',[(v,128,0,0,old) for v in ('cache','spm') for old in (True,False)])
        for i in (0,2):
            a,ca=controls[i];b,cb=controls[i+1]
            assert a['output']==b['output']
            for key in a['counters'].keys()-{'path'}:
                assert a['counters'][key]==b['counters'][key],('default timing/counter change',key)
            audit(ca,cb,0,old=True)
        assert controls[1][0]['output']['hash']==controls[3][0]['output']['hash']
        audit(controls[1][1],controls[3][1],0)
        batch('smoke',[(v,128,m) for m in (1,2,4,8,15) for v in ('cache','spm')],controls[1])
        defaults=batch('factorial',[(v,1024,0) for v in ('cache','spm')])
        assert defaults[0][0]['output']['hash']==defaults[1][0]['output']['hash']
        audit(defaults[0][1],defaults[1][1],0)
        # Paired by mask, two simulations at a time. All 2^4 combinations.
        for m in range(1,16):
            batch('factorial',[(v,1024,m) for v in ('cache','spm')],defaults[0])
        batch('endpoint_repeats',[(v,1024,m,r) for r in (1,2) for m in (0,15) for v in ('cache','spm')],defaults[0])
        report(out,rows)
        status('complete')
    except BaseException as error:
        state['failed']=repr(error);status('failed');raise

if __name__=='__main__':main()
