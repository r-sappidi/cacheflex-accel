"""Cache-only bounded search on the persistent-V experiment's exact hardware."""
import concurrent.futures as cf
import importlib.util,json,os,re,shutil,subprocess,sys,threading,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(HERE.parent/'kv_partition')]
from build_access_timing import sha,save
from run_access_timing import stats
from test_micro_timing import check_global_guard
from reference import reference,validate
from export_safe_elf import export_hash
spec=importlib.util.spec_from_file_location('llama_run',HERE.parent/'llama_cpu/run.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
BASE=ROOT/'results/cacheflex_flash_attention/cache_tuning_20260925'
OUT=BASE/os.environ.get('CACHEFLEX_CACHE_EXPERIMENT','experiment_export_safe_20260926')
KV=ROOT/'results/cacheflex_flash_attention/kv_partition_20260925'
OLD=KV/'experiment_v3'
REPAIR=ROOT/'results/cacheflex_flash_attention/export_repair_20260926'
GUESTS={m:REPAIR/'builds'/m/'guest' for m in ('original','qpack')}

def label(c):return f"{c['mode']}_t{c['t']}_br{c['br']}_bc{c['bc']}_g{c['g']}"
def main():
    OUT.mkdir();(OUT/'runs').mkdir();(OUT/'references').mkdir()
    shutil.copytree(OLD/'configs',OUT/'configs')
    cmd0=json.loads((shared.CONTROL/'command.json').read_text())
    paths=[*HERE.glob('*.py'),HERE.parent/'llama_cpu/run.py',HERE.parent/'kv_partition/reference.py',
           HERE.parent/'export_safe_elf.py',HERE.parent/'fa_export_refresh.S',
           *GUESTS.values(),Path(cmd0[0]),shared.CONTROL/'config.json',shared.CONTROL/'command.json',
           *list((OUT/'configs').rglob('*.py'))]
    frozen={str(p):sha(p) for p in paths};frozen.update(json.loads((BASE/'build/frozen.json').read_text()))
    save(OUT/'inputs.json',frozen)
    state=dict(pid=os.getpid(),stage='starting',active={},completed=[])
    lock=threading.RLock();halt=threading.Event();refs={}
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state)
    def unchanged():
        for p,h in frozen.items():assert sha(p)==h,('changed input',p)
    def getref(t,bc):
        key=(t,bc)
        if key not in refs:
            status(f'reference_t{t}_bc{bc}')
            old=OLD/'references'/f't{t}_bc{bc}_parts1.npy'
            refs[key]=np.load(old) if old.exists() else reference(t,bc,1)
            dest=OUT/'references'/f't{t}_bc{bc}.npy';np.save(dest,refs[key])
            save(dest.with_suffix('.json'),dict(sha256=sha(dest),source=str(old) if old.exists() else 'independent rounded reference'))
        return refs[key]
    def run(c):
        if halt.is_set():raise RuntimeError('cancelled before launch')
        name=label(c);path=OUT/'runs'/name;path.mkdir();proc=None;start=time.monotonic()
        try:
            unchanged();guest=GUESTS[c['mode']]
            args=list(map(str,(c['t'],8,c['br'],c['bc'],c['g'],1024,0,0)))
            cmd=cmd0.copy();cmd[1]='--outdir='+str(path);cmd[2]=str(OUT/'configs/deprecated/example/micro_timing_se.py')
            cmd[cmd.index('--cmd')+1]=str(guest);cmd[cmd.index('--options')+1]=' '.join(args)
            for i,s in enumerate(cmd):
                if s.startswith('system.cpu[0].workload[0].cmd='):cmd[i]=f"system.cpu[0].workload[0].cmd={['kv-fa']+args!r}"
            save(path/'command.json',cmd)
            with (path/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=shared.ENV,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][name]=dict(pid=proc.pid,case=c,path=str(path));status()
                while proc.poll() is None:
                    assert not halt.is_set(),'peer failure'
                    assert time.monotonic()-start<172800,'timeout'
                    assert shutil.disk_usage(OUT).free>40*2**30,'disk guard'
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            log=(path/'run.log').read_text()
            assert proc.returncode==0 and 'KV_ROI_END' in log and 'Simulated exit code not 0!' not in log,log[-2000:]
            config=shared.audit(path,guest,args);ss=stats(path)
            for s in ss:check_global_guard(s)
            s=ss[0];assert sum(s[f'system.ruby.l0_cntrl{i}.sequencer.m_spmcpRequestsIssued'] for i in range(8))==0
            export_hash(path/'output.fp16',log)
            math=validate(path/'output.fp16',refs[c['t'],c['bc']],c['t'],dense=c['t']==1024)
            h=1469598103934665603
            for b in np.fromfile(path/'output.fp16',dtype=np.uint16):h=((h^int(b))*1099511628211)&((1<<64)-1)
            assert f'hash={h:016x} nonfinite=0' in log
            unchanged();row=dict(case=c,roi_us=s['simTicks']/1e6,arithmetic=math,config=config,
                output_sha256=sha(path/'output.fp16'),host_seconds=time.monotonic()-start,path=str(path))
            save(path/'validated.json',row)
            with lock:state['active'].pop(name);state['completed'].append(row);status()
            return row
        except BaseException as exc:
            halt.set();save(path/'failed.json',dict(error=repr(exc)))
            with lock:state.setdefault('error',f'{name}: {exc!r}')
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            with lock:state['active'].pop(name,None);status()
            raise
    def batch(cases,stage):
        for c in cases:getref(c['t'],c['bc'])
        status(stage)
        with cf.ThreadPoolExecutor(max_workers=2) as pool:return list(pool.map(run,cases))
    def c(t,br,bc,g,mode='qpack'):return dict(t=t,br=br,bc=bc,g=g,mode=mode)
    try:
        status()
        for bc in (128,256,512):
            rows=batch([c(1024,16,bc,4,m) for m in ('original','qpack')],f'gate_bc{bc}')
            assert rows[0]['output_sha256']==rows[1]['output_sha256'],'Q packing not bit-preserving'
        save(OUT/'gates_passed.json',dict(cases=6,exact_original_hash=True,full_independent_math=True))
        screen=[c(4096,br,bc,qr//br) for br in (16,32) for bc in (128,256,512) for qr in (128,256)]
        screen += [c(4096,16,bc,32) for bc in (128,256)]
        rows=batch(screen,'t4096_screen')
        best=sorted(rows,key=lambda r:r['roi_us'])[:4]
        # Do not assume short-context ranking transfers: include larger reuse
        # groups and unchanged 128-query-row packing ablation at the true target.
        targets=[dict(r['case'],t=16384) for r in best]
        targets += [c(16384,16,128,8),c(16384,16,256,32),c(16384,16,128,32)]
        targets += [dict(best[0]['case'],t=16384,mode='original')]
        targets=list({label(x):x for x in targets}.values())
        save(OUT/'target_selection.json',dict(top_four=best,target_cases=targets,
            caveat='T4096 screening is not a substitute for T16384 measurements; fixed long-context corners retained'))
        rows=batch(targets,'t16384_confirmation');save(OUT/'winners.json',sorted(rows,key=lambda r:r['roi_us']))
        status('completed_validated')
    except BaseException as exc:state.setdefault('error',repr(exc));status('failed');raise
if __name__=='__main__':main()
