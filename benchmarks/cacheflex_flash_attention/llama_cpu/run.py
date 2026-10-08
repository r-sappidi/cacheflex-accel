"""Matched hardware, full-output validated ggml FA comparison; two sims maximum."""
import concurrent.futures as cf
import json,os,re,shutil,subprocess,sys,threading,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(HERE.parent)]
from build_access_timing import sha,save
from run_access_timing import stats
from fa_headroom_profiles import leaves
from test_micro_timing import check_global_guard
from fa_independent_reference import inputs
from export_safe_elf import export_hash
BASE=ROOT/'results/cacheflex_flash_attention/llama_cpu_20260925'
OUT=BASE/os.environ.get('CACHEFLEX_LLAMA_EXPERIMENT','experiment_export_safe_20260926')
REPAIR=ROOT/'results/cacheflex_flash_attention/export_repair_20260926'
KV=ROOT/'results/cacheflex_flash_attention/kv_partition_20260925/experiment_v3'
CONTROL=KV/'runs/target_t4096_n8_q_cache_br16_g4_l2512'
MODES=('upstream_sve_ready','ported_sve_ready')
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')

def audit(path,guest,args):
    old=json.loads((CONTROL/'config.json').read_text().replace(str(CONTROL),'<OUT>'))
    new=json.loads((path/'config.json').read_text().replace(str(path),'<OUT>'))
    a,b=dict(leaves(old)),dict(leaves(new))
    delta={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    allowed=r'system.cpu\[\d+\].workload\[0\].(cmd\[\d+\]|executable)'
    assert all(re.fullmatch(allowed,k) for k in delta),delta
    proc=new['system']['cpu'][0]['workload'][0]
    assert proc['executable']==str(guest) and proc['cmd'][1:]==args
    return dict(matched_all_fields_except_guest=True,delta=delta)

def dense(t):
    # All outputs; O(chunk*T) memory, not a full T-by-T score allocation.
    q,k,v=inputs(t,True);q=q[0].astype(np.float64)
    k=k.astype(np.float64);v=v.astype(np.float64);out=np.empty_like(q)
    for i in range(0,t,128):
        z=q[i:i+128]@k.T*.125;z-=z.max(axis=1)[:,None];np.exp(z,out=z)
        out[i:i+128]=(z@v)/z.sum(axis=1)[:,None]
    return out

def numerical(path,ref):
    result={}
    for name,dtype in [('output.fp16',np.float16),('output.fp32',np.float32)]:
        x=np.fromfile(path/name,dtype=dtype).astype(np.float64).reshape(ref.shape)
        err=np.abs(x-ref);rel=float(np.linalg.norm(err)/np.linalg.norm(ref))
        row=dict(elements=x.size,nonfinite=int((~np.isfinite(x)).sum()),relative_l2=rel,
            max_abs=float(err.max()),errors=int((err>2e-6+.01*np.abs(ref)).sum()))
        assert row['nonfinite']==row['errors']==0 and rel<.01,row
        result[name]=row
    return result

def main():
    OUT.mkdir();(OUT/'runs').mkdir();(OUT/'references').mkdir()
    shutil.copytree(KV/'configs',OUT/'configs')
    state=dict(pid=os.getpid(),stage='waiting_build',active={},completed=[])
    lock=threading.RLock();halt=threading.Event()
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state)
    status()
    while not all((BASE/m/'frozen.json').exists() for m in MODES):time.sleep(5)
    frozen={}
    for m in MODES:frozen.update(json.loads((BASE/m/'frozen.json').read_text()))
    cmd0=json.loads((CONTROL/'command.json').read_text())
    paths=[Path(cmd0[0]),CONTROL/'command.json',CONTROL/'config.json',*HERE.glob('*.py'),*HERE.glob('*.cpp'),HERE/'sve_gemm.inc',
           HERE.parent/'export_safe_elf.py',HERE.parent/'fa_export_refresh.S',
           *[REPAIR/'builds'/m/'guest' for m in MODES],
           *list((OUT/'configs').rglob('*.py'))]
    frozen.update({str(p):sha(p) for p in paths});save(OUT/'inputs.json',frozen)
    def unchanged():
        for p,h in frozen.items():assert sha(p)==h,('Changed input',p)
    def run(mode,t,ref):
        if halt.is_set():raise RuntimeError('cancelled before launch')
        name=f'{mode}_t{t}';path=OUT/'runs'/name;path.mkdir();proc=None;start=time.monotonic()
        try:
            unchanged();guest=REPAIR/'builds'/mode/'guest';args=[str(t),'8'];cmd=cmd0.copy()
            cmd[1]='--outdir='+str(path);cmd[2]=str(OUT/'configs/deprecated/example/micro_timing_se.py')
            cmd[cmd.index('--cmd')+1]=str(guest);cmd[cmd.index('--options')+1]=' '.join(args)
            for i,s in enumerate(cmd):
                if s.startswith('system.cpu[0].workload[0].cmd='):cmd[i]=f"system.cpu[0].workload[0].cmd={['llama-fa']+args!r}"
            save(path/'command.json',cmd)
            with (path/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][name]=dict(pid=proc.pid,path=str(path));status()
                while proc.poll() is None:
                    assert not halt.is_set(),'Peer failure'
                    assert time.monotonic()-start<172800,'48h timeout'
                    assert shutil.disk_usage(OUT).free>40*2**30,'Disk guard'
                    save(path/'progress.json',dict(stage='simulating',pid=proc.pid,host_seconds=time.monotonic()-start))
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            text=(path/'run.log').read_text()
            assert proc.returncode==0 and 'LLAMA_ROI_END status=0' in text and 'Simulated exit code not 0!' not in text,text[-2000:]
            config=audit(path,guest,args);ss=stats(path);assert len(ss)>=1
            for s in ss:check_global_guard(s)
            s=ss[0]
            assert sum(s[f'system.ruby.l0_cntrl{i}.sequencer.m_spmcpRequestsIssued'] for i in range(8))==0
            export_hash(path/'output.fp16',text)
            math=numerical(path,ref)
            h=1469598103934665603
            for b in np.fromfile(path/'output.fp16',dtype=np.uint16):h=((h^int(b))*1099511628211)&((1<<64)-1)
            assert f'hash={h:016x} nonfinite=0' in text
            # Real retired-PC evidence, not just finding SVE somewhere in ELF.
            dis=(BASE/mode/'guest.dis').read_text();pcs={}
            for line in dis.splitlines():
                m=re.match(r'\s*([0-9a-f]+):\s+[0-9a-f]+\s+(.*)',line)
                if m and re.search(r'\bz\d+\.',m[2]):pcs[int(m[1],16)]=m[2]
            retired=[]
            for k,v in s.items():
                m=re.search(r'pcRetiredMicroops::(\d+)$',k)
                if m and int(m[1]) in pcs and v:retired.append(dict(counter=k,count=v,instruction=pcs[int(m[1])]))
            assert retired,'No dynamically retired SVE instructions found'
            save(path/'sve_retired.json',retired)
            unchanged();row=dict(mode=mode,t=t,roi_us=s['simTicks']/1e6,arithmetic=math,config=config,
                output_sha256=sha(path/'output.fp16'),host_seconds=time.monotonic()-start,
                precision='FP16 external input/output; FP32 ggml Q/GEMMs/state; conversions in ROI',sve_retired_pcs=len(retired))
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
    try:
        refs={}
        for t in (128,512,4096,16384):
            status(f'reference_t{t}');refs[t]=dense(t);np.save(OUT/'references'/f't{t}.npy',refs[t])
        def chain(mode):
            for t in (128,512,4096,16384):
                assert not halt.is_set()
                run(mode,t,refs[t])
                if t==512:save(OUT/(mode+'_gates_passed.json'),dict(cases=2,full_dense_validation=True))
        status('independent_gated_chains')
        with cf.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(chain,MODES))
        status('completed_validated')
    except BaseException as exc:state.setdefault('error',repr(exc));status('failed');raise
if __name__=='__main__':main()
