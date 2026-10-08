"""Numerical/functional gate for ORIGINAL MICRO FA drivers on one Ruby core.

No existing source/kernel/queue changes. Does not claim paper reproduction.
"""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'benchmarks/cacheflex_flash_attention';sys.path.insert(0,str(HERE))
from run_clean_controls import save,sha
from fa_independent_reference import gemm
MICRO=Path(os.environ.get('CACHEFLEX_MICRO_SOURCE',str(ROOT.parent/'cacheflex_micro'))).resolve()
SRC=MICRO/'experiments/end2end/src'

def build(out,arch,vl,correct=False):
    out.mkdir(parents=True)
    for f in SRC.glob('*.hpp'):shutil.copy2(f,out/f.name)
    source=(SRC/f'{arch}_flash_attn.cpp').read_text()
    source=source.replace('std::sqrtf(', 'std::sqrt(') # same float overload, portable spelling
    # Post-ROI export only. Do not apply conversion fixes to the original pilot.
    prefix='''extern "C" unsigned long cf_export_hook(void*,unsigned long,unsigned long,const char*);
static void cf_dump(const char* name,const void* data,unsigned long bytes) {
    unsigned long long h=1469598103934665603ULL;
    const unsigned short* p=(const unsigned short*)data;
    for(unsigned long i=0;i<bytes/2;i++) h=(h^p[i])*1099511628211ULL;
    printf("EXPORT %s bytes=%lu hash=%016llx\\n",name,bytes,h);
    // Ruby functional reads may not see a filename held only in dirty cache.
    // Materialize it with timing accesses, like the exported payload.
    volatile char export_name[64];
    unsigned j=0;
    do { if(j==63) std::abort(); export_name[j]=name[j]; } while(name[j++]);
    asm volatile("dsb sy" ::: "memory");
    if(cf_export_hook((void*)data,bytes,0,(const char*)export_name)!=bytes) std::abort();
}
'''
    pos=source.index('int main(');source=source[:pos]+prefix+'\n'+source[pos:]
    end=source.rfind('    return 0;');assert end>=0
    source=source[:end]+'''    cf_dump("Q.fp16",Q_buf.data(),(size_t)H_q*T*d*2);
    cf_dump("K.fp16",K_buf.data(),(size_t)H_kv*T*d*2);
    cf_dump("V.fp16",V_buf.data(),(size_t)H_kv*T*d*2);
    cf_dump("O.fp16",O_buf.data(),(size_t)H_q*T*d*2);
'''+source[end:]
    (out/'driver.cpp').write_text(source)
    conversion=f'-include "{HERE}/correct_fa_conversions.hpp"' if correct else ''
    script=f'''set -e
source "{ROOT}/setup_spm_env.sh" >/dev/null
"$CROSS_CXX" -O3 -g -std=c++17 -march=armv8.2-a+sve+fp16 -mbranch-protection=none -DGEM5 -DGEM5_SE -DVL_{vl} {conversion} -I"$M5_INCLUDE" -I"{out}" -I"{MICRO}/kernels/llama_bench_spm" -I"{MICRO}/kernels/attention_kernels_spm" -S "{out}/driver.cpp" -o "{out}/guest.s"
python3 "$SPM_COMPILER" "{out}/guest.s" "{out}/guest.enc.s"
"$CROSS_CXX" -c "{HERE}/fa_export_hook.S" -o "{out}/export.o"
"$CROSS_CXX" -static "{out}/guest.enc.s" "{out}/export.o" "$M5OP_OBJ" -lm -o "{out}/guest"
'''
    (out/'build.sh').write_text(script)
    with (out/'build.log').open('w') as f:subprocess.run(['bash',str(out/'build.sh')],stdout=f,stderr=subprocess.STDOUT,check=True)
    return out/'guest'

def numerical(p,t,bc):
    text=(p/'run.log').read_text(errors='replace')
    # Drivers round requested Bc to their microkernel NT (96 at VL4,
    # 128 at VL16). Reference the executed, not requested, recurrence.
    actual_bc=re.search(r'\bBc=(\d+)',text)
    assert actual_bc, 'missing executed tile geometry'
    bc=int(actual_bc.group(1))
    arrays={}
    for name in ('Q','K','V','O'):
        raw=np.fromfile(p/(name+'.fp16'),dtype=np.uint16);h=1469598103934665603
        for x in raw:h=((h^int(x))*1099511628211)&((1<<64)-1)
        assert f'EXPORT {name}.fp16 bytes={raw.size*2} hash={h:016x}' in text
        arrays[name]=raw.view(np.float16).reshape(t,64)
    q,k,v,o=(arrays[n] for n in ('Q','K','V','O'))
    expected=np.empty_like(o)
    for i in range(0,t,32):
        qq=q[i:i+32];n=len(qq);m=np.full(n,-1e30,np.float32);l=np.zeros(n,np.float32);acc=np.zeros((n,64),np.float32)
        for j in range(0,t,bc):
            s=gemm(qq,k[j:j+bc].T).astype(np.float32)*np.float32(.125)
            mn=np.maximum(m,s.max(axis=1));alpha=np.exp((m-mn).astype(np.float64)).astype(np.float32)
            z=np.exp((s-mn[:,None]).astype(np.float64)).astype(np.float32)
            l=(l*alpha+z.sum(axis=1,dtype=np.float32)).astype(np.float32)
            acc=(acc*alpha[:,None]+gemm(z.astype(np.float16),v[j:j+bc]).astype(np.float32)).astype(np.float32);m=mn
        expected[i:i+32]=(acc/l[:,None]).astype(np.float16)
    err=np.abs(o.astype(np.float64)-expected.astype(np.float64))
    return dict(effective_Bc=bc,elements=o.size,nonfinite=int((~np.isfinite(o)).sum()),errors=int((err>2e-6+.01*np.abs(expected.astype(np.float64))).sum()),maxabs=float(err.max()))

def main():
    t=int(os.environ.get('MICRO_GATE_T','256'))
    br=int(os.environ.get('MICRO_GATE_BR','64'))
    bc=int(os.environ.get('MICRO_GATE_BC','384'))
    out=Path(sys.argv[1]).resolve();out.mkdir();(out/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='building',completed=[])
    def status(stage):state.update(stage=stage,updated=time.time());save(out/'status.json',state);print(stage,flush=True)
    status('building')
    try:
        save(out/'upstream.json',dict(path=str(MICRO),commit=subprocess.check_output(['git','-C',str(MICRO),'rev-parse','HEAD'],text=True).strip(),remote=subprocess.check_output(['git','-C',str(MICRO),'remote','get-url','origin'],text=True).strip()))
        correct='--correct' in sys.argv
        guests={(a,v):build(out/'builds'/f'{a}_vl{v}',a,v,correct) for v in (4,16) for a in ('cache','spm')}
        save(out/'source_hashes.json',{str(p):sha(p) for p in list(SRC.glob('*.hpp'))+list(SRC.glob('*flash_attn.cpp'))+list(guests.values())})
        save(out/'plan.json',dict(scope='MICRO driver one-core Ruby port, single-head mathematical gate',gate_T=t,gate_Br=br,gate_Bc=bc,corrected_conversions=correct,target_T=4096,target_paper_speedups={'4':1.11,'16':1.34},
             unresolved='Paper winning tiles and head count not confirmed. Repo LLaMA sweep uses Hq32/Hkv8. corrected_conversions records whether dense-lane conversion repair is applied.',
             geometry=dict(L1_KiB=64,L2_KiB=512,LLC_MiB=4,cores=1,spm_ways=4),
             caution='Ruby coherence, queueing and prefetch implementation differs from Classic; cache-size matching is not complete timing-model matching. No target timing accepted if numerical gate fails.'))
        base=ROOT/'results/cacheflex_flash_attention/clean_controls_20260917/runs/spm_t512_rec0'
        original=json.loads((base/'command.json').read_text())
        def run(arch,vl):
            p=out/'runs'/f'{arch}_vl{vl}_t{t}';p.mkdir();cmd=[];i=0
            while i<len(original):
                x=original[i]
                if x=='-P':
                    v=original[i+1];i+=2
                    m=re.search(r'system.cpu\[(\d+)\]',v)
                    if m and int(m[1])>0:continue
                    if 'system.ruby.l2_cntrl' in v or 'cacheFlexPcProfile' in v or 'cacheflex_message_profile' in v:continue
                    v=v.replace('sve_vl_se=16',f'sve_vl_se={vl}').replace('spmLoadPorts=2','spmLoadPorts=1')
                    if v.startswith('system.cpu[0].workload[0].cmd='):v=f"system.cpu[0].workload[0].cmd=['micro-fa','{t}','1','1','64','1','{br}','{bc}']"
                    cmd+=['-P',v];continue
                replacements={'--num-cpus=':'1','--num-l2caches=':'1','--num-dirs=':'2','--mesh-rows=':'1','--l1d_size=':'512kB','--l1d_num_spm_ways=':'4','--l2_size=':'4MB','--mem-size=':'16GB'}
                if x.startswith('--outdir='):x=f'--outdir={p}'
                for prefix,val in replacements.items():
                    if x.startswith(prefix):x=prefix+val
                cmd.append(x);i+=1
            cmd[cmd.index('--cmd')+1]=str(guests[arch,vl])
            cmd+=['--mem-channels=2','--mem-type=DDR4_2400_8x8']
            save(p/'command.json',cmd)
            env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1')
            start=time.monotonic()
            with (p/'run.log').open('w') as f:
                proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
                save(p/'progress.json',dict(pid=proc.pid,started=time.time()))
                try:rc=proc.wait(timeout=3600)
                except subprocess.TimeoutExpired:proc.terminate();proc.wait();raise
            save(p/'exit.json',dict(returncode=rc,host_elapsed_s=time.monotonic()-start));assert rc==0,(p,rc)
            result=numerical(p,t,bc);save(p/'numerical.json',result)
            return dict(arch=arch,VL=vl,path=str(p),numerical=result)
        status('original_numerical_gates')
        for vl in (4,16):
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                for row in pool.map(lambda a:run(a,vl),('cache','spm')):
                    state['completed'].append(row);status('original_numerical_gates')
        failures=[r for r in state['completed'] if r['numerical']['errors'] or r['numerical']['nonfinite']]
        status('original_numerical_failure_target_not_started' if failures else 'gates_passed_target_configuration_pending')
    except BaseException as e:state['error']=repr(e);status('failed');raise

if __name__=='__main__':main()
