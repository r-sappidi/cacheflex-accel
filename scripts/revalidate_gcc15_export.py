"""Immutable-ROI export repair, small equivalence gates, then failed targets."""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import threading
import time
from run_micro_fa_ruby_pilot import ROOT, save, sha, numerical
from run_micro_fa_ruby_target import validate

B=ROOT/'results/cacheflex_flash_attention'
G=B/'micro_gcc15_corrected_gate_20260918'
C=B/'micro_gcc15_classic_gate_20260918'
OLD=B/'micro_gcc15_comparison_20260918'
OUT=B/'micro_gcc15_export_revalidation_v2_20260919'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
stop=threading.Event()

def patch(src,dst):
    data=src.read_bytes()
    symbols={w[2]:int(w[0],16) for l in subprocess.check_output(['aarch64-linux-gnu-nm',str(src)],text=True).splitlines() if len(w:=l.split())==3}
    start,end= symbols['m5_arm'],symbols['m5_sum']
    writer=symbols['cf_export_hook']
    dis=subprocess.check_output(['aarch64-linux-gnu-objdump','-d',str(src)],text=True)
    for line in dis.splitlines():
        if not re.match(r'^\s*[0-9a-f]+:',line) or '\tadrp\t' in line:continue
        for target in re.findall(r'\b([0-9a-f]+) <[^>]+>',line):
            assert not start<=int(target,16)<end, ('referenced trampoline storage',line)
    phoff=struct.unpack_from('<Q',data,32)[0];ents,n=struct.unpack_from('<HH',data,54)
    def offset(va,size):
        for i in range(n):
            typ,flags,off,addr,_,length,_,_=struct.unpack_from('<IIQQQQQQ',data,phoff+i*ents)
            if typ==1 and addr<=va and va+size<=addr+length:
                assert flags&1
                return off+va-addr
        raise ValueError(hex(va))
    # Verify the frozen exporter has exactly the known loop + fallback pseudo-ops.
    original_hook=offset(writer,176)
    op=struct.pack('<I',0xff4f0110)
    locations=[i for i in range(0,176,4) if data[original_hook+i:original_hook+i+4]==op]
    assert locations==[108,168], locations
    subprocess.run(['aarch64-linux-gnu-as',str(ROOT/'benchmarks/cacheflex_flash_attention/export_filename_each_chunk.S'),'-o',str(dst.with_suffix('.o'))],check=True)
    subprocess.run(['aarch64-linux-gnu-objcopy','-O','binary','--only-section=.text',str(dst.with_suffix('.o')),str(dst.with_suffix('.bin'))],check=True)
    hook=dst.with_suffix('.bin').read_bytes();assert len(hook)<=end-start
    a=offset(start,len(hook));b=original_hook+locations[0]
    delta=(start-(writer+locations[0]))//4
    assert -(1<<25)<=delta<(1<<25)
    patched=bytearray(data);patched[a:a+len(hook)]=hook
    patched[b:b+4]=struct.pack('<I',0x94000000|(delta&0x3ffffff))
    assert len(patched)==len(data)
    assert all(x==y or a<=i<a+len(hook) or b<=i<b+4 for i,(x,y) in enumerate(zip(data,patched)))
    dst.write_bytes(patched);dst.chmod(0o755)
    save(dst.with_suffix('.audit.json'),dict(original=str(src),original_sha256=sha(src),patched_sha256=sha(dst),ranges=[[a,a+len(hook)],[b,b+4]],invariant='Only unused m5 stubs and post-ROI export pseudo-op changed; all other ELF bytes/layout unchanged'))

def epochs(p):
    out=[]
    for block in (p/'stats.txt').read_text().split('---------- Begin Simulation Statistics ----------')[1:]:
        row={}
        for line in block.split('---------- End Simulation Statistics')[0].splitlines():
            w=line.split()
            if len(w)<2 or w[0].startswith('host'):continue
            try:float(w[1])
            except ValueError:continue
            row[w[0]]=w[1]
        out.append(row)
    return out

def main():
    assert json.loads((B/'gcc15_export_diagnosis_full_20260919/results.json').read_text())['refresh']['full_correct']
    OUT.mkdir();(OUT/'builds').mkdir();(OUT/'runs').mkdir()
    state=dict(pid=os.getpid(),completed=[])
    def status(stage):
        state.update(stage=stage,updated=time.time());save(OUT/'status.json',state);print(stage,flush=True)
    guests={}
    for vl in (16,4):
        for arch in ('cache','spm'):
            dst=OUT/'builds'/f'{arch}_vl{vl}';patch(G/'builds'/f'{arch}_vl{vl}/guest',dst);guests[arch,vl]=dst
    save(OUT/'plan.json',dict(gates='8 T320 gates, exact pre-export epochs and output hashes vs original, full independent reference',targets='Ruby T4096 VL16 pair; reuse accepted Classic VL16 controls; then four-way VL4',T=4096,Hq=32,Hkv=8,cores=1,VL=[16,4],Br=64,Bc=384,max_simulators=4,model_changes=False,original_results_preserved=True))
    def run(model,arch,vl,gate):
        label=f'{model}_{arch}_vl{vl}'+('_gate' if gate else '')
        p=OUT/'runs'/label;p.mkdir()
        oldgate=G/'runs'/f'{arch}_vl{vl}_t320' if model=='ruby' else C/f'{arch}_vl{vl}'
        cmd=json.loads((oldgate/'command.json').read_text())
        flag='--cmd' if model=='ruby' else '-c';cmd[cmd.index(flag)+1]=str(guests[arch,vl])
        cmd=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in cmd]
        if not gate:
            if model=='ruby':cmd=["system.cpu[0].workload[0].cmd=['micro-fa','4096','32','8','64','1','64','384']" if x.startswith('system.cpu[0].workload[0].cmd=') else x for x in cmd]
            else:cmd[cmd.index('-o')+1]='4096 32 8 64 1 64 384'
        if model=='classic':
            original_argv=json.loads((oldgate/'config.json').read_text())['system']['cpu'][0]['workload'][0]['cmd']
            argv=[original_argv[0]]+cmd[cmd.index('-o')+1].split()
            cmd+=['-P',f'system.cpu[0].workload[0].cmd={argv!r}']
        hashes={str(guests[arch,vl]):sha(guests[arch,vl]),cmd[0]:sha(cmd[0])}
        save(p/'command.json',cmd);save(p/'hashes.json',hashes)
        start=time.monotonic()
        with (p/'run.log').open('w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=ENV)
            try:
                while child.poll() is None:
                    save(p/'progress.json',dict(stage='simulating',pid=child.pid,host_elapsed_s=time.monotonic()-start,updated=time.time()))
                    if stop.is_set():raise RuntimeError('peer failed')
                    if time.monotonic()-start>48*3600:raise RuntimeError('timeout')
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=30)
        assert child.returncode==0,(label,child.returncode)
        assert all(sha(f)==h for f,h in hashes.items())
        save(p/'progress.json',dict(stage='validating',updated=time.time()))
        check=numerical(p,320,128) if gate else validate(p)
        assert check['errors']==check['nonfinite']==0,check
        old=oldgate if gate else OLD/'runs'/f'{model}_{arch}_vl{vl}'
        if (old/'stats.txt').exists():
            # Only explicit measured ROI epochs: final post-export epoch intentionally changes.
            # Cache's second stats block is final (warm run AND exporter),
            # not a separately delimited pre-export ROI.
            count=1
            before=epochs(old)[:count];after=epochs(p)[:count]
            assert len(before)==len(after)==count
            assert before==after,('ROI counter drift',label)
            save(p/'roi_equivalence.json',dict(passed=True,epochs=count,counters=sum(map(len,before)),original=str(old)))
        if gate:
            eq={n:sha(p/f'{n}.fp16')==sha(old/f'{n}.fp16') for n in ('Q','K','V','O')}
            assert all(eq.values());save(p/'output_equivalence.json',eq)
        row=dict(model=model,arch=arch,VL=vl,gate=gate,path=str(p),roi_us=float(epochs(p)[0]['simTicks'])/1e6,numerical=check)
        save(p/'accepted.json',row);save(p/'progress.json',dict(stage='accepted',updated=time.time()))
        return row
    def batch(cases):
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(run,*c) for c in cases]
            try:
                for future in cf.as_completed(futures):state['completed'].append(future.result());status(state['stage'])
            except BaseException:stop.set();raise
    try:
        status('small_equivalence_gates')
        batch([(m,a,v,True) for v in (16,4) for m in ('ruby','classic') for a in ('cache','spm')])
        save(OUT/'gates_passed.json',dict(passed=True,gates=8))
        for vl in (16,4):
            status(f't4096_vl{vl}')
            models=('ruby',) if vl==16 else ('ruby','classic')
            batch([(m,a,vl,False) for m in models for a in ('cache','spm')])
            times={}
            for model in ('ruby','classic'):
                times[model]={}
                for arch in ('cache','spm'):
                    p=(OLD if vl==16 and model=='classic' else OUT)/'runs'/f'{model}_{arch}_vl{vl}'
                    row=json.loads((p/'accepted.json').read_text());times[model][arch]=row['roi_us']
            for arch in ('cache','spm'):
                cp=(OLD if vl==16 else OUT)/'runs'/f'classic_{arch}_vl{vl}'
                rp=OUT/'runs'/f'ruby_{arch}_vl{vl}'
                eq={n:sha(cp/f'{n}.fp16')==sha(rp/f'{n}.fp16') for n in ('Q','K','V','O')}
                assert all(eq.values());save(OUT/f'cross_model_{arch}_vl{vl}.json',eq)
            save(OUT/f'summary_vl{vl}.json',dict(latency_us=times,speedup={m:t['cache']/t['spm'] for m,t in times.items()}))
        status('complete')
    except BaseException as e:state['error']=repr(e);status('failed');raise

if __name__=='__main__':main()
