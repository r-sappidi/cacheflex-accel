"""Frozen same-ELF MICRO/Ruby latency probes, no change to other queues."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from build_access_timing import ROOT,sha,save
from run_access_timing import stats

BASE=ROOT/'results/cacheflex_flash_attention'
OUT=BASE/'timing_alignment_20260922'
UP=BASE/'micro_upstream_20260918'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib')

def build():
    p=OUT/'guest_v4';p.mkdir(exist_ok=False)
    src=ROOT/'benchmarks/spm_smoke/timing_probe.cpp'
    compiler=BASE/'micro_toolchain_audit_20260918/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++'
    commands=[
        [str(compiler),'-O3','-std=c++17','-static','-march=armv8.2-a+sve+fp16','-mbranch-protection=none','-DGEM5','-I'+str(ROOT/'gem5/include'),'-S',str(src),'-o',str(p/'guest.s')],
        ['python3',str(ROOT/'spm_tools/spm_compiler.py'),str(p/'guest.s'),str(p/'guest.enc.s')],
        [str(compiler),'-static',str(p/'guest.enc.s'),str(ROOT/'gem5/util/m5/build/arm64/abi/arm64/m5op.o'),'-o',str(p/'guest')]]
    for cmd in commands:subprocess.run(cmd,check=True)
    shutil.copy2(src,p/'timing_probe.cpp')
    save(p/'inputs.json',{str(f):sha(f) for f in [src,compiler,p/'guest',ROOT/'spm_tools/spm_compiler.py']})

def classic_options():
    source=(UP/'experiments/end2end/scripts/sweep_vl16.sh').read_text()
    opts=shlex.split(re.search(r'GEM5_CONFIG="(.*?)"',source,re.S)[1].replace('\\\n',' '))
    # Cold initialization is irrelevant; disable prefetch consistently for
    # isolated residency/latency calibration, not the final FA measurements.
    clean=[];i=0
    while i<len(opts):
        if '-hwp-type=' in opts[i]:i+=1;continue
        if opts[i]=='-P' and '.prefetcher.' in opts[i+1]:i+=2;continue
        clean.append(opts[i]);i+=1
    return clean

def ruby_options():
    # One active core, fixed8 LLC banks/router geometry from validated scaling.
    p=BASE/'micro_head_scaling_20260921/runs'
    template=next(x for x in p.rglob('command.json') if 'original' in str(x) and 'cache' in str(x) and '4096' in str(x))
    cmd=json.loads(template.read_text());return cmd,template

def run(model,mode,kib,width,label,extra,profile=False):
    p=OUT/label/f'{model}_{mode}_{kib}k_b{width}';p.mkdir(parents=True,exist_ok=False)
    guest=OUT/'guest_v4/guest';argv=[mode,str(kib),str(width),'16' if mode.startswith('spm') else ('16' if kib==4 else '2')]
    if model=='classic':
        build=OUT/'classic_build_v2'
        cmd=[str(build/'classic.opt'),f'--outdir={p}',str(build/'configs/deprecated/example/se.py'),*classic_options(),'-c',str(guest),'-o',' '.join(argv)]
    else:
        cmd,template=ruby_options()
        cmd=[f'--outdir={p}' if x.startswith('--outdir=') else
             'system.cpu[0].workload[0].cmd='+repr([str(guest),*argv]) if x.startswith('system.cpu[0].workload[0].cmd=') else x for x in cmd]
        cmd[0]=str(BASE/'demand_isolation_20260922/build/gem5.opt')
        if profile:
            cmd[2]=str(OUT/'configs/deprecated/example/micro_timing_se.py')
            cmd.append('--cacheflex-micro-timing')
        cmd[cmd.index('--cmd')+1]=str(guest)
        cmd=[x for x in cmd if x!='--enable-prefetch']
        cmd=[('--l1d_num_spm_ways=4' if mode.startswith('spm') else '--l1d_num_spm_ways=0') if x.startswith('--l1d_num_spm_ways=') else x for x in cmd]
        for value in extra:cmd+=['-P',value]
    save(p/'command.json',cmd)
    with (p/'run.log').open('w') as log:
        proc=subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    assert proc.returncode==0,(p,proc.returncode)
    log=(p/'run.log').read_text();assert 'TIMING_PROBE PASS' in log,(p,log[-1000:])
    count=int(re.search(r'loads=(\d+)',log)[1]);epochs=stats(p)
    assert len(epochs)>=3,(p,len(epochs))
    summary=[]
    for s in epochs[:3]:
        summary.append(dict(cycles_per_load=s['simTicks']/400/count,
            mean_load_to_use=s.get('system.cpu.spmLsq0.loadToUse::mean' if mode.startswith('spm') else 'system.cpu.regularLsq0.loadToUse::mean'),
            counters={k:v for k,v in s.items() if any(t in k for t in ['ReadSharedReq.misses','ReadSharedReq.hits','L0Cache_Controller.I.Load','L1Cache_Controller.I.Load','L1Cache_Controller.MM.Load','L2Cache_Controller.M.L1_GETS::total','SPMLD.latency_hist_seqr::mean'])}))
    save(p/'result.json',dict(passed=True,loads=count,epochs=summary));print(p.name,summary,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--build',action='store_true');ap.add_argument('--prepare-configs',action='store_true');ap.add_argument('--profile',action='store_true');ap.add_argument('--model',choices=['classic','ruby']);ap.add_argument('--label',default='initial');ap.add_argument('--extra',action='append',default=[]);ap.add_argument('--case',default='all');a=ap.parse_args()
    if a.build:build();return
    if a.prepare_configs:
        shutil.copytree(BASE/'micro_head_scaling_20260921/configs',OUT/'configs')
        shutil.copy2(ROOT/'scripts/micro_timing_se.py',OUT/'configs/deprecated/example/micro_timing_se.py')
        return
    cases=[('cache',4,64),('cache',256,64),('cache',2048,64),('spm',4,64),('spm',4,128),('spm',4,256)]
    if a.case!='all':cases=[tuple([v if i==0 else int(v) for i,v in enumerate(a.case.split(','))])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(run,a.model,*c,a.label,a.extra,a.profile) for c in cases]
        for t in tasks:t.result()

if __name__=='__main__':main()
