"""Test upstream build flags/toolchain, preserving simulators and active guests."""
from pathlib import Path
import subprocess
import json
import os
import re
import hashlib

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_flash_attention'
OUT=BASE/'micro_toolchain_audit_20260918'
UP=BASE/'micro_upstream_20260918'
SRC=UP/'experiments/end2end/src'
TC15=OUT/'arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++'
TC11=ROOT/'tools/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++'
M5=ROOT/'gem5/util/m5/build/arm64/abi/arm64/m5op.o'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib')

def save(p,x):p.write_text(json.dumps(x,indent=2))
rows=[]
for version,compiler in ((15,TC15),(11,TC11)):
    for arch in ('cache','spm'):
        p=OUT/f'{arch}_gcc{version}';p.mkdir()
        common=[str(compiler),'-O3','-std=c++17','-march=armv8.2-a+sve+fp16','-static',f'-I{SRC}','-DGEM5','-DGEM5_SE',f'-I{UP}/gem5/include']
        if arch=='spm':common+=['-DVL_16','-mbranch-protection=none']
        src=SRC/f'{arch}_flash_attn.cpp'
        # Exactly upstream cache flags; SPM needs its documented assembly pass.
        def compile_source(source,logname):
            cmd=common+(['-S',str(source),'-o',str(p/'guest.s')] if arch=='spm' else [str(source),str(M5),'-lm','-o',str(p/'guest')])
            save(p/(logname+'.command.json'),cmd)
            with (p/(logname+'.log')).open('w') as f:return subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
        rc=compile_source(src,'exact')
        spelling_fix=False
        if rc:
            # Only an observed std::sqrtf namespace portability failure is repaired.
            log=(p/'exact.log').read_text()
            if 'sqrtf' not in log:raise RuntimeError(log)
            local=p/'driver.cpp';local.write_text(src.read_text().replace('std::sqrtf(', 'std::sqrt('))
            rc=compile_source(local,'portable_sqrt');spelling_fix=True
        assert rc==0,(p,rc)
        if arch=='spm':
            subprocess.run(['python3',str(UP/'gem5/cacheflex_sw/spm_compiler.py'),str(p/'guest.s'),str(p/'guest.enc.s')],check=True,stdout=subprocess.DEVNULL)
            subprocess.run([str(compiler),'-static',str(p/'guest.enc.s'),str(M5),'-lm','-o',str(p/'guest')],check=True)
        dump=subprocess.check_output([str(compiler).replace('g++','objdump'),'-d',str(p/'guest')],text=True)
        matches=[line for line in dump.splitlines() if re.search(r'\bfmov\s+h\d+,\s*w\d+',line)]
        (p/'fmov_hw.txt').write_text('\n'.join(matches))
        row=dict(arch=arch,gcc=version,sqrt_spelling_fix=spelling_fix,fmov_hw_count=len(matches),guest_sha256=hashlib.sha256((p/'guest').read_bytes()).hexdigest())
        # Run small workload on untouched upstream Classic with identical prior config.
        run=p/'classic';run.mkdir()
        cmd=json.loads((BASE/'micro_classic_comparison_v7_20260918/runs'/f'{arch}_vl16_t320/command.json').read_text())
        cmd=[f'--outdir={run}' if x.startswith('--outdir=') else x for x in cmd]
        cmd[cmd.index('-c')+1]=str(p/'guest')
        save(run/'command.json',cmd)
        with (run/'run.log').open('w') as f:
            try:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=ENV,timeout=180);row['returncode']=result.returncode
            except subprocess.TimeoutExpired:row['timeout']=True
        log=(run/'run.log').read_text(errors='replace')
        row['failure_lines']=[line for line in log.splitlines() if 'panic:' in line or 'fatal:' in line]
        rows.append(row);save(OUT/'results.json',rows);print(row,flush=True)
