"""Isolate the existing dense-lane test under the repository's GCC15 compiler.

Diagnostic only: no changes to active queues, guests, simulators, or source.
"""
import json
import os
from pathlib import Path
import subprocess
import time
from run_micro_fa_ruby_pilot import ROOT, sha, save

B = ROOT/'results/cacheflex_flash_attention'
OUT = B/'micro_conversion_gcc15_20260921'
CONTROL = B/'micro_paper_reproduction_v2_20260921'
COMPILER = B/'micro_toolchain_audit_20260918/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++'
SOURCE = ROOT/'benchmarks/cacheflex_flash_attention/test_dense_conversion.cpp'
ENV = dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib')


def main():
    OUT.mkdir()
    guest=OUT/'lane_probe'
    cmd=[str(COMPILER), '-O3', '-std=c++17', '-static', '-march=armv8.2-a+sve+fp16', str(SOURCE), '-o', str(guest)]
    save(OUT/'build_command.json',cmd)
    with (OUT/'build.log').open('w') as f:
        subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    version=subprocess.check_output([str(COMPILER),'--version'],text=True)
    objdump=COMPILER.with_name('aarch64-none-linux-gnu-objdump')
    with (OUT/'guest.disasm').open('w') as f:
        subprocess.run([str(objdump),'-d',str(guest)],stdout=f,check=True)
    save(OUT/'inputs.json',dict(source=str(SOURCE),source_sha256=sha(SOURCE),compiler=str(COMPILER),compiler_version=version,guest_sha256=sha(guest)))
    results=[]
    for model in ('classic','ruby'):
        for vl in (4,16):
            p=OUT/f'{model}_vl{vl}';p.mkdir()
            template=CONTROL/'runs'/f'{model}_cache_control_br32_bc128/command.json'
            cmd=json.loads(template.read_text())
            cmd=[f'--outdir={p}' if x.startswith('--outdir=') else
                 "system.cpu[0].workload[0].cmd=['conversion-probe']" if x.startswith('system.cpu[0].workload[0].cmd=') else
                 f'system.cpu[0].isa[0].sve_vl_se={vl}' if x.startswith('system.cpu[0].isa[0].sve_vl_se=') else
                 '--abs-max-tick=10000000000' if x.startswith('--abs-max-tick=') else x for x in cmd]
            cmd[cmd.index('-c' if model=='classic' else '--cmd')+1]=str(guest)
            save(p/'command.json',cmd)
            simulator_hash=sha(cmd[0]);start=time.monotonic()
            with (p/'run.log').open('w') as f:
                result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=ENV,timeout=120)
            log=(p/'run.log').read_text(errors='replace')
            row=dict(model=model,VL=vl,returncode=result.returncode,host_seconds=time.monotonic()-start,
                     evidence=[l for l in log.splitlines() if 'widen ' in l or 'LANE_CHECK' in l or 'panic:' in l or 'fatal:' in l],
                     simulator_sha256=simulator_hash)
            assert sha(cmd[0])==simulator_hash
            save(p/'result.json',row);results.append(row);save(OUT/'results.json',results)
            print(row,flush=True)
            assert result.returncode==0 and 'fixed_errors=0' in log, row


if __name__=='__main__':main()
