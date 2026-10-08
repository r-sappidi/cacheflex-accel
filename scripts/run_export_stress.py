"""Isolated functional-export bug reproducer; no FA ROI/model modifications."""
import concurrent.futures
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import sys
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_flash_attention'
OUT=BASE/'gcc15_export_diagnosis_20260919'
FULL='--full' in sys.argv
if FULL: OUT=BASE/'gcc15_export_diagnosis_full_20260919'
N=16777216 if FULL else 1048576
CC=BASE/'micro_toolchain_audit_20260918/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++'
SRC=ROOT/'benchmarks/cacheflex_flash_attention'
def main():
    OUT.mkdir(exist_ok=False)
    template=json.loads((BASE/'micro_gcc15_corrected_gate_20260918/runs/cache_vl16_t320/command.json').read_text())
    def run(kind):
        p=OUT/kind;p.mkdir()
        hook=SRC/('fa_export_hook.S' if kind=='old' else 'fa_export_refresh.S')
        subprocess.run([str(CC),'-O3','-static','-march=armv8.2-a+sve+fp16',str(SRC/'export_stress.cpp'),str(hook),'-o',str(p/'guest')],check=True)
        cmd=[x for x in template]
        cmd[cmd.index('--cmd')+1]=str(p/'guest')
        cmd=[f'--outdir={p}' if x.startswith('--outdir=') else
             f"system.cpu[0].workload[0].cmd=['export-stress','{N}']" if x.startswith('system.cpu[0].workload[0].cmd=') else
             '--l2_size=64kB' if not FULL and x.startswith('--l2_size=') else
             '--l1d_size=16kB' if not FULL and x.startswith('--l1d_size=') else
             '--l0d_size=16kB' if not FULL and x.startswith('--l0d_size=') else x for x in cmd]
        cmd[1:1]=['--debug-flags=PseudoInst','--debug-file=export.trace']
        (p/'command.json').write_text(json.dumps(cmd,indent=2))
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib')
        start=time.time()
        with (p/'run.log').open('w') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            (p/'progress.json').write_text(json.dumps(dict(pid=proc.pid,start=start)))
            code=proc.wait(timeout=1200)
        dump=p/'stress.bin';data=dump.read_bytes() if dump.exists() else b''
        expected=b''.join(struct.pack('<Q',i^0x123456789abcdef0) for i in range(N//8))
        row=dict(returncode=code,bytes=len(data),prefix_correct=data==expected[:len(data)],full_correct=data==expected,host_s=time.time()-start)
        (p/'result.json').write_text(json.dumps(row,indent=2))
        return kind,row
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        result=dict(pool.map(run,['old','refresh']))
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
