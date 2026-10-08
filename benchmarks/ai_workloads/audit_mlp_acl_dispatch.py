"""Verify passive profiling is non-perturbing and identify executed ACL kernels.

Usage: audit_mlp_acl_dispatch.py CONTROL_RUN PROFILE_RUN ELF
"""
from collections import defaultdict
from pathlib import Path
import json
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
from analyze_traffic import analyze

control,profile,elf=map(Path,sys.argv[1:4])
a,b=analyze(control),analyze(profile)
a.pop('path');b.pop('path')
assert a==b, {k:(a[k],b[k]) for k in a if a[k]!=b[k]}
for name in ('gate','up','silu','hidden','output'):
    assert (control/(name+'.bin')).read_bytes()==(profile/(name+'.bin')).read_bytes()
nm=ROOT/'tools/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-nm'
symbols=[]
for line in subprocess.check_output([str(nm),'-n','-S','-C','--defined-only',str(elf)],text=True).splitlines():
    w=line.split(maxsplit=3)
    if len(w)==4 and w[2] in ('T','t') and any(x in w[3] for x in ('arm_gemm::sve_', 'arm_compute::cpu::sve_', 'arm_compute::cpu::neon_', 'arm_compute::cpu::mul_F16')):
        symbols.append((int(w[0],16),int(w[1],16),w[3]))
counts=defaultdict(float)
stats=(profile/'stats.txt').read_text().split('End Simulation Statistics')[0]
for pc,n in re.findall(r'^system\.cpu\d+\.commit\.pcRetiredMicroops::(\d+)\s+(\S+)',stats,re.M):
    pc=int(pc)
    for start,size,name in symbols:
        if start<=pc<start+size:
            counts[name]+=float(n)
            break
assert any('sve_' in name and 'fp16' in name and n>0 for name,n in counts.items())
result=dict(control_counters_equal=True,outputs_equal=True,
            executed_kernel_microops=dict(sorted(counts.items(),key=lambda item:-item[1])))
(profile/'dispatch_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
