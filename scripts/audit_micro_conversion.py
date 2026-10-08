"""Independent dense FP64 attention and isolated SVE lane regression checks."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_flash_attention'
OUT=BASE/'micro_conversion_audit_20260918'
OUT.mkdir(exist_ok=False)
rows=[]
for corrected in (False,True):
    gate=BASE/('micro_upstream_corrected_gate_20260918' if corrected else 'micro_upstream_gate_20260918')
    for vl in (4,16):
        for arch in ('cache','spm'):
            run=gate/'runs'/f'{arch}_vl{vl}_t256'
            q,k,v,o=[np.fromfile(run/f'{name}.fp16',dtype=np.float16).reshape(256,64).astype(np.float64) for name in ('Q','K','V','O')]
            # No online recurrence, packing, FP16 GEMM helper or guest exp code.
            scores=q@k.T/np.sqrt(64.)
            p=np.exp(scores-scores.max(axis=1,keepdims=True))
            ref=(p/p.sum(axis=1,keepdims=True))@v
            error=np.abs(o-ref)
            rows.append(dict(corrected=corrected,vl=vl,arch=arch,max_abs=float(error.max()),relative_l2=float(np.linalg.norm(o-ref)/np.linalg.norm(ref)),nonfinite=int((~np.isfinite(o)).sum())))
(OUT/'dense_fp64.json').write_text(json.dumps(rows,indent=2))
build=f'source "{ROOT}/setup_spm_env.sh" >/dev/null\n"$CROSS_CXX" -O2 -static -march=armv8.2-a+sve+fp16 "{ROOT}/benchmarks/cacheflex_flash_attention/test_dense_conversion.cpp" -o "{OUT}/lane_probe"'
subprocess.run(['bash','-c',build],check=True)
for vl in (4,16):
    run=OUT/f'vl{vl}';run.mkdir()
    template=BASE/'micro_upstream_gate_20260918/runs'/f'cache_vl{vl}_t256/command.json'
    cmd=json.loads(template.read_text())
    cmd=[f'--outdir={run}' if x.startswith('--outdir=') else x for x in cmd]
    cmd[cmd.index('--cmd')+1]=str(OUT/'lane_probe')
    (run/'command.json').write_text(json.dumps(cmd,indent=2))
    with (run/'run.log').open('w') as f:
        subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180,env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
    log=(run/'run.log').read_text()
    assert 'fixed_errors=0' in log,log[-1000:]
    print([x for x in log.splitlines() if 'LANE_CHECK' in x],flush=True)
print(json.dumps(rows,indent=2))
