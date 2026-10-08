"""Small identical-binary Classic/Ruby correctness gate, no timing changes."""
from pathlib import Path
import json
import os
import subprocess
from run_micro_fa_ruby_pilot import numerical,sha,save,ROOT

B=ROOT/'results/cacheflex_flash_attention'
G=B/'micro_gcc15_corrected_gate_20260918'
O=B/'micro_gcc15_classic_gate_20260918'
O.mkdir()
assert json.loads((G/'status.json').read_text())['stage']=='gates_passed_target_configuration_pending'
rows=[]
for vl in (16,4):
    for arch in ('cache','spm'):
        p=O/f'{arch}_vl{vl}';p.mkdir()
        cmd=json.loads((B/'micro_classic_comparison_v7_20260918/runs'/f'{arch}_vl16_t320/command.json').read_text())
        cmd=[f'--outdir={p}' if x.startswith('--outdir=') else f'system.cpu[0].isa[0].sve_vl_se={vl}' if x.startswith('system.cpu[0].isa[0].sve_vl_se=') else x for x in cmd]
        cmd[cmd.index('-c')+1]=str(G/'builds'/f'{arch}_vl{vl}/guest')
        save(p/'command.json',cmd)
        with (p/'run.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'),check=True,timeout=180)
        check=numerical(p,320,128)
        assert check['errors']==check['nonfinite']==0,check
        other=G/'runs'/f'{arch}_vl{vl}_t320'
        equality={n:sha(p/f'{n}.fp16')==sha(other/f'{n}.fp16') for n in ('Q','K','V','O')}
        assert all(equality.values()),equality
        row=dict(arch=arch,vl=vl,numerical=check,bit_identical=equality)
        save(p/'accepted.json',row);rows.append(row);save(O/'results.json',rows)
        print(row,flush=True)
save(O/'passed.json',dict(cases=4,all_numerical_and_cross_model_hash_checks_passed=True))
