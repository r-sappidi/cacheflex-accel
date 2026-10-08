"""Continue the ordered study after the already-running stage2_r24.

All stages are bounded, sequential and correctness gated. At most four gem5
processes run concurrently. Source changes or any failed gate stop the queue.
This is a model-shape study with synthetic weights, not trained-model inference.
"""
from pathlib import Path
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
study=Path(sys.argv[1]).resolve();out=study/'ordered';out.mkdir()
sources=[HERE/n for n in ('mlp_study.cpp','mlp_reference.py','build_mlp_study.sh',
 'sweep_mlp_study.py','make_mlp_row_kernels.py','make_mlp_fp32_kernel.py',
 'audit_mlp_math.py','run_mlp_config.sh','mlp_acl_study.cpp','build_mlp_acl.sh',
 'run_mlp_acl_shapes.py','run_mlp_acl.py','validate_mlp_acl.py','run_mlp_ordered.py')]
sources += [ROOT.parent/'cacheflex_micro'/n for n in (
 'experiments/vl_length/tile_shape/v1_24x1vl.cpp',
 'experiments/vl_length/tile_shape/kernels_tile_spm.hpp',
 'kernels/llama_bench/kernels_sve.hpp','kernels/llama_bench/common.hpp',
 'kernels/llama_bench_spm/common_spm.hpp')]
sources += [ROOT/'benchmarks/cacheflex_spm_gemm/vl16_2m4n_driver.cpp',
            ROOT/'benchmarks/cacheflex_flash_attention/run_8core.sh',ROOT/'spm_tools/spm_compiler.py']
if os.environ.get('MLP_GEM5_BIN'):
 sources.append(Path(os.environ['MLP_GEM5_BIN']))
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
(out/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
state=dict(stage='waiting_for_stage2_r24',completed=[],failed=None)
def status(stage):
 state['stage']=stage;(out/'status.json').write_text(json.dumps(state,indent=2));print(stage,flush=True)
def unchanged():
 for p,digest in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,('source changed',p)
def command(name,args,env=None):
 unchanged();status(name)
 with (out/(name+'.host.log')).open('w') as log:
  subprocess.run(args,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',**(env or {})),stdout=log,stderr=subprocess.STDOUT,check=True)
 state['completed'].append(name);status(name+'_complete')
def build(rows,combined=0,acc32=0):
 name=f'build_r{rows}_combined{combined}_acc{acc32}';p=out/name
 if not p.exists():command(name,['bash',str(HERE/'build_mlp_study.sh'),str(p)],dict(MLP_ROWS=str(rows),MLP_COMBINED=str(combined),MLP_ACC32=str(acc32)))
 return p
def sweep(stage,name,b,cases,env=None):
 p=out/name;casefile=out/(name+'.cases.json');casefile.write_text(json.dumps(sorted(set(cases)),indent=2))
 command(name,[sys.executable,str(HERE/'sweep_mlp_study.py'),str(stage),str(p),str(b),str(casefile)],env)
 rows=json.loads((p/'summary.json').read_text())
 for row in rows:
  row['build']=str(b)
  row['combined']=int((b/'combined.txt').read_text()) if (b/'combined.txt').exists() else 0
 return rows
def winners(rows,t,d=512,f=1536):
 candidates=[r for r in rows if r['t']==t and r['d']==d and r['f']==f and r['warm']>0]
 confirmed=[r for r in candidates if r['repeats']>=3]
 if confirmed:candidates=confirmed
 return {v:min((r for r in candidates if r['variant']==v),key=lambda r:r['median_us']) for v in ('cache','spm')}

try:
 status('waiting_for_stage2_r24')
 wait_start=time.monotonic()
 while not (study/'stage2_r24/summary.json').exists():
  for log in (study/'stage2_r24').glob('*/run.log'):
   text=log.read_text()
   if any(token in text for token in ('panic:','fatal:',' FAIL','Simulated exit code not 0')):
    raise RuntimeError(f'upstream stage2 failure: {log}')
  if time.monotonic()-wait_start>24*3600:raise TimeoutError('stage2_r24 did not complete in 24h')
  time.sleep(10);unchanged()
 pool=json.loads((study/'stage2_r24/summary.json').read_text())
 for r in pool:r.update(build=str(study/'build_r24'),combined=0)
 # Smaller-row candidates: retune all grids at both architectures' winning KCs.
 for rows in (8,16):
  cases=set()
  for t in (64,128):
   kcs={r['kc'] for r in winners(pool,t).values()}
   for kc in kcs:
    for mp in (1,2,4,8):cases.add((t,512,1536,mp,kc,1,1,1))
  b=study/f'build_r{rows}'
  pool+=sweep(2,f'stage2_r{rows}',b,cases)
 (out/'stage2_results.json').write_text(json.dumps(pool,indent=2))
 # Retain all prior contenders. Test fused gate/up against fresh unfused controls.
 selected={}
 for t in (64,128):
  for winner in winners(pool,t).values():
   selected.setdefault(winner['tile_rows'],set()).add((t,512,1536,winner['mp'],winner['kc'],1,1,1))
 for rows,cases in selected.items():
  for combined in (0,1):
   pool+=sweep(3,f'stage3_r{rows}_combined{combined}',build(rows,combined),cases)
 (out/'stage3_results.json').write_text(json.dumps(pool,indent=2))
 # Model config dimensions, not pretrained tensors or whole-model execution.
 models=[dict(name='Qwen2.5-0.5B',d=896,f=4864,source='https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/b937cc0/config.json'),
         dict(name='Qwen3-0.6B',d=1024,f=3072,source='https://huggingface.co/Qwen/Qwen3-0.6B/blob/167b810/config.json')]
 (out/'model_shapes.json').write_text(json.dumps(dict(models=models,tokens=[1,16,64],note='FP16 synthetic inputs/weights; model configs specify BF16'),indent=2))
 model_rows=[];selected={}
 for winner in winners(pool,64).values():
  selected.setdefault((winner['tile_rows'],winner['combined']),set()).add((winner['mp'],winner['kc']))
 for (rows,combined),configs in selected.items():
  cases={(t,m['d'],m['f'],mp,kc,1,1,1) for m in models for t in (1,16,64) for mp,kc in configs}
  model_rows+=sweep(4,f'stage4_r{rows}_combined{combined}',build(rows,combined),cases)
 aclbuild=out/'acl_build'
 aclroot=ROOT/'results/ai_workloads/mlp_acl_20260909/upstream/source'
 command('acl_build',['bash',str(HERE/'build_mlp_acl.sh'),str(aclbuild)],dict(ACL_ROOT=str(aclroot),ACL_LIB=str(aclroot/'build/libarm_compute-static.a'),MLP_ACL_SOURCE=str(HERE/'mlp_acl_study.cpp')))
 aclcases=out/'acl_shapes.json';aclcases.write_text(json.dumps([[t,m['d'],m['f']] for m in models for t in (1,16,64)]))
 command('stage4_acl',[sys.executable,str(HERE/'run_mlp_acl_shapes.py'),str(out/'stage4_acl'),str(aclbuild),str(aclcases)])
 (out/'stage4_results.json').write_text(json.dumps(model_rows,indent=2))
 # Common numerical policy: <=1% relative L2 and <=2% peak-normalized error.
 # Retain the tighter implementation-reference check as well; failures halt.
 accuracy_rows=[]
 for seed,xdiv in ((20260909,128),(20260910,8),(20260911,4)):
  for (rows,combined),configs in selected.items():
   cases={(64,512,1536,mp,kc,1,1,1) for mp,kc in configs}
   accuracy_rows+=sweep(5,f'stage5_fp16_r{rows}_c{combined}_seed{seed}',build(rows,combined),cases,
                       dict(MLP_SEED=str(seed),MLP_XDIV=str(xdiv)))
  accuracy_rows+=sweep(5,f'stage5_fp32_seed{seed}',build(8,0,1),{(64,512,1536,mp,256,1,1,1) for mp in (1,2,4,8)},
                      dict(MLP_SEED=str(seed),MLP_XDIV=str(xdiv)))
 (out/'stage5_results.json').write_text(json.dumps(accuracy_rows,indent=2))
 # Hardware sensitivity: same selected software candidates in each paired config.
 hardware=[];selected_hw={}
 for winner in winners(pool,128).values():
  selected_hw.setdefault((winner['tile_rows'],winner['combined']),set()).add(winner['kc'])
 for vl in (16,4):
  for llc in (32,4):
   for cores in (1,2,4,8):
    for (rows,combined),kcs in selected_hw.items():
     cases={(128,512,1536,mp,kc,1,1,1) for kc in kcs for mp in (1,2,4,8) if mp<=cores}
     hardware+=sweep(6,f'stage6_vl{vl}_llc{llc}_cores{cores}_r{rows}_c{combined}',build(rows,combined),cases,
                     dict(MLP_VL=str(vl),MLP_LLC_MIB=str(llc),MLP_CPUS=str(cores)))
 (out/'stage6_results.json').write_text(json.dumps(hardware,indent=2))
 report=['# Ordered MLP study results','',
         'All queued stages finished. These are synthetic operator measurements, not trained-model latency.',
         'VL4 uses narrowed one-vector tiles: sensitivity only, not a fully tuned native VL4 best-to-best claim.','',
         '| T | Best cache us | Best SPM us | Cache/SPM |','|---:|---:|---:|---:|']
 for t in (64,128):
  w=winners(pool,t);a=w['cache']['median_us'];b=w['spm']['median_us'];report.append(f'| {t} | {a:.4f} | {b:.4f} | {a/b:.4f} |')
 report+=['','See per-stage JSON for individual epochs, spread, accuracy and hardware details.',
          'A best-to-best paper claim still requires reviewing convergence, model-shape retuning, and precision coverage; do not hide negative cases.']
 (out/'REPORT.md').write_text('\n'.join(report)+'\n');status('complete')
except BaseException as error:
 state['failed']=repr(error);status('failed');raise
