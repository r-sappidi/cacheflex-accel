"""Auditable V-only full-width FA traffic/storage model, not a cycle simulator.

Exact structural formulas are validated against counters. LLC misses require a
reuse model or measurements; fitted DRAM accounting is NOT independent prediction.
No simulation is launched and no existing experiment files are modified.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import argparse,csv,json,math,re
from collections import Counter
from analyze_attribution import instructions

@dataclass(frozen=True)
class Shape:
    t:int
    group:int=16
    br:int=16
    bc:int=512
    cores:int=8
    d:int=64

def model(s):
    if s.t<=0 or s.t%128 or s.br<=0 or s.br%16 or s.bc not in (128,256,512) or s.group<0:
        raise ValueError('Model supports positive T multiple128, Br multiple16, Bc128/256/512, nonnegative group')
    if s.d!=64 or s.cores!=8:
        raise ValueError('Only measured d64/eight-head/eight-core mapping is implemented')
    blocks=math.ceil(s.t/s.br);ib=min(s.group or blocks,blocks)
    g=min(ib*s.br,s.t);groups=math.ceil(blocks/ib)
    copies=s.cores*groups*s.t*s.d*2//64
    spm=s.bc*s.d*2
    qap=ib*s.br*s.d*2
    scratch=dict(qap=qap,scores=s.br*s.bc*2,probabilities=s.br*s.bc*2,
                 pv=16*s.d*2,packed_p=16*s.bc*2,output_acc=g*s.d*4,
                 max_and_sum=g*8)
    # Current V placement: two64B ways at each of Bc sets (1024 sets/cache).
    # These are claimed slots, not the four eligible ways all reserved upfront.
    total_scratch=sum(scratch.values());ways_used=2
    return dict(shape=asdict(s),query_rows_per_group=g,groups_per_core=groups,
      v_unique_bytes=s.t*s.d*2,spmcp_count=copies,staged_bytes=64*copies,
      spm_load_instructions=s.cores*(s.t//16)*s.t,
      fp16_fmla_instructions=s.cores*s.t*s.t,gemm_flops=4*s.cores*s.t*s.t*s.d,
      scratch_bytes_per_core=scratch,total_scratch_bytes_per_core=total_scratch,
      claimed_spm_bytes_per_core=spm,coherent_capacity_bytes_per_core=512*1024-spm,
      aggregate_private_footprint_fits=total_scratch<=512*1024-spm,
      scratch_fraction_of_coherent_capacity=total_scratch/(512*1024-spm),
      note_capacity='Aggregate necessary-screen only; not a residency prediction. Partial-set two-way claims, K/source/code/stack and conflicts not included.',
      main_array_bytes=(2*s.cores+4)*s.t*s.d*2,
      spm_ways_at_claimed_sets=ways_used)

def scalar_stats(path):
    stats={}
    for line in (path/'stats.txt').read_text().split('End Simulation Statistics')[0].splitlines():
        f=line.split()
        if len(f)>1:
            try:stats[f[0]]=float(f[1])
            except ValueError:pass
    return stats

def observe(path):
    row=json.loads((path/'accepted.json').read_text());stats=scalar_stats(path)
    args=json.loads((path/'command.json').read_text());guest=Path(args[args.index('--cmd')+1])
    ins=instructions(guest);fmla=0
    for k,v in stats.items():
        if re.fullmatch(r'system.cpu\d+.commit.pcRetiredMicroops::\d+',k):
            op=ins.get(int(k.split('::')[1]),{}).get('instruction','')
            if op.startswith('fmla') and '.h' in op:fmla+=int(v)
    messages=Counter()
    for k,v in stats.items():
        prefix='system.ruby.network.message_packets::'
        if k.startswith(prefix) and '__' in k:
            src,dst,_,msg=k[len(prefix):].split('__')
            if src.startswith('L2Cache') and dst.startswith('Directory'):messages[msg]+=int(v)
    return dict(path=str(path),t=int(row['output']['T']),counters=row['counters'],
      fmla=fmla,dram_gets=messages['req_GETS'],dram_silent=messages['req_GETS_SILENT'])

def verify(o,spm):
    m=model(Shape(o['t']));c=o['counters']
    expected=dict(spmcp=m['spmcp_count'] if spm else 0,
                  spm_loads=m['spm_load_instructions'] if spm else 0,
                  fmla=m['fp16_fmla_instructions'])
    measured=dict(spmcp=int(c['m_spmcpRequestsCompleted']),
                  spm_loads=int(c['spm_responses']),fmla=o['fmla'])
    assert expected==measured,(expected,measured)
    assert c['m_spmcpRequestsIssued']==expected['spmcp']
    # Observed request decomposition, a conservation check NOT a miss predictor.
    ordinary=o['dram_gets']*64;silent=o['dram_silent']*64
    # Injected directory requests and DRAM reads are different observation
    # points. Requests crossing reset/dump need not balance within the ROI.
    residual=ordinary+silent-c['dram_read_bytes']
    if o['t']==16384:assert residual==0
    return dict(independent_predictions=expected,measured=measured,
      structural_error_percent=0,ordinary_read_bytes=ordinary,silent_read_bytes=silent,
      inferred_silent_miss_fraction=o['dram_silent']/m['spmcp_count'] if spm else None,
      directory_request_minus_dram_bytes=residual)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];base=root/'results/cacheflex_flash_attention'
    paths=dict(cache1024=base/'pv_improvements_20260913/runs/3c716544869382b2',
      spm1024=base/'pv_improvements_20260913/runs/550744dbae7b1832',
      cache16384=base/'long_context_pv_20260914/runs/cache_t16384',
      spm16384=base/'long_context_pv_20260914/runs/spm_t16384')
    obs={k:observe(p) for k,p in paths.items()}
    validation={k:verify(o,k.startswith('spm')) for k,o in obs.items()}
    p=validation['spm16384']['inferred_silent_miss_fraction']
    psmall=validation['spm1024']['inferred_silent_miss_fraction']
    rows=[]
    for bc in (128,256,512):
        for group in (4,8,16,32,64,128,256,512,0):
            m=model(Shape(16384,group=group,bc=bc));staged=m['staged_bytes']
            rows.append(dict(group=group,bc=bc,G=m['query_rows_per_group'],
              copies=m['spmcp_count'],staged_MiB=staged/2**20,
              scratch_KiB=m['total_scratch_bytes_per_core']/1024,
              output_acc_KiB=m['scratch_bytes_per_core']['output_acc']/1024,
              spm_KiB=m['claimed_spm_bytes_per_core']/1024,
              coherent_KiB=m['coherent_capacity_bytes_per_core']/1024,
              scratch_fits=m['aggregate_private_footprint_fits'],
              silent_MiB_if_same_miss_fraction=staged*p/2**20,
              silent_MiB_min=0,silent_MiB_max=staged/2**20))
    timing={}
    for a in ('cache','spm'):
        predicted=obs[a+'1024']['counters']['roi_us']*256
        actual=obs[a+'16384']['counters']['roi_us']
        timing[a]=dict(quadratic_prediction_us=predicted,measured_us=actual,
                        error_percent=100*(predicted/actual-1))
    result=dict(validation=validation,observations=obs,model_t16384=model(Shape(16384)),
      screening=rows,quadratic_timing_crosscheck=timing,
      miss_fraction_transfer=dict(t1024=psmall,t16384=p,
          predicted_t16384_silent_bytes_from_t1024=psmall*model(Shape(16384))['staged_bytes']),
      limits=['No calibrated-to-T16K numbers labeled independent validation',
              'No bank-aware eviction or overlap/critical-path model',
              'Same-miss-fraction scenarios are not predicted speedups',
              'No retention-run outcomes consumed; available for later validation'])
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'model.json').write_text(json.dumps(result,indent=2))
    with (args.out/'screening.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# V-only full-width FA analytical model','',
      'Structural validation uses T1024 and T16384 cache/SPM runs, first ROI stats only.',
      'No hardware/guest changes or new simulations. Numerical acceptance already passed.',
      '', '| T | Predicted/measured SPMCPs | Predicted/measured SPM load instructions/responses | Predicted/measured FP16 FMLAs (each architecture) |',
      '|---|---:|---:|---:|']
    for t in (1024,16384):
        e=validation['spm'+str(t)]['independent_predictions']
        lines.append(f"| {t} | {e['spmcp']:,} | {e['spm_loads']:,} | {e['fmla']:,} |")
    lines += ['', 'All structural predictions match exactly (0% error), without fitting.',
      'SPM response counts match the full-width PV logical vector loads; these are NOT 64B line counts.',
      '', '## DRAM accounting versus prediction','',
      f'T16K measured silent miss fraction is {p:.8%}. Multiplying it by predicted staging bytes reproduces silent DRAM traffic, but this is calibration, NOT independent accuracy.',
      f'T1024 fraction is {psmall:.8%}. Transferring it predicts {result["miss_fraction_transfer"]["predicted_t16384_silent_bytes_from_t1024"]:.0f} silent bytes at T16K, versus measured {validation["spm16384"]["silent_read_bytes"]:,}: a failed cross-size prediction.',
      'At T16K ordinary and silent directory request counts times64 reproduce measured DRAM bytes exactly. At T1024 cache balances exactly; SPM has128B more injected requests than DRAM bytes. These are different observation points with possible in-flight boundary requests; no forced conservation fit was applied. No claim that capacity alone predicts misses.',
      '', '## Group-size screen at T16K, Bc512','',
      '| Query rows G | Staged MiB | Oacc KiB/core | Total scratch KiB/core | Aggregate scratch fits remaining coherent capacity? |',
      '|---|---:|---:|---:|---|']
    for r in rows:
        if r['bc']==512:lines.append(f"| {r['G']} | {r['staged_MiB']:.0f} | {r['output_acc_KiB']:.0f} | {r['scratch_KiB']:.0f} | {r['scratch_fits']} |")
    lines += ['', 'G512/G1024 are targeted candidates: 2x/4x fewer copies, but more coherent state. G1024 leaves only6KiB aggregate headroom at Bc512, before K/code/stack; G512 is a more conservative candidate. No optimality claim.',
      'Bc128/256 variants reduce scratch and claimed SPM bytes without changing whole-V staging volume at fixed G; they increase panel-loop overhead. See CSV for27 candidates.',
      'Aggregate capacity is not set residency: SPM claims two ways only at the occupied sets. This screen cannot resolve conflict misses or replacement interleaving.',
      '', '## Timing sanity check','']
    for a,r in timing.items():lines.append(f"{a}: T1024 quadratic extrapolation {r['quadratic_prediction_us']/1000:.3f}ms, actual {r['measured_us']/1000:.3f}ms, error {r['error_percent']:+.2f}%.")
    lines += ['', 'The simple timing model also predicts the wrong sign of the speedup. Do not use it to rank speedups. Memory/compute overlap and nonquadratic overhead require separate calibration.',
      '', '## Scope','',
      'The formulas apply to noncausal d64,8 query heads/1 KV head, one head/core, Br multiple16, full-width PV, V-only single-buffer, no fusion/interleaving/multicast/DMA.',
      'Persistent source arrays, temporary scratch and logical traffic are distinct. Scratch excludes allocator padding/metadata, stack, code, packed K/source working data and compiler spills.',
      'Next model improvement: address/set-aware trace or reuse-distance model for miss fractions. Validate new G choices with small targeted simulations, then one or two T16K finalists. Current model screens traffic/storage, not exact latency.']
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(args.out),validation=validation,timing=timing),indent=2))

if __name__=='__main__':main()
