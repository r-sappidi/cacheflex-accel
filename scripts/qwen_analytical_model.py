"""Causal GQA loop/footprint model and bounded sensitivity-based selection.

Counts are exact for guest_v4. Ranking weights are explicit decision scenarios,
not fitted latency fractions. Prior noncausal timing coefficients are not reused.
"""
import argparse
import itertools
import json
from pathlib import Path
import statistics

ROOT=Path('/data2/rsappidi/cacheflex_mc')
SOURCE=ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928/experiment_v2'
OUT=ROOT/'results/cacheflex_l3_spm/qwen_analytical_20260928'


def describe(c):
    t,r,bc,br,hq=(c[k] for k in ('t','r','bc','br','hq'));arm=c['arm']
    if not (t>0 and t%r==0 and t%bc==0 and r%br==0 and br%16==0 and
            0<r<=512 and bc in (128,256,512,1024) and c['hkv']>0 and hq%c['hkv']==0):
        return dict(valid=False,reason='outside current checked guest geometry')
    groups=t//r;cores=[dict(tasks=0,pairs=0,updates=0,copies=0) for _ in range(8)]
    # Enumerate natural (head, query-group) coordinates, invert guest rotation
    # to recover core ownership. No reuse of the runner's expected-count helper.
    for head in range(hq):
        for group in range(groups):
            ordinal_group=(group-head)%groups
            core=(head*groups+ordinal_group)%8
            end=min(t,((group+1)*r+bc-1)//bc*bc) if c['causal'] else t
            x=cores[core];x['tasks']+=1;x['pairs']+=r*end
            x['updates']+=r*(end//bc);x['copies']+=end*2 if arm!='cache' else 0
    pairs=sum(x['pairs'] for x in cores);updates=sum(x['updates'] for x in cores)
    copies=sum(x['copies'] for x in cores);l3=arm=='l3';spm=arm!='cache'
    scratch=4*br*bc+32*bc+4*br  # scores, probabilities, packed P, alpha
    qstate=128*r+8*r+(0 if l3 else 256*r)
    vspm=128*bc if spm else 0
    # Reuse footprints, NOT measured resident bytes or set-associative miss counts.
    micro_reuse=128*br+8*br+(0 if l3 else 256*br)+scratch+128*bc*(1+int(not spm))
    coherent=qstate+scratch+128*bc*(1+int(not spm))
    useful=hq*t*(t+1)//2 if c['causal'] else hq*t*t
    return dict(valid=True,cores=cores,pairs=pairs,useful_pairs=useful,
        padded_pairs=pairs-useful,padding_pct=100*(pairs/useful-1),gemm_flops=256*pairs,
        max_core_pairs=max(x['pairs'] for x in cores),updates=updates,
        commands=updates if l3 else 0,initializations=hq*t if l3 else 0,
        transport_bytes=144*updates if l3 else 0,copies=copies,staged_bytes=64*copies,
        tasks=sum(x['tasks'] for x in cores),max_core_tasks=max(x['tasks'] for x in cores),
        max_core_staged_bytes=64*max(x['copies'] for x in cores),
        max_core_updates=max(x['updates'] for x in cores),
        imbalance=max(x['pairs'] for x in cores)*8/pairs,
        scratch_bytes=scratch,coherent_task_reuse_bytes=coherent,
        coherent_l2_capacity_bytes=512*1024-vspm,private_v_spm_bytes=vspm,
        micro_reuse_bytes=micro_reuse,l1_reuse_pressure=micro_reuse/(64*1024),
        l3_accumulator_bytes_per_bank=256*r if l3 else 0,
        note='Payloads/footprints are not cache misses, DRAM bytes, or additive runtime fractions')


def case(t,arm,r=256,bc=256,br=16):
    return dict(t=t,arm=arm,r=r,bc=bc,br=br,hq=14,hkv=2,causal=True,depth=4)


def structural_validation(rows):
    records=[]
    for row in rows:
        c=row['case'];m=describe(c);assert m['valid']
        assert (m['commands'],m['copies'])==(row['commands'],row['copies']),row['name']
        assert m['transport_bytes']==row['transport_bytes'],row['name']
        assert [x['tasks'] for x in m['cores']]==row['work']['tasks']
        assert [x['updates'] for x in m['cores']]==row['work']['updates']
        records.append(dict(name=row['name'],counts_match=True,roi_us=row['roi_us']))
    return records


def rank(t,arm):
    baseline=describe(case(t,arm));pool=[];deferred=[]
    for r,bc in itertools.product((64,128,256,512),(128,256,512,1024)):
        c=case(t,arm,r,bc);m=describe(c)
        reason=None
        if not m['valid']:reason=m['reason']
        elif bc==1024:reason='defer: larger tile regressed in L3-05; first budget targets nearer geometries'
        elif r==64:reason='defer: at least fourfold staging versus R256; retain in analytical space'
        elif m['imbalance']>1.15:reason='defer: busiest core exceeds mean causal work by >15% with frozen scheduler'
        elif m['coherent_task_reuse_bytes']>m['coherent_l2_capacity_bytes']:reason='defer: aggregate coherent footprint already exceeds remaining L2'
        row=dict(case=c,model=m)
        if reason:row['reason']=reason;deferred.append(row)
        else:pool.append(row)
    # A sensitivity ensemble replaces an underidentified nine-coefficient fit.
    # These nonnegative weights are ranking preferences, NOT timing attribution.
    scenarios=[]
    for staging,update,tasks,pressure in itertools.product((0,.02,.1,.25),(.01,.05,.15,.3),(0,.005,.02),(0,.05,.15)):
        scenarios.append(dict(compute=1-staging-update-tasks,staging=staging,
                              update=update,tasks=tasks,pressure=pressure))
    scores={id(x):[] for x in pool};ranks={id(x):[] for x in pool}
    for w in scenarios:
        values=[]
        for row in pool:
            m=row['model']
            score=(w['compute']*m['max_core_pairs']/baseline['max_core_pairs']+
                   w['staging']*m['max_core_staged_bytes']/baseline['max_core_staged_bytes']+
                   w['update']*m['max_core_updates']/baseline['max_core_updates']+
                   w['tasks']*m['max_core_tasks']/baseline['max_core_tasks']+
                   w['pressure']*(max(0,m['l1_reuse_pressure']-1)-max(0,baseline['l1_reuse_pressure']-1)))
            scores[id(row)].append(score);values.append((score,row))
        for row in pool:ranks[id(row)].append(1+sum(s< scores[id(row)][-1]-1e-12 for s,_ in values))
    for row in pool:
        ss=scores[id(row)];rr=ranks[id(row)];m=row['model']
        row.update(mean_rank=statistics.mean(rr),win_fraction=sum(x==1 for x in rr)/len(rr),
                   score_range=[min(ss),max(ss)],median_score=statistics.median(ss),
                   ratios=dict(max_core_compute=m['max_core_pairs']/baseline['max_core_pairs'],
                               copies=m['copies']/baseline['copies'],updates=m['updates']/baseline['updates']))
    pool.sort(key=lambda x:(x['mean_rank'],x['median_score'],x['case']['r'],x['case']['bc']))
    return dict(ranked=pool,deferred=deferred,scenarios=scenarios,baseline=baseline,
                caveat='Dimensionless sensitivity ranking, not predicted microseconds; weights are not fitted or causal')


def analyze(out=OUT):
    state=json.loads((SOURCE/'status.json').read_text());rows=state['completed']
    validation=structural_validation(rows)
    rankings={str(t):{a:rank(t,a) for a in ('private','l3')} for t in (2048,4096)}
    # Same two geometries for both arms: preserve a matched CPU-merge comparison.
    # Select one ranked candidate plus an R-changing diagnostic. The latter
    # isolates the grouping tradeoff absent from all current timing anchors.
    bygeom={}
    for arm in ('private','l3'):
        for x in rankings['2048'][arm]['ranked']:
            c=x['case'];g=(c['br'],c['bc'],c['r'])
            if g==(16,256,256):continue
            bygeom.setdefault(g,[]).append(x['mean_rank'])
    ordered=sorted((statistics.mean(v),g) for g,v in bygeom.items() if len(v)==2)
    chosen=[ordered[0]]
    first=chosen[0][1]
    if first[2]!=256 and first[1]==256:
        alternatives=[x for x in ordered[1:] if x[1][2]==256 and x[1][1]!=256]
        diagnostic='orthogonal Bc diagnostic: tile footprint versus command count'
    else:
        alternatives=[x for x in ordered[1:] if x[1][1]==256 and x[1][2]!=256]
        diagnostic='R-changing diagnostic: current anchors do not identify grouping cost'
    chosen.append((alternatives or ordered[1:])[0])
    geometries=[dict(br=g[0],bc=g[1],r=g[2],pooled_mean_rank=score,
                    reason='ranked sensitivity candidate' if i==0 else diagnostic)
                for i,(score,g) in enumerate(chosen)]
    selected=[dict(case(2048,a,g['r'],g['bc'],g['br']),selection_rank=g['pooled_mean_rank'])
              for g in geometries for a in ('private','l3')]
    out.mkdir(parents=True,exist_ok=True)
    result=dict(source=str(SOURCE),source_stage=state['stage'],accepted_count=len(rows),
                structural_validation=validation,rankings=rankings,selected=selected,
                selection_status='final' if state['stage']=='completed_validated' else 'provisional',
                ledger=['FA-09','FA-29','L3-05','L3-06','L3-07'],
                method='Exact causal/core loop counts + capacity screens + sensitivity ranks; no old-profile timing fit')
    (out/'model.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'selection.json').write_text(json.dumps(dict(selected=selected,geometries=geometries,
        source_stage=state['stage'],cache_frozen=True,max_new_simulators=2,
        budget='4 T1024 gates + 4 T2048 candidates; at most 2 winning transfers to T4096'),indent=2)+'\n')
    lines=['# Analytical Qwen kernel selection','',f"Source stage: {state['stage']}; {len(validation)} accepted count checks exact.",'',
      'FA-09 supplied traffic/footprint modeling; FA-29 used nonnegative fitting and bounded candidates.',
      'Its historical held-out error was 17.08%; old timing coefficients are not transferred here.',
      'Current observations do not vary R, so R sensitivity is not identified by measured timings.',
      'Scores below are explicit sensitivity scenarios, NOT predicted latency or speedup.',
      'Cache stays frozen. Br16 is retained from L3-05; no broad Br/R/Bc simulator sweep.','',
      '| T | R | Bc | Busiest-core compute ratio | Copy ratio | Update ratio | Core max/mean | L3 bank accumulator KiB |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in (2048,4096):
        b=describe(case(t,'l3'))
        for r,bc in ((128,128),(128,256),(256,128),(256,256),(256,512),(512,256)):
            m=describe(case(t,'l3',r,bc))
            lines.append(f"| {t} | {r} | {bc} | {m['max_core_pairs']/b['max_core_pairs']:.4f} | {m['copies']/b['copies']:.4f} | {m['updates']/b['updates']:.4f} | {m['imbalance']:.4f} | {m['l3_accumulator_bytes_per_bank']/1024:.0f} |")
    lines+=['','## Selected measurements','']
    for g in geometries:lines.append(f"- Br{g['br']}/Bc{g['bc']}/R{g['r']} on BOTH private-SPM and L3; pooled mean rank {g['pooled_mean_rank']:.3f}.")
    lines+=['','Wait for all original target controls; preserve and reuse their complete outputs/configs/references.',
      'Validate each candidate at T1024, then T2048. Transfer only a measured T2048 improvement to T4096.',
      'Include the inherited R256 winner if candidates regress. Compare strongest accepted private-SPM, not a weakened merge control.',
      'R512 may warrant a later T4096-only test: its T2048 imbalance screen does not prove it universally inferior.',
      'Total head count does not multiply each core\'s simultaneously live output tile. Shared K/V is not staged once across heads by this guest.',
      'Aggregate capacity checks omit address/set conflicts and overlap; logical staging is not DRAM traffic.',
      'Current hardware/guest/scheduler unchanged; no accumulator SRAM or queue enlargement.',
      'All new results require complete independent arithmetic, exact same-Bc outputs, counts/configuration/resource checks.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    x=analyze(a.out);print(json.dumps(dict(output=str(a.out),validated=len(x['structural_validation']),selected=x['selected']),indent=2))
