"""Exact loop accounting and uncalibrated sensitivity ranking for Qwen FFN."""
import itertools
import json
import math
from pathlib import Path

ROOT=Path('/data2/rsappidi/cacheflex_mc')
BASE=ROOT/'results/ai_workloads/qwen_swiglu_20260928'
BASELINE=dict(mp=2,kc=128,g=0)


def describe(t,d,f,c):
    mp,kc,g=(c[k] for k in ('mp','kc','g'));np_=8//mp
    assert mp in (1,2,4,8) and kc in (128,256,512,1024)
    assert d%128==f%128==0 and g in (0,1)
    blocks=(t+7)//8
    stages=[(d,2*f),(f,d)] if g else [(d,f),(d,f),(f,d)]
    cores=[dict(flops=0,copies=0,a_pack_bytes=0,accumulator_bytes=0) for _ in range(8)]
    critical={k:0 for k in cores[0]};pressure=0;max_ap=0;ownership=[]
    for k,n in stages:
        stage=[]
        for i in range(8):
            mi,ni=divmod(i,np_)
            rows=8*(blocks*(mi+1)//mp-blocks*mi//mp)
            cols=128*((n//128)*(ni+1)//np_-(n//128)*ni//np_)
            x=dict(flops=2*rows*k*cols,
                   copies=4*k*(cols//128) if rows else 0,
                   a_pack_bytes=2*rows*k,
                   accumulator_bytes=4*rows*cols*(2*math.ceil(k/kc)-1))
            stage.append(x);ownership.append(dict(k=k,n=n,core=i,rows=rows,cols=cols))
            for key,value in x.items():cores[i][key]+=value
            # Working-set proxy: packed A cohort + current B and C panel.
            # Not a cache-miss estimate; C across panels streams separately.
            ap=2*rows*min(k,kc);max_ap=max(max_ap,ap)
            pressure=max(pressure,(ap+256*min(k,kc)+512*rows)/(512*1024))
        for key in critical:critical[key]+=max(x[key] for x in stage)
    totals={key:sum(x[key] for x in cores) for key in critical}
    assert totals['flops']==6*(blocks*8)*d*f
    assert totals['copies']==min(mp,blocks)*4*(3*d*f//128)
    return dict(shape=[t,d,f],case=c,cores=cores,ownership=ownership,
                totals=totals,critical=critical,spm_panel_bytes=256*min(kc,max(d,f)),
                a_cohort_max_bytes=max_ap,l2_pressure_proxy=pressure,
                l1_b_panel_ratio=256*min(kc,max(d,f))/(64*1024),
                flops_imbalance=critical['flops']/(totals['flops']/8),
                setup_bytes=6*d*f+(8*d*f if g else 0))


def analyze():
    BASE.mkdir(parents=True,exist_ok=True)
    configs=[dict(mp=m,kc=k,g=g) for m,k,g in itertools.product((1,2,4,8),(128,256,512,1024),(0,1))]
    rows=[describe(1024,896,4864,c) for c in configs]
    baseline=next(x for x in rows if x['case']==BASELINE)
    selected={};rankings={}
    for arm in ('cache','spm'):
        scenarios=[]
        for movement,acc,pressure in itertools.product((.05,.15,.30),(.05,.15,.30),(.02,.10,.25)):
            scores=[]
            for x in rows:
                c=x['critical'];b=baseline['critical']
                compute=c['flops']/b['flops']
                traffic=(c['copies']*64+c['a_pack_bytes'])/(b['copies']*64+b['a_pack_bytes'])
                accum=c['accumulator_bytes']/b['accumulator_bytes']
                capacity=max(0,x['l2_pressure_proxy']-1)/max(1,baseline['l2_pressure_proxy'])
                # Cache operand delivery competes for L1; SPM avoids that path.
                delivery=max(0,x['l1_b_panel_ratio']-1) if arm=='cache' else 0
                startup=x['setup_bytes']/baseline['setup_bytes']
                score=(1-movement-acc)*compute+movement*traffic+acc*accum+pressure*(capacity+.25*delivery)+.02*startup
                scores.append(score)
            order=sorted(range(len(rows)),key=lambda i:(scores[i],rows[i]['case']['kc']))
            scenarios.append(dict(scores=scores,order=order))
        ranked=[]
        for i,x in enumerate(rows):
            ranked.append(dict(case=x['case'],mean_rank=sum(s['order'].index(i) for s in scenarios)/len(scenarios),
                               wins=sum(s['order'][0]==i for s in scenarios),
                               score_range=[min(s['scores'][i] for s in scenarios),max(s['scores'][i] for s in scenarios)]))
        ranked.sort(key=lambda x:(x['mean_rank'],-x['wins']))
        # Equal budget: inherited control, two robust candidates and a mapping
        # diagnostic with a distinct MP. These are hypotheses, not fitted times.
        picks=[dict(BASELINE)]
        for x in ranked:
            if x['case'] not in picks:picks.append(x['case'])
            if len(picks)==3:break
        used={x['mp'] for x in picks}
        picks.append(next(x['case'] for x in ranked if x['case']['mp'] not in used))
        selected[arm]=picks;rankings[arm]=ranked
    result=dict(ledger=['AI-03','AI-04','AI-06','AI-08','AI-09','L3-06','L3-07'],
                shape=dict(d=896,f=4864,screen_t=1024,targets=[2048,4096]),
                scope='Existing FP16-storage/FP32-accumulation SwiGLU guest; no new arithmetic offload.',
                baseline=BASELINE,selected=selected,rankings=rankings,structures=rows,
                method='27 uncalibrated sensitivity scenarios per backend; equal four-case budget; scores are not predicted latency.',
                objective='Complete first-invocation ROI including packing, staging, all three GEMMs, SiLU-product and barriers.',
                repeat_policy='Two calls: first is primary; second reported separately as a repeat diagnostic, not stable warm median.')
    (BASE/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Qwen SwiGLU analytical selection','',result['method'],'',result['objective'],'',
           '| Arm | MP | KC | Combined | SPM panel KiB | Packed A max KiB | Critical FLOP imbalance |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for arm,cases in selected.items():
        for c in cases:
            x=describe(1024,896,4864,c)
            lines.append(f"| {arm} | {c['mp']} | {c['kc']} | {c['g']} | {x['spm_panel_bytes']/1024:g} | {x['a_cohort_max_bytes']/1024:g} | {x['flops_imbalance']:.4f} |")
    lines+=['','MP partitions token rows; 8/MP partitions output columns. The down projection has seven 128-column tiles, so N partitioning can leave uneven work.',
            'KC changes partial-accumulator traffic and staging footprint, but not total SPM copy bytes. Combined gate/up reduces duplicated A packing; it adds first-call concatenation and changes column ownership.',
            'All candidates use the existing eight-row kernel. L2/L1 pressure terms are heuristics, not measured miss rates. No prior-model timings are pooled.',
            'After small gates: four candidates per arm at T1024, then inherited control plus each arm\'s first-call winner at T2048 and T4096. All negative results retained.']
    (BASE/'SELECTION.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    result=analyze();print(json.dumps(result['selected'],indent=2))
