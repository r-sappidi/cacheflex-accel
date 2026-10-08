"""Reproducible evidence for PV-loop instruction supply cadence."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
from analyze_spm_consumers import parse_lines, distribution, CYCLE
from build_access_timing import save

FETCH = re.compile(r'^(\d+): system.cpu([07]).fetch: (.*)')
CREATED = re.compile(r'Instruction PC \((0x[0-9a-f]+).*created \[sn:(\d+)\]')
REQUEST = re.compile(r'Fetching cache line (0x[0-9a-f]+) for addr')


def analyze(path):
    assert json.loads((path/'validated_prefix.json').read_text())['all_finite_nonhost_stats_equal']
    with gzip.open(path/'trace.gz', 'rt') as f:
        nodes, _, _ = parse_lines(f)
    created = {}; events = defaultdict(list); pending = {}; refills = []
    with gzip.open(path/'trace.gz', 'rt') as f:
        for line_no,line in enumerate(f,1):
            m = FETCH.match(line)
            if not m:continue
            tick, core, tail = int(m[1]), int(m[2]), m[3]
            c = CREATED.search(tail); r = REQUEST.search(tail)
            if c:
                created[(core,int(c[2]))] = dict(tick=tick, pc=int(c[1],16), line=line_no)
            if r:
                pending[core] = dict(core=core,request=tick,pc=int(r[1],16),line=line_no)
            if 'Waking up from cache miss.' in tail and core in pending:
                p=pending.pop(core);p['response']=tick;p['latency_cycles']=(tick-p['request'])/CYCLE
                refills.append(p)
            if c or r or 'Waking up from cache miss.' in tail:
                events[core].append(dict(tick=tick,line=line_no,text=tail))
    pairs=[]
    for core in (0,7):
        loads=sorted((n for n in nodes.values() if n['core']==core and n.get('spm') and
                      n.get('load') and 'commit' in n and 'dispatch' in n and len(n['issues'])==1
                      and (core,n['sn']) in created),key=lambda n:n['sn'])
        for a,b in zip(loads,loads[1:]):
            if b['va']!=a['va']+64:continue
            fa,fb=created[(core,a['sn'])],created[(core,b['sn'])]
            pairs.append(dict(core=core,load_sn=a['sn'],next_sn=b['sn'],
                              fetch_tick=fa['tick'],next_fetch_tick=fb['tick'],
                              fetch_gap=(fb['tick']-fa['tick'])/CYCLE,
                              dispatch_gap=(b['dispatch']-a['dispatch'])/CYCLE,
                              issue_gap=(b['issues'][0]-a['issues'][0])/CYCLE))
    assert pairs
    # First steady sample after initial warm-up, not a worst-gap selection.
    steady=[p for p in pairs[20:] if p['fetch_gap']==p['dispatch_gap']==p['issue_gap']==7]
    assert steady
    example=steady[0];core=example['core'];lo=example['fetch_tick'];hi=example['next_fetch_tick']
    instruction_groups=defaultdict(list)
    for (c,sn),ev in created.items():
        if c==core and lo<=ev['tick']<hi:
            instruction_groups[(ev['tick']-lo)//CYCLE].append(hex(ev['pc']))
    assert sum(map(len,instruction_groups.values()))==19
    assert sorted(map(len,instruction_groups.values()))==[1,3,5,5,5]
    sample_events=[e for e in events[core] if lo<=e['tick']<=hi]
    relevant=[r for r in refills if r['pc'] in (0x402bc0,0x402c00)]
    result=dict(pairs=len(pairs),metrics={k:distribution([p[k] for p in pairs])
                                       for k in ('fetch_gap','dispatch_gap','issue_gap')},
                all_three_seven=sum(p['fetch_gap']==p['dispatch_gap']==p['issue_gap']==7 for p in pairs),
                local_refill_responses=distribution([r['latency_cycles'] for r in relevant]),
                example=example,instruction_groups=dict(instruction_groups),events=sample_events)
    save(path/'frontend_analysis.json',result)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('experiment',type=Path);args=ap.parse_args()
    ex=args.experiment;out=ex/'frontend_followup'
    result={v:analyze(out/v) for v in ('normal','fast')}
    save(out/'analysis.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                results=result))
    lines=['# Direct-SPMLD latency masking: instruction-supply evidence','',
           'T4096, single head, 8 cores, VL16; unchanged V-staging kernel and cache hardware. '
           'Only direct-SPM service latency differs. Both frontend traces exactly match the '
           'untraced prefixes in all finite non-host statistics and configuration.', '',
           'The 19-instruction PV loop spans 76 bytes (0x402bc0–0x402c08). '
           'With fetchWidth=5 and a 64-byte fetch buffer, the observed steady fetch pattern is:', '',
           '| Relative cycle | Instructions fetched | Event |','|---:|---:|---|',
           '| 0 | 5 | Loop begins; includes SPMLD |',
           '| 1 | 5 | |','| 2 | 5 | |',
           '| 3 | 1 | Request next fetch block, 0x402c00 |',
           '| 4 | 0 | Response arrives one tick after this CPU edge |',
           '| 5 | 3 | Fetch loop tail/branch; request block 0x402bc0 |',
           '| 6 | 0 | Response arrives one tick after this CPU edge |',
           '| 7 | 5 | Next iteration begins |', '',
           'This is an instruction-supply throughput constraint in the sampled steady loop, '
           'not an SPM response or FMA-ready-queue delay. Pipelined loads can finish four cycles '
           'earlier while successive iterations are still supplied seven cycles apart.', '',
           '| Measurement | Normal | Fast SPMLD |','|---|---:|---:|']
    for k in ('fetch_gap','dispatch_gap','issue_gap'):
        lines.append(f'| Median {k} (cycles) | {result["normal"]["metrics"][k]["median"]} | '
                     f'{result["fast"]["metrics"][k]["median"]} |')
    lines.append(f'| All three gaps exactly 7 / observed pairs | {result["normal"]["all_three_seven"]} / '
                 f'{result["normal"]["pairs"]} | {result["fast"]["all_three_seven"]} / {result["fast"]["pairs"]} |')
    for v,r in result.items():
        e=r['example'];lines += ['',f'## {v}: concrete trace example','',
            f'core{e["core"]}, SPMLD SN{e["load_sn"]}→SN{e["next_sn"]}, '
            f'fetch ticks {e["fetch_tick"]}→{e["next_fetch_tick"]}. '
            'Full event lines and cycle groups are in frontend_analysis.json.']
    lines += ['', '## Whole-run context and limits','',
        'Previously validated full-ROI results: normal SPM5908.1544us, fast SPMLD5921.8840us '
        '(0.232% slower). Access latency really falls9→5cycles; the early and middle traces '
        'both retain a7-cycle median dispatch/issue cadence. First-FMA samples mostly have the '
        'SPM operand last or tied, so it is incorrect to say most are waiting on coherent operands.', '',
        'Some samples do wait on coherent packed-P loads or accumulator FMAs. Other FA phases '
        'do not use this SPMLD path. These observations explain a concrete local masking mechanism; '
        'they do not assign an end-to-end latency fraction to the front end or establish its '
        'optimization benefit. That needs a matched full-ROI intervention for cache and SPM.', '',
        'The Fetch debug string “Waking up from cache miss” is printed on every instruction-cache '
        'completion; it is NOT evidence of actual cache capacity misses. The observed401-tick '
        'refill responses equal1.0025 CPUcycles and become usable on the next CPU edge. '
        'The one-tick timing detail should be audited before treating it as a silicon requirement.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
