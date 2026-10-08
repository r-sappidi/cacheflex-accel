"""Replayable event-boundary accounting; not a whole-ROI causal attribution."""
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/cacheflex_flash_attention/spm_dataflow_trace_20260918'
targets = json.loads((OUT/'plan.json').read_text())['targets'][:16]
events = {a: {} for a in targets}
evidence = []
traffic = collections.Counter()
for line in (OUT/'trace/dataflow.trace').open(errors='replace'):
    m = re.match(r'\s*(\d+)', line)
    if not m:
        continue
    tick = int(m[1])
    if 'netifs08: Scheduling' in line and 27178688000 <= tick <= 27178720500:
        typ = re.search(r' Type = (\w+)', line)
        if typ:
            traffic[typ[1]] += 1
    match = re.search(r'\[?(0x10cc[0-9a-f]+)', line)
    if not match or match[1] not in events:
        continue
    a = match[1]
    row = events[a]
    key = None
    if re.search(r'0\s+Seq\s+Begin', line): key = 'begin'
    elif re.search(r'0\s+Seq\s+Done', line): key = 'done'
    elif 'l1_cntrl0.requestToL2: Enqueue' in line: key = 'request_enqueue'
    elif 'netifs08: Scheduling' in line and 'GETS_SILENT' in line:
        key = 'request_send'; tick = int(re.search(r'time:(\d+)', line)[1])
    elif 'L1RequestToL2Cache: Enqueue' in line: key = 'llc_arrival'
    elif re.search(r'L2Cache\s+SPM_GETS_SILENT\s+M>M', line): key = 'llc_hit'
    elif 'responseFromL2Cache: Enqueue' in line: key = 'response_enqueue'
    elif 'Scheduling' in line and 'ResponseMsg' in line and 'Dest NI=8 ' in line:
        key = 'response_head_send' if ' Id=0 ' in line else 'response_tail_send'
        tick = int(re.search(r'time:(\d+)', line)[1])
    elif 'l1_cntrl0.responseFromL2: Enqueue' in line: key = 'response_arrival'
    elif re.search(r'0\s+L1Cache\s+Data_all_Acks\s+IX_D>X', line): key = 'install'
    if key:
        if key == 'response_tail_send': row[key] = tick
        else: row.setdefault(key, tick)
        evidence.append(line)
        ready = re.search(r'arrival_time: (\d+)', line)
        if ready: row[key+'_ready'] = int(ready[1])

boundaries = ['begin','request_enqueue_ready','request_send','llc_arrival',
              'llc_hit','response_head_send','response_arrival','install','done']
rows = []
for a, e in events.items():
    assert all(k in e for k in boundaries), (a,e)
    phases = {f'{x} -> {y}': (e[y]-e[x])/400 for x,y in zip(boundaries,boundaries[1:])}
    assert all(v >= 0 for v in phases.values())
    assert sum(phases.values()) == (e['done']-e['begin'])/400
    rows.append(dict(address=a,events=e,cycles=phases,total_cycles=sum(phases.values())))
mean = {k:sum(r['cycles'][k] for r in rows)/len(rows) for k in rows[0]['cycles']}
result = dict(requests=rows,mean_cycles=mean,source_ni8_scheduled_flits_during_first_wait=dict(traffic),
              caveat='Event accounting includes configured latency, serialization and contention. Not removable cycles or whole-runtime percentages.')
(OUT/'dependency_analysis.json').write_text(json.dumps(result,indent=2))
(OUT/'dependency_evidence.log').write_text(''.join(evidence))
report = ['# Downstream dependency analysis','',
 'All 16 copies occupying the shared admission pool were served by LLC M->M hits in this window.',
 'The already established stack-load admission wait is 111 core cycles. These overlapping copy lifetimes must not be summed as its cause.',
 '', '| Event interval | First copy cycles | Mean of 16 |','|---|---:|---:|']
for k,v in mean.items(): report.append(f"| {k} | {rows[0]['cycles'][k]:.2f} | {v:.2f} |")
report += ['', 'Source NI8 scheduled flits while first request waited for injection: '+str(dict(traffic)),
 '', 'These boundaries identify request injection and return transport as major holding intervals, not DRAM service for these copies. They do not separate every router arbitration/credit stall from normal transport latency.',
 'The trace is a worst-gap-selected multi-head prefix, not a whole-runtime sample. No end-to-end speedup is implied. A fixed-storage network/admission intervention requires matched full-ROI timing before a performance claim.']
(OUT/'DEPENDENCIES.md').write_text('\n'.join(report)+'\n')
print(json.dumps(result,indent=2))
