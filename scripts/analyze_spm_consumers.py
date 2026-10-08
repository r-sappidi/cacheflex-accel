"""Dependency evidence, NOT an additive attribution of whole-FA runtime.

Rename-register provenance includes values ready before IQ dispatch. Actual
wakeup edges override that provenance. Only committed, uncensored consumers
are used, with incomplete source provenance counted explicitly.
"""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
from build_access_timing import save

LINE=re.compile(r'^(\d+): system.cpu(\d+)\.[^:]+: CFD (\w+) (.*)')
FIELDS=re.compile(r'(\w+)=([^\s]+)')
CYCLE=400

def parse_lines(lines):
    nodes={};registers={};rob=defaultdict(list);conflicts=0
    def node(core,sn):
        return nodes.setdefault((core,sn),dict(core=core,sn=sn,sources=[],issues=[],waits=[],edges=[]))
    for line_no,line in enumerate(lines,1):
        m=LINE.match(line)
        if not m:continue
        tick,core,kind,tail=int(m[1]),int(m[2]),m[3],m[4]
        f=dict(FIELDS.findall(tail))
        for k in list(f):
            if k!='reason' and k!='asm':
                try:f[k]=int(f[k],0)
                except ValueError:pass
        if kind=='EDGE':
            n=node(core,f['consumer']);n['edges'].append(dict(producer=f['producer'],reg=f['reg'],tick=tick))
            # Capture actual wakeups, including any that cross the trace start.
            found=False
            for s in n['sources']:
                if s['reg']==f['reg']:
                    if s.get('producer') not in (None,f['producer']):conflicts+=1
                    s['producer']=f['producer'];s['wake']=tick;found=True
            if not found:n['sources'].append(dict(idx=-1,reg=f['reg'],producer=f['producer'],wake=tick,ready=0))
            continue
        n=node(core,f['sn'])
        if kind=='NODE':
            n.update(f);n['dispatch']=tick;n['asm']=tail.partition(' asm=')[2];n['node_line']=line_no
        elif kind=='SRC':
            n['sources'].append(dict(idx=f['idx'],reg=f['reg'],ready=f['ready'],
                                     producer=registers.get((core,f['reg']))))
        elif kind=='DST':registers[(core,f['reg'])]=f['sn']
        elif kind=='ISSUE':n['issues'].append(tick);n['issue_line']=line_no;n['op_latency']=f['latency']
        elif kind=='DONE':n.update(f);n['done']=tick;n['done_line']=line_no
        elif kind=='MEM_RESP':n['response']=tick;n['response_line']=line_no
        elif kind=='REGS_READY':n.setdefault('regs_ready',tick)
        elif kind=='READY':n.setdefault('ready',tick);n['ready_line']=line_no
        elif kind=='WAIT':n['waits'].append((tick,f['reason']))
        elif kind=='COMMIT':n['commit']=tick;n['commit_line']=line_no
        elif kind=='ROB_BLOCK':rob[core].append((tick,f['sn']))
    return nodes,rob,conflicts

def producer_kind(n):
    if n.get('spm') and n.get('load'):return 'SPMLD'
    if n.get('load'):return 'coherent_load'
    asm=n.get('asm','').strip().split()
    op=asm[0] if asm else 'unknown'
    if op.startswith(('fmla','fmls','fmadd','fmsub')):return 'FMA_accumulator_or_compute'
    return 'compute_or_address:'+op

def describe(n):
    return dict(core=n['core'],sn=n['sn'],pc=hex(n.get('pc',0)),asm=n.get('asm',''),
                kind=producer_kind(n),va=hex(n.get('va',0)),dispatch=n.get('dispatch'),
                issue=n['issues'][0] if n['issues'] else None,done=n.get('done'),
                node_line=n.get('node_line'),done_line=n.get('done_line'))

def source_time(s,n,nodes):
    p=nodes.get((n['core'],s.get('producer')),{})
    if 'done' in p:return p['done'],p
    if 'wake' in s:return s['wake'],p
    # This upper bound can establish that a boundary-ready input was early,
    # but cannot identify its producer or exact completion time.
    if s.get('ready') and 'dispatch' in n:return n['dispatch'],p
    return None,p

def consumer_rows(nodes,anchors=None):
    users=defaultdict(list)
    for n in nodes.values():
        if not all(k in n for k in ('dispatch','ready','done','commit')) or not n['issues']:continue
        for sn in {s.get('producer') for s in n['sources']}:
            if sn is not None:users[(n['core'],sn)].append(n)
    rows=[];excluded=Counter()
    for key,load in nodes.items():
        if not (load.get('spm') and load.get('load') and 'commit' in load):continue
        if anchors and load['sn']<anchors[load['core']]['sn']:continue
        if not all(k in load for k in ('dispatch','done')) or len(load['issues'])!=1:
            excluded['load_boundary_or_reissue']+=1;continue
        consumers=sorted(users.get(key,[]),key=lambda n:(n['issues'][0],n['sn']))
        if not consumers:excluded['no_complete_consumer']+=1;continue
        # First use avoids counting each reuse of the same V vector as an
        # independent latent-latency opportunity.
        c=consumers[0];issued=c['issues'][0];ready=c['ready'];ld=load['done']
        if issued<ld:excluded['invalid_or_pinned_provenance']+=1;continue
        others=[];unknown=0
        for s in c['sources']:
            if s.get('producer')==load['sn']:continue
            t,p=source_time(s,c,nodes)
            if t is None:unknown+=1
            else:others.append((t,p))
        late=[(t,p) for t,p in others if t>ld]
        explicit=[(t,p) for t,p in late if 'done' in p or p.get('sn') is not None]
        if unknown:
            reason='incomplete_source_provenance';blocker=None
        elif explicit:
            t,blocker=max(explicit,key=lambda x:x[0]);reason='other_operand:'+producer_kind(blocker)
        elif c['dispatch']>ld:
            reason='consumer_dispatched_after_SPM_ready';blocker=None
        elif ready<=ld:
            reason='SPM_is_last_or_tied_ready_operand';blocker=None
        else:
            reason='readiness_or_memory_order_residual';blocker=None
        waits=Counter(why for tick,why in c['waits'] if ready<=tick<issued)
        rows.append(dict(core=load['core'],load=describe(load),consumer=describe(c),
                         ready=ready,issue=issued,load_latency_cycles=(ld-load['issues'][0])/CYCLE,
                         data_to_ready_cycles=max(0,ready-ld)/CYCLE,
                         ready_to_issue_cycles=(issued-ready)/CYCLE,
                         data_to_use_cycles=(issued-ld)/CYCLE,
                         reason=reason,blocker=describe(blocker) if blocker else None,
                         unknown_sources=unknown,ready_queue_wait_cycles=dict(waits),
                         all_direct_committed_consumers=len(consumers)))
    return rows,dict(excluded)

def distribution(values):
    if not values:return dict(count=0)
    v=sorted(values)
    return dict(count=len(v),mean=statistics.mean(v),median=statistics.median(v),
                p90=v[int(.9*(len(v)-1))],max=max(v))

def anchors(path):
    found={};pat=re.compile(r'^(\d+): system.cpu([07])\..*Load Issue to Cache: SN=(\d+)')
    with gzip.open(path/'trace.gz','rt') as f:
        for line in f:
            m=pat.search(line)
            if m:found.setdefault(int(m[2]),dict(tick=int(m[1]),sn=int(m[3])))
    assert set(found)=={0,7},found
    return found

def analyze(ex,label,variant):
    path=ex/(label+'_trace')/variant
    with gzip.open(path/'trace.gz','rt') as stream:nodes,rob,conflicts=parse_lines(stream)
    rows,excluded=consumer_rows(nodes,anchors(ex/(label+'_scan')/variant))
    assert len(rows)>64,('Insufficient complete samples',variant,len(rows))
    by_reason=Counter(r['reason'] for r in rows)
    waits=Counter();blockers=Counter()
    for r in rows:
        waits.update(r['ready_queue_wait_cycles'])
        if r['blocker']:blockers[(r['blocker']['pc'],r['blocker']['asm'],r['blocker']['kind'])]+=1
    summary=dict(variant=variant,rows=len(rows),cores=sorted({r['core'] for r in rows}),
                 reasons=dict(by_reason),excluded=excluded,wakeup_provenance_overrides=conflicts,
                 ready_queue_wait_cycles=dict(waits),
                 metrics={k:distribution([r[k] for r in rows]) for k in
                          ('load_latency_cycles','data_to_ready_cycles','ready_to_issue_cycles','data_to_use_cycles')},
                 blocking_producer_pcs=[dict(pc=k[0],asm=k[1],kind=k[2],count=v) for k,v in blockers.most_common(12)],
                 examples=[])
    for reason,count in by_reason.most_common():
        rr=[r for r in rows if r['reason']==reason]
        # Median, not worst-gap-selected examples.
        rr.sort(key=lambda r:r['data_to_use_cycles']);summary['examples'].append(rr[len(rr)//2])
    save(path/'consumer_rows.json',rows);save(path/'consumer_summary.json',summary)
    return summary,rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('experiment',type=Path);ap.add_argument('label');args=ap.parse_args()
    ex=args.experiment;label=args.label
    save(ex/f'{label}_analysis_provenance.json',dict(
        script=str(Path(__file__).resolve()),
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    assert json.loads((ex/f'{label}_observer_equivalence.json').read_text())['passed']
    summaries={};rowsets={}
    for v in ('normal','fast'):summaries[v],rowsets[v]=analyze(ex,label,v)
    # Compare local response-to-consumer intervals. Do not equate SNs across
    # interventions: speculative control flow and issue order can differ.
    lines=[f'# SPM consumer dependency diagnostic: {label}', '',
        'Matched T4096/VL16 8-core real-staging FA; only direct-SPM service latency changes5->1. '
        'Sampled cores0/7. Trace ON/OFF matches ALL finite non-host counters at identical stop ticks. '
        'These are bounded prefixes of previously fully numerically validated runs, not new full-FA timings.', '',
        '| Metric (core cycles) | Normal | Fast SPMLD |', '|---|---:|---:|']
    for k in ('load_latency_cycles','data_to_ready_cycles','ready_to_issue_cycles','data_to_use_cycles'):
        lines.append(f'| {k}, mean | {summaries["normal"]["metrics"][k]["mean"]:.3f} | {summaries["fast"]["metrics"][k]["mean"]:.3f} |')
    lines += ['', '| First-consumer readiness reason | Normal samples | Fast samples |','|---|---:|---:|']
    for k in sorted(set(summaries['normal']['reasons'])|set(summaries['fast']['reasons'])):
        lines.append(f'| {k} | {summaries["normal"]["reasons"].get(k,0)} | {summaries["fast"]["reasons"].get(k,0)} |')
    for v,s in summaries.items():
        lines += ['', f'## {v}', '',f'Complete first-consumer samples: {s["rows"]}; exclusions: {s["excluded"]}.',
                  f'Observed ready-queue wait cycles (summed across consumers, NOT runtime): {s["ready_queue_wait_cycles"]}.', '',
                  'Top later-ready producer instructions:', '']
        for r in s['blocking_producer_pcs']:lines.append(f'- {r["pc"]}: `{r["asm"]}` ({r["kind"]}), {r["count"]} samples.')
        lines += ['', 'Median examples by category:','']
        for r in s['examples']:
            lines.append(f'- core{r["core"]}, load SN{r["load"]["sn"]} {r["load"]["pc"]} ready@{r["load"]["done"]}; '
                         f'consumer SN{r["consumer"]["sn"]} `{r["consumer"]["asm"]}` '
                         f'ready@{r["ready"]}, issue@{r["issue"]}: {r["reason"]}.')
    lines += ['', 'Cautions: first *committed* direct consumers only; boundary/reissued loads excluded. '
              'Source-ready-at-dispatch supplies an upper bound for missing pre-window producers. '
              'Unknown provenance is not assigned a cause. FU_unavailable means a real allocation failure '
              'for the same operation class in that cycle; issue_width means the global issue budget was '
              'exhausted without observing such a failure. Neither proves a unique end-to-end critical path. '
              'Samples across variants are distributions, not assumed identical dynamic sequence numbers.']
    save(ex/f'{label}_analysis.json',summaries)
    (ex/f'{label}_REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
