"""Selected-load timelines. Per-access latencies, NOT additive FA runtime costs.

CPU SN and packet identity join fragments; RubyRequest identity joins ingress.
No address-only transaction joins. Censored/reallocated/forwarded loads excluded
from the additive dispatch-to-data-ready decomposition and counted explicitly.
"""
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import re
import sys
from analyze_spm_consumers import parse_lines, distribution, source_time, producer_kind
from build_access_timing import save

LINE=re.compile(r'^(\d+): ([^:]+): (CFL|CFT) (\w+) (.*)')
FIELDS=re.compile(r'(\w+)=([^\s]+)')
CPU=re.compile(r'system.cpu(\d+)\.')
SEQ=re.compile(r'system.ruby.l0_cntrl(\d+).sequencer')
PCS={'spm':{0x4020c0:'QK_K',0x402120:'QK_K',0x402a58:'O_rescale'},
     'cache':{0x402040:'QK_K',0x4020a0:'QK_K',0x4029d8:'O_rescale'}}
CYCLE=400

def register_ready(n):
    if 'regs_ready' in n:return n['regs_ready']
    # MemDepUnit::insert handles initially-ready loads directly, bypassing the
    # IQ addIfReady hook. SRC ready bits still prove readiness at dispatch.
    # For unready sources use the actual recorded wake, never an inferred
    # producer completion. Unknown/boundary sources remain excluded.
    if 'dispatch' not in n or not n.get('sources'):return None
    times=[n['dispatch']]
    for source in n['sources']:
        if source.get('ready'):continue
        if 'wake' not in source:return None
        times.append(source['wake'])
    return max(times)

def fields(s):
    f=dict(FIELDS.findall(s))
    for k,v in list(f.items()):
        try:f[k]=int(v,0)
        except ValueError:pass
    return f

def read_events(path):
    loads=defaultdict(list);seq=defaultdict(list);queues=defaultdict(list)
    with gzip.open(path,'rt') as stream:
        for number,line in enumerate(stream,1):
            m=LINE.match(line)
            if not m:continue
            tick,name,family,kind,tail=int(m[1]),m[2],m[3],m[4],m[5]
            if family!='CFL':continue
            f=fields(tail);e=dict(tick=tick,event=kind,name=name,line=number,**f)
            core=CPU.match(name)
            if core and 'sn' in f:loads[int(core[1]),f['sn']].append(e)
            core=SEQ.match(name)
            if core and 'id' in f:seq[int(core[1]),f['id']].append(e)
    wanted={e['msg'] for es in seq.values() for e in es if e['event']=='SEQ_ENQUEUE'}
    with gzip.open(path,'rt') as stream:
        for number,line in enumerate(stream,1):
            m=LINE.match(line)
            if not m or m[3]!='CFT' or not m[4].startswith('QUEUE_'):continue
            f=fields(m[5])
            if f.get('msg') in wanted:
                queues[f['msg']].append(dict(tick=int(m[1]),name=m[2],event=m[4],line=number,**f))
    return loads,seq,queues

def assemble(nodes,loads,seq,queues,backend):
    rows=[];excluded=Counter();rejections=Counter();attempts=Counter()
    for key,es in loads.items():
        n=nodes.get(key,{})
        pc=n.get('pc',es[0].get('pc'))
        if pc not in PCS[backend]:continue
        for e in es:
            if e['event']=='FRAG_TRY':
                attempts['all']+=1
                if e['cache_blocked']:attempts['existing_retry_block']+=1
                elif not e['port_available']:attempts['no_cpu_load_port']+=1
            if e['event']=='FRAG_RESULT' and not e['accepted'] and e['rejected_by_ruby']:
                matches=[r for r in seq.get((key[0],e['id']),[]) if r['tick']==e['tick'] and r['event']=='SEQ_REJECT']
                if not matches:rejections['other_insert_or_coalescing_reject']+=1
                for r in matches:rejections[r.get('reason','unknown')]+=1
        reg_ready=register_ready(n)
        if not all(k in n for k in ('dispatch','ready','done','commit')) or not n.get('issues') or reg_ready is None:
            excluded['boundary_uncommitted_or_unknown_readiness']+=1;continue
        kinds=defaultdict(list)
        for e in es:kinds[e['event']].append(e)
        if kinds['STORE_FORWARD']:
            excluded['store_forwarded']+=1;continue
        accepted=[e for e in kinds['FRAG_RESULT'] if e['accepted']]
        received=kinds['FRAG_RESPONSE']
        if len(kinds['ADDRESS'])!=1 or not kinds['TRANSLATED_READ'] or not kinds['ELIGIBLE'] or not accepted:
            excluded['missing_cpu_interval']+=1;continue
        a=kinds['ADDRESS'][0]
        # Normal blocked loads reissue using the SAME LSQRequest and packet
        # fragments; excluding these would discard the admission bottleneck.
        # A discarded/reallocated request is different and remains excluded.
        requests={e.get('req') for e in kinds['TRANSLATED_READ'] if 'req' in e}
        if len(requests)>1:
            excluded['reallocated_request']+=1;continue
        identities={e['id'] for e in accepted}
        if len(identities)!=len(accepted) or len(received)!=len(accepted) or {e['id'] for e in received}!=identities:
            excluded['incomplete_fragment_set']+=1;continue
        if sum(e['bytes'] for e in accepted)!=a['bytes'] or any(e['squashed'] for e in received):
            excluded['partial_predicate_or_squashed']+=1;continue
        frags=[]
        for sent in accepted:
            response=next(e for e in received if e['id']==sent['id'])
            related=[e for e in seq.get((key[0],sent['id']),[]) if sent['tick']<=e['tick']<=response['tick']]
            done=[e for e in related if e['event']=='SEQ_DONE']
            enqueue=[e for e in related if e['event']=='SEQ_ENQUEUE']
            f=dict(packet=hex(sent['id']),pa=hex(sent['pa']),bytes=sent['bytes'],
                   accepted=sent['tick'],response=response['tick'],service_cycles=(response['tick']-sent['tick'])/CYCLE,
                   source=done[0]['source'] if len(done)==1 else 'unresolved',
                   external=done[0]['external'] if len(done)==1 else None)
            if len(done)==1:f['callback_to_cpu_cycles']=(response['tick']-done[0]['tick'])/CYCLE
            if len(enqueue)==1:
                en=enqueue[0]
                q=[e for e in queues.get(en['msg'],[]) if en['tick']<=e['tick']<=response['tick'] and e['event']=='QUEUE_DEQUEUE' and e.get('final')==1]
                if len(q)==1:
                    de=q[0];f.update(ingress_queue=de['name'],ingress_dequeue=de['tick'],
                                    ingress_latency_cycles=(en['ready']-en['tick'])/CYCLE,
                                    ingress_ready_to_dequeue_cycles=(de['tick']-en['ready'])/CYCLE)
                    if len(done)==1:f['after_ingress_to_callback_cycles']=(done[0]['tick']-de['tick'])/CYCLE
            frags.append(f)
        # These disjoint intervals sum to dispatch->DONE. Earlier fragment
        # service overlaps issuance and is NOT added to the tail wait again.
        timeline=[n['dispatch'],reg_ready,n['ready'],n['issues'][0],a['tick'],
                  kinds['TRANSLATED_READ'][0]['tick'],kinds['ELIGIBLE'][0]['tick'],
                  max(e['tick'] for e in accepted),max(e['tick'] for e in received),n['done']]
        if timeline!=sorted(timeline):
            excluded['nonmonotonic_or_replayed_timeline']+=1;continue
        labels=['register_dependencies','memory_ready_gate','ready_to_issue','issue_to_address',
                'translation_and_read_retry','lsq_order_check','fragment_admission_and_retry_span',
                'remaining_memory_response','response_to_register_ready']
        intervals={k:(b-a)/CYCLE for k,a,b in zip(labels,timeline,timeline[1:])}
        latest=[]
        for s in n['sources']:
            t,p=source_time(s,n,nodes)
            if t is not None:latest.append((t,producer_kind(p)))
        tail=max(frags,key=lambda f:f['response'])
        rows.append(dict(core=key[0],sn=key[1],pc=hex(pc),phase=PCS[backend][pc],
                         dispatch=n['dispatch'],done=n['done'],cycles=(n['done']-n['dispatch'])/CYCLE,
                         intervals=intervals,fragments=frags,last_fragment=tail,
                         issue_attempts=len(n['issues']),execute_attempts=len(kinds['EXECUTE']),
                         register_ready_basis='explicit' if 'regs_ready' in n else 'source_ready_bits_and_actual_wakeups',
                         last_known_source=max(latest)[1] if latest else 'unknown',
                         provenance_note='ready-at-dispatch sources may have boundary-unknown producers',
                         order_waits=kinds['ORDER_WAIT']))
    return rows,dict(excluded),dict(attempts),dict(rejections)

def analyze(path,backend,destination=None):
    destination=destination or path
    destination.mkdir(parents=True,exist_ok=True)
    with gzip.open(path/'trace.gz','rt') as f:nodes,rob,conflicts=parse_lines(f)
    loads,seq,queues=read_events(path/'trace.gz')
    rows,excluded,attempts,rejections=assemble(nodes,loads,seq,queues,backend)
    summaries={}
    for phase in ('QK_K','O_rescale'):
        rs=[r for r in rows if r['phase']==phase]
        if not rs:
            summaries[phase]={'count':0};continue
        total=sum(r['cycles'] for r in rs)
        keys=rs[0]['intervals']
        fragments=[f for r in rs for f in r['fragments']]
        summaries[phase]=dict(count=len(rs),cores=dict(Counter(r['core'] for r in rs)),
          dispatch_to_done_cycles=distribution([r['cycles'] for r in rs]),
          interval_mean_cycles={k:sum(r['intervals'][k] for r in rs)/len(rs) for k in keys},
          interval_percent_of_summed_load_lifetimes={k:100*sum(r['intervals'][k] for r in rs)/total if total else 0 for k in keys},
          fragment_service_cycles=distribution([f['service_cycles'] for f in fragments]),
          tail_fragment_service_cycles=distribution([r['last_fragment']['service_cycles'] for r in rs]),
          fragment_sources=dict(Counter(f['source'] for f in fragments)),
          ingress_ready_to_dequeue_cycles=distribution([f['ingress_ready_to_dequeue_cycles'] for f in fragments if 'ingress_ready_to_dequeue_cycles' in f]),
          last_known_sources=dict(Counter(r['last_known_source'] for r in rs)))
    report=dict(backend=backend,source=str(path/'trace.gz'),summaries=summaries,
                exclusions=excluded,fragment_attempts=attempts,ruby_rejections=rejections,
                dependency_conflicts=conflicts,complete_loads=len(rows),
                caveat='Sampled per-load lifetime, not additive end-to-end latency or a causal intervention. Queue ready-to-dequeue includes controller/resource stalls, not isolated SRAM bank arbitration.')
    save(destination/'analysis.json',report)
    with gzip.open(destination/'load_lifetimes.json.gz','wt') as f:json.dump(rows,f)
    return report

def main():
    ex=Path(sys.argv[1]);label=sys.argv[2]
    assert json.loads((ex/f'{label}_equivalence.json').read_text())['all_finite_nonhost_stats_equal']
    reports=[analyze(ex/(label+'_trace')/v,v,ex/'analysis'/label/v) for v in ('cache','spm')]
    save(ex/f'{label}_analysis.json',reports)
    lines=['# Coherent load lifetime: '+label,'',
           'Fixed improved-IFETCH T4096, 8 cores. Cores 0/7 sampled. Trace ON/OFF stats identical.',
           'Percentages are summed load lifetimes, NOT fractions of total FA time.','']
    for r in reports:
        lines+=['## '+r['backend'],'',f"Complete loads: {r['complete_loads']}; exclusions: {r['exclusions']}",'']
        for phase,s in r['summaries'].items():
            lines += [f"{phase}: {s['count']} complete loads.",'']
            if not s['count']:continue
            lines += ['| Interval | Mean cycles | % load lifetime |','|---|---:|---:|']
            for k,m in s['interval_mean_cycles'].items():
                lines.append(f"| {k} | {m:.3f} | {s['interval_percent_of_summed_load_lifetimes'][k]:.2f} |")
            lines += ['',f"Fragment sources: {s['fragment_sources']}",f"Ingress ready→dequeue: {s['ingress_ready_to_dequeue_cycles']}",'']
        lines += [f"Ruby rejection reasons: {r['ruby_rejections']}",'']
    (ex/f'{label}_REPORT.md').write_text('\n'.join(lines)+'\n')
    # Lack of any useful observation is a coverage failure, not a zero latency result.
    assert all(r['complete_loads'] for r in reports),'No complete selected loads; need another window'

if __name__=='__main__':main()
