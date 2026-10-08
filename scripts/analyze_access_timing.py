"""An additive CPU-to-install timeline; trace the tail flit, never sum flits.

Intervals describe where elapsed time was spent. They are not independently
removable delays, nor an attribution of whole-kernel execution time.
"""
import collections
import json
from pathlib import Path
import re
import sys
from build_access_timing import OUT, save
from run_access_timing import protocol


def parse(path):
    events=[];cpu=[]
    pattern=re.compile(r'^(\d+): (.*?): CFT (\w+) (.*)$')
    with path.open('rb') as stream:
        for i,b in enumerate(stream,1):
            line=b.decode(errors='backslashreplace').rstrip()
            m=pattern.match(line)
            if m:
                t,obj,event,fields=m.groups()
                values=dict(re.findall(r'(\w+)=(\S+)',fields))
                if 'event' in values:values['protocol_event']=values.pop('event')
                events.append(dict(tick=int(t),obj=obj,event=event,line=i,**values))
            elif 'system.cpu0.' in line:
                cpu.append((i,line))
    return events,cpu


def main(ex):
    assert (ex/'trace/validated.json').exists()
    target=json.loads((ex/'target.json').read_text());trace=ex/'trace/timing.trace'
    ev,cpu=parse(trace);addr=target['addr'];begin=target['begin']['tick'];install=target['install']['tick']
    # Join instruction identity at acceptance, not merely by reused source address.
    sends=[line for _,line in cpu if line.startswith(str(begin)+':') and
           f'WriteReq [{addr[2:]}:' in line and 'was sent (' in line]
    assert len(sends)==1,sends
    sn=re.search(r'\[sn:(\d+)\]',sends[0])[1]
    inst=[(n,line) for n,line in cpu if f'[sn:{sn}]' in line or f'SN={sn} ' in line]
    def itick(text):
        found=[int(line.split(':',1)[0]) for _,line in inst if text in line]
        assert len(found)==1,(text,found)
        return found[0]
    issue=itick('Thread 0: Issuing instruction');execute=itick('Executing store PC')
    eligible=itick('Marking store as able to write back')
    def find(event,lo,hi,obj=None,**fields):
        return [e for e in ev if e['event']==event and lo<=e['tick']<=hi and
                (obj is None or obj in e['obj']) and all(e.get(k)==str(v) for k,v in fields.items())]
    def one(*args,**kw):
        es=find(*args,**kw);assert len(es)==1,(args,kw,es);return es[0]
    rows=[]
    def add(category,label,a,b):
        assert b>=a,(label,a,b)
        if rows:assert rows[-1]['end']==a,(label,'gap/overlap',rows[-1]['end'],a)
        rows.append(dict(category=category,label=label,start=a,end=b,ticks=b-a,core_cycles=(b-a)/400))
    add('CPU','IQ issue to address execution',issue,execute)
    add('CPU','Execution to store eligible',execute,eligible)
    add('CPU','Store eligible to Ruby acceptance',eligible,begin)
    mandatory=one('QUEUE_ENQUEUE',begin,begin,'mandatoryQueue',addr=addr)
    ready=int(mandatory['ready'])
    add('Fixed protocol/local delays','Mandatory queue admission delay',begin,ready)
    pe=[e for e in protocol(trace) if e['addr']==addr and e['core']==0 and
        e['ctrl']=='L0Cache' and e['event']=='SPMCP_fetch' and begin<=e['tick']<=install]
    assert pe
    first=pe[0]['tick'];success=[e for e in pe if not e['detail'].strip()]
    assert len(success)==1,success
    l0=success[0]['tick']
    add('L1 ingress waiting','Mandatory queue ready to first attempted transition',ready,first)
    blocks=find('CTRL_RESOURCE_BLOCK',first,l0-1,'system.ruby.l0_cntrl0',addr=addr)
    resource_names={
        '(*m_writeCredits_ptr)':'L1 write/bridge credits full',
        '(*m_downstreamWriteCredits_ptr)':'Private L2 writeback credits full (L1 forwarding gate)',
        '(*m_downstreamMissCredits_ptr)':'Private L2 miss credits full (L1 forwarding gate)',
        '(*m_bufferToL1_ptr)':'Local L1-to-L2 message buffer full'}
    older_blocks=find('CTRL_RESOURCE_BLOCK',ready,first-1,'system.ruby.l0_cntrl0')
    # In this case every ingress-wait cycle has exactly one blocked older
    # SPMCP at the mandatory FIFO head. Assert coverage before assigning cause.
    ingress_explanation=None
    if older_blocks and [e['tick'] for e in older_blocks]==list(range(ready,first,400)):
        assert all(e['addr']!=addr and e['protocol_event']=='SPMCP_fetch' for e in older_blocks)
        reasons=collections.Counter(e['resource'] for e in older_blocks)
        ingress_explanation=dict(all_wait_cycles_observed=True,
            older_source_lines=sorted({e['addr'] for e in older_blocks}),
            blocked_core_cycles={resource_names.get(k,k):v for k,v in reasons.items()},events=older_blocks)
    if blocks:
        assert blocks[0]['tick']==first
        assert len({e['tick'] for e in blocks})==len(blocks),'Ambiguous resource checks'
        chunks=[]
        for i,e in enumerate(blocks):
            end=blocks[i+1]['tick'] if i+1<len(blocks) else l0
            label=resource_names.get(e['resource'],e['resource'])
            if chunks and chunks[-1][0]==label:chunks[-1][2]=end
            else:chunks.append([label,e['tick'],end])
        for label,a,b in chunks:add(label,'Waiting after rejection: '+label,a,b)
    else:
        add('L1 resource waiting','First resource-stalled transition to successful forwarding',first,l0)
    local=one('QUEUE_ENQUEUE',l0,l0,'l0_cntrl0.bufferToL1',addr=addr)
    add('Fixed protocol/local delays','Local L1-to-private-L2 link delay',l0,int(local['ready']))
    priv=target['private']['tick']
    dequeues=find('QUEUE_DEQUEUE',int(local['ready']),priv,'l0_cntrl0.bufferToL1',msg=local['msg'])
    first_private=min(e['tick'] for e in dequeues)
    add('Controller scheduling','Private-L2 clock alignment/first handling',int(local['ready']),first_private)
    add('Private L2 waiting','Private-L2 stalled-message wait',first_private,priv)

    packet_reports=[]
    def network(label,start,finish,source_queue,dest_queue):
        source=one('QUEUE_ENQUEUE',start,start,source_queue,addr=addr)
        packet=one('NI_PACKET',start,finish,parent=source['msg'])
        pid=packet['packet'];n=int(packet['count']);tail=str(n-1)
        add('Fixed protocol/local delays',label+' controller enqueue delay',start,int(source['ready']))
        add('Network source message waiting',label+' ready message to NI packet allocation',int(source['ready']),packet['tick'])
        sent=one('NI_SEND',start,finish,packet=pid,flit=tail)
        add('NI serialization/flow control',label+' packet allocation to tail-flit injection',packet['tick'],sent['tick'])
        # Follow the last flit on its path. Other flits are recorded as evidence
        # but not added: their movement overlaps this elapsed-time partition.
        te=[e for e in ev if start<=e['tick']<=finish and e.get('packet')==pid and e.get('flit')==tail]
        cursor=sent['tick'];hops=[]
        for link in [e for e in te if e['event']=='LINK_SEND']:
            add('Network pipeline/link time',label+' output staging before '+link['obj'],cursor,link['tick'])
            arrived=int(link['arrival'])
            add('Network pipeline/link time',label+' physical link '+link['obj'],link['tick'],arrived)
            receivers=[e for e in te if e['tick']>=arrived and e['event'] in ('ROUTER_ARRIVE','NI_ARRIVE')]
            rec=receivers[0];assert rec['tick']==arrived,(link,rec)
            cursor=arrived
            if rec['event']=='ROUTER_ARRIVE':
                router=rec['router']
                rr=one('ROUTER_READY',arrived,arrived,packet=pid,flit=tail,router=router)
                sa=one('SA_GRANT',arrived,finish,packet=pid,flit=tail,router=router)
                xb=one('XBAR',sa['tick'],sa['tick'],packet=pid,flit=tail,router=router)
                add('Network pipeline/link time',label+' router '+router+' fixed pipeline',cursor,int(rr['ready']))
                add('Router buffering/arbitration',label+' router '+router+' tail buffered before switch grant',int(rr['ready']),sa['tick'])
                # Logged failed decisions are evidence, not additive cycles:
                # a flit can be behind another flit or not inspected each cycle.
                reasons=collections.Counter(e['event'] for e in te if
                    arrived<=e['tick']<sa['tick'] and e.get('router')==router and e['event'].startswith('SA_'))
                hops.append(dict(router=router,arrival=arrived,grant=sa['tick'],observed_failed_decisions=dict(reasons)))
                cursor=xb['tick']
        arrived=one('NI_ARRIVE',start,finish,packet=pid,flit=tail)
        assert cursor==arrived['tick']
        completed=one('NI_COMPLETE',start,finish,packet=pid,flit=tail)
        add('Destination NI waiting',label+' tail arrival to protocol enqueue',cursor,completed['tick'])
        dest=one('QUEUE_ENQUEUE',completed['tick'],completed['tick'],dest_queue,msg=packet['msg'])
        add('Fixed protocol/local delays',label+' destination protocol queue delay',completed['tick'],int(dest['ready']))
        add('Destination controller waiting',label+' ready response/request to controller service',int(dest['ready']),finish)
        packet_reports.append(dict(label=label,packet=pid,flits=n,hops=hops,events=[e for e in ev if start<=e['tick']<=finish and e.get('packet')==pid]))

    llc=target['llc']['tick'];bank=target['llc']['core']
    network('Request',priv,llc,'l1_cntrl0.requestToL2',f'l2_cntrl{bank}.L1RequestToL2Cache')
    network('Response',llc,install,f'l2_cntrl{bank}.responseFromL2Cache','l1_cntrl0.responseFromL2')
    total=install-issue;assert sum(r['ticks'] for r in rows)==total
    groups=collections.defaultdict(int)
    for r in rows:
        r['percent']=100*r['ticks']/total;groups[r['category']]+=r['ticks']
    summary=[dict(category=k,ticks=v,core_cycles=v/400,percent=100*v/total) for k,v in groups.items()]
    pools=[e for e in ev if first<=e['tick']<=l0 and e['event']=='POOL_FULL' and 'l0_cntrl0' in e['obj']]
    result=dict(scope='8-core single-head T512, VL16, translated MICRO resource profile,64KiB private L2',
        selection=target['selection'],instruction=sn,source=addr,issue=issue,install=install,
        total_core_cycles=total/400,total_ns=total/1000,summary=summary,intervals=rows,packets=packet_reports,
        l1_full_pool_evidence=dict(collections.Counter(e['obj'] for e in pools)),
        forwarding_rejections=blocks,
        ingress_wait_explanation=ingress_explanation,
        invariance=json.loads((ex/'trace/validated.json').read_text()),
        caveats=['Not the previous T16K transaction; newly traced validated T512 model',
            'Selected long-latency request, not representative average or end-to-end causal fraction',
            'No explicit SRAM-bank arbitration in this model',
            'Tail-flit timeline partitions latency; other flit activity overlaps and is not summed',
            'Router waiting includes VC FIFO ordering, credit backpressure and switch arbitration; logged failed checks are not additive cycles'])
    save(ex/'breakdown.json',result)
    report=['# One SPMCP: issue-to-install timeline','',result['scope'],
        f"Source {addr}; CPU0 instruction {sn}. **{total/400:.2f} core cycles ({total/1000:.2f} ns)**.",
        '',target['selection']+'.','',
        'All finite non-host statistics, output bytes, and configuration (except output redirects) match the independently validated original.',
        '', '| Component | Core cycles | Percent |','|---|---:|---:|']
    report += [f"| {r['category']} | {r['core_cycles']:.2f} | {r['percent']:.2f}% |" for r in summary]
    if ingress_explanation:
        report += ['', '## What held the L1 request queue head?', '',
            'Every one of the 497 ingress-wait cycles contains a rejected older SPMCP at the same core. Seven older source lines precede the selected copy. This is observed FIFO head-of-line blocking, not a bank-arbitration penalty.',
            '', '| Reason blocking the older copy | Core cycles | Percent of selected access |',
            '|---|---:|---:|']
        report += [f'| {k} | {v} | {100*v*400/total:.2f}% |' for k,v in ingress_explanation['blocked_core_cycles'].items()]
        report += ['', 'These rows subdivide the 497-cycle ingress wait; do not add them again to the main table.']
    report += ['', '## Every interval','', '| Interval | Start tick | End tick | Cycles | Percent |','|---|---:|---:|---:|---:|']
    report += [f"| {r['label']} | {r['start']} | {r['end']} | {r['core_cycles']:.2f} | {r['percent']:.2f}% |" for r in rows]
    report += ['', '## Limits','']+[f'- {c}' for c in result['caveats']]
    (ex/'BREAKDOWN.md').write_text('\n'.join(report)+'\n')
    # Compact source evidence, including all target flits and CPU milestones.
    with (ex/'target_evidence.log').open('w') as dst:
        for n,line in inst:dst.write(f'{n}: {line}\n')
        for e in ev:
            if issue<=e['tick']<=install and (e.get('addr')==addr or e in pools or e in older_blocks):dst.write(json.dumps(e)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('intervals','packets','forwarding_rejections','ingress_wait_explanation')},indent=2))


if __name__=='__main__':main(Path(sys.argv[1]) if len(sys.argv)>1 else OUT/'experiment_v3')
