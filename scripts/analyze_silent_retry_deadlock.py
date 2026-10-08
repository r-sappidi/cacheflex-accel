"""Reconstruct observed FIFO occupants and blocked flits from preserved replay."""
import collections
import gzip
import re
from pathlib import Path
from build_access_timing import save

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_flash_attention/spm_set_spread_20260927/diagnosis'

def main():
    queues=collections.defaultdict(dict)
    enqueued={}
    blocked={}
    last_movement=0
    heads={}
    histories=collections.deque(maxlen=100)
    with gzip.open(BASE/'protocol.gz','rt') as f:
        for line in f:
            if 'CFT QUEUE_' in line:
                t,node,body=line.rstrip().split(': ',2)
                tick=int(t); fields=dict(re.findall(r'(\w+)=(\S+)',body))
                msg=fields.get('msg');event=body.split()[1]
                if event=='QUEUE_ENQUEUE':
                    queues[node][msg]=dict(tick=tick,**fields)
                    enqueued[node]=msg
                elif event=='QUEUE_DEQUEUE': queues[node].pop(msg,None)
            elif 'Enqueue arrival_time:' in line and 'RequestMsg:' in line:
                t,node,body=line.rstrip().split(': ',2)
                msg=enqueued.get(node)
                if msg in queues[node]:
                    queues[node][msg]['message']=body.split(' DataBlk =')[0]
            if int(line.lstrip().split(':',1)[0].split()[0]) >=1089500000:
                if 'CFT POOL_ALLOC' in line or 'CFT POOL_RELEASE' in line:
                    last_movement=int(line.split(':',1)[0])
                if 'CFT ' in line and any(s in line for s in ('CTRL_RESOURCE_BLOCK','NI_NO_VC','SA_NO_VC','SA_NO_CREDIT','NI_STALL')):
                    t,node,body=line.rstrip().split(': ',2)
                    # Collapse repeated identical waits, retaining concrete packet IDs.
                    blocked[node+': '+body]=int(t)
                if 'Message: [RequestMsg:' in line:
                    t,node,body=line.rstrip().split(': ',2)
                    heads[node]=body.split(' DataBlk =')[0]
            if 'L1Cache' in line and '[0x1013c0,' in line and 'Stall' not in line:
                histories.append(line.strip())
    selected={k:list(v.values()) for k,v in queues.items() if v and
              any(x in k for x in ('requestToL2','requestFromL2','L1RequestToL2Cache','L1RequestFromL2Cache'))}
    save(BASE/'fifo_evidence.json',dict(final_window_begin=1089500000,
        last_pool_movement_in_final_window=last_movement,queues=selected,
        blocked=blocked,heads=heads,line_1013c0_history=list(histories),
        caveat='Trace-window reconstruction; pre-window occupants may be absent. Not exhaustive liveness proof.'))
    for key in ('system.ruby.l1_cntrl0.requestToL2','system.ruby.l1_cntrl0.requestFromL2',
                'system.ruby.l2_cntrl7.L1RequestFromL2Cache'):
        q=selected.get(key,[])
        print(key, 'observed occupancy',len(q), 'head',q[:1])
    print('wrote', BASE/'fifo_evidence.json')

if __name__=='__main__':main()
