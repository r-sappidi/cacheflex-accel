"""Keep shared admission live if the original long-run supervisor finishes first.
Read-only until clean original completion; retain all reservations and limits.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc');sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'results/parallel_overlap_20260930')]
import broker
from parallel_budget_scheduler import owned
while True:
 time.sleep(5)
 procs=owned()
 with broker.locked() as b:
  if time.time()-b['heartbeat']<20:continue
  if not b['leases']:break
  old=json.loads((ROOT/'results/parallel_memory_recovery_20260929/status.json').read_text())
  if procs.get(old['pid'],{}).get('state','Z')!='Z':continue
  if not (ROOT/'results/parallel_overlap_20260930/complete.json').exists():continue
  if not all(k.startswith('fa35_') or k=='formal_refresh' for k in b['leases']):continue
  rss=sum(p['rss'] for p in procs.values())
  available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
  hold=rss>b['cap']-2*2**30 or available<64*2**30+max(0,b['cap']-rss)
  b['heartbeat']=time.time();b['hold_new']=hold;b['continuation_owner']=os.getpid()
