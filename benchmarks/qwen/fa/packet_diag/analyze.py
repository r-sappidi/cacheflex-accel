"""Exact tail-flit interval accounting; conditional samples, not runtime fractions."""
import gzip,re,json,collections
from pathlib import Path
LINE=re.compile(r'^\s*(\d+):\s+(.*?):\s+CFT\s+(\w+)\s+(.*)$')
FIELDS=re.compile(r'(\w+)=([^\s]+)')
def parse(lines):
 packets={};events=collections.Counter();orphan=collections.Counter()
 for line in lines:
  m=LINE.match(line)
  if not m:continue
  tick,obj,event,rest=m.groups();tick=int(tick);f=dict(FIELDS.findall(rest));events[event]+=1
  if 'packet' not in f:continue
  pid=int(f['packet'])
  if event=='NI_PACKET':
   assert pid not in packets,'duplicate packet ID'
   packets[pid]=dict(id=pid,created=tick,src=int(f['src']),dst=int(f['dst']),vnet=int(f['vnet']),count=int(f['count']),routers=[],blocks=collections.Counter(),ni_samples=collections.Counter(),ni_ready={},sends=set())
  if pid not in packets:orphan[event]+=1;continue
  p=packets[pid]
  if event=='NI_CLASS':p.update(kind=f['kind'],type=f['type'],spm_dst=int(f['spm_dst'],0),born=int(f['born']),source_ready=int(f['source_ready']))
  if event.startswith('SA_') and event!='SA_GRANT':p['blocks'][event]+=1
  if event=='NI_STATE':
   # One snapshot per VC head per NI arbitration opportunity. Count samples,
   # not elapsed cycles or additive critical-path stall fractions.
   key=(tick,int(f['flit']));p['ni_ready'][key]=(int(f['credit']),int(f['older']))
  if event=='NI_SEND':p['sends'].add((tick,int(f['flit'])))
  if 'flit' not in f or int(f['flit'])!=p['count']-1:continue
  if event=='NI_SEND':p['send']=tick
  elif event=='ROUTER_ARRIVE':p['routers'].append(dict(router=int(f['router']),arrive=tick))
  elif event in ('ROUTER_READY','SA_GRANT'):
   assert p['routers'] and p['routers'][-1]['router']==int(f['router'])
   p['routers'][-1]['ready' if event=='ROUTER_READY' else 'grant']=int(f['ready']) if event=='ROUTER_READY' else tick
  elif event=='NI_ARRIVE':p['arrive']=tick
  elif event=='NI_COMPLETE':p['complete']=tick
  elif event=='NI_EJECT_STALL':p['eject_quota']=bool(int(f['quota']));p['eject_direct']=bool(int(f['direct']))
 rows=[];censored=0
 for p in packets.values():
  if not all(k in p for k in ('send','arrive','complete','kind')) or not p['routers'] or any(not all(k in r for k in ('ready','grant')) for r in p['routers']):censored+=1;continue
  for (t,fl),(credit,older) in p['ni_ready'].items():
   label='no_credit' if not credit else 'ordering' if older else 'selected' if (t,fl) in p['sends'] else 'eligible_not_selected'
   p['ni_samples'][label]+=1
  rs=p['routers'];pipeline=sum(r['ready']-r['arrive'] for r in rs);waiting=sum(r['grant']-r['ready'] for r in rs)
  transport=rs[0]['arrive']-p['send']+sum(b['arrive']-a['grant'] for a,b in zip(rs,rs[1:]))+p['arrive']-rs[-1]['grant']
  parts=dict(source_message_age=p['created']-p['born'],source_ready_wait=p['created']-p['source_ready'],ni_tail_wait=p['send']-p['created'],router_pipeline=pipeline,router_wait=waiting,crossbar_and_links=transport,destination_wait=p['complete']-p['arrive'])
  assert all(v>=0 for v in parts.values()),parts
  assert sum(v for k,v in parts.items() if k not in ('source_message_age','source_ready_wait'))==p['complete']-p['created'],'interval conservation'
  if p['type']=='DATA' and p['kind']=='response':group='copy_DATA' if p['spm_dst'] else 'coherent_DATA'
  else:group=p['kind']+'_'+p['type']
  rows.append(dict(packet=p['id'],src=p['src'],dst=p['dst'],group=group,created=p['created'],count=p['count'],parts_ticks=parts,router_hops=rs,router_block_samples=dict(p['blocks']),ni_samples=dict(p['ni_samples']),eject_quota=p.get('eject_quota'),eject_direct=p.get('eject_direct')))
 groups={}
 for row in rows:
  g=groups.setdefault(row['group'],dict(packets=0,parts_ticks=collections.Counter(),router_block_samples=collections.Counter(),ni_samples=collections.Counter(),router_wait_ticks=collections.Counter()))
  g['packets']+=1
  for field in ('parts_ticks','router_block_samples','ni_samples'):g[field].update(row[field])
  for hop in row['router_hops']:g['router_wait_ticks'][str(hop['router'])]+=hop['grant']-hop['ready']
 for g in groups.values():g['mean_ns']={k:v/g['packets']/1000 for k,v in g['parts_ticks'].items()}
 return dict(events=dict(events),orphan_events=dict(orphan),packets_created=len(packets),complete_packets=len(rows),censored_packets=censored,groups=groups,packets=rows,limitations=['Only packets created and completed inside the window; censored packets excluded explicitly.','Tail-flit intervals partition latency, not FA runtime. NI wait includes serialization and effects of earlier flits.','Router block samples cover all flits; they are not additive stall-cycle fractions.','Source message age includes configured controller delay, not just queue contention.','Credit stalls identify blocked flow control, not automatically its ultimate downstream cause.'])
def analyze(path):
 with gzip.open(path/'trace.gz','rt',errors='replace') as f:result=parse(f)
 assert result['complete_packets']>0,'no complete packets captured'
 assert any(k.endswith('DATA') for k in result['groups']),'no classified DATA'
 config=json.loads((path/'config.json').read_text())
 net=config['system']['ruby']['network']
 nodes={int(x['link_id']):x['ext_node'] for x in net['ext_links']}
 # Current topology assigns contiguous external link / NI IDs. Fail closed
 # if that mapping changes; do not silently relabel a different topology.
 assert sorted(nodes)==list(range(len(nodes)))
 assert sorted(int(x['id']) for x in net['netifs'])==list(range(len(nodes)))
 result['ni_nodes']=nodes;routes={};bins={}
 for row in result['packets']:
  row['src_node']=nodes[row['src']];row['dst_node']=nodes[row['dst']]
  key=row['src_node']+'->'+row['dst_node']+'/'+row['group']
  g=routes.setdefault(key,dict(packets=0,parts_ticks=collections.Counter()))
  g['packets']+=1;g['parts_ticks'].update(row['parts_ticks'])
  bucket=str(row['created']//1_000_000)
  b=bins.setdefault(bucket,collections.Counter());b[row['group']]+=1
 for g in routes.values():g['mean_ns']={k:v/g['packets']/1000 for k,v in g['parts_ticks'].items()}
 result['routes']=routes;result['complete_packet_creation_bins_1us']=bins
 (path/'packet_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 return {k:v for k,v in result.items() if k!='packets'}
def report(root):
 s=json.loads((root/'status.json').read_text());lines=['FA-42 packet diagnostic: bounded samples only, not full-ROI performance results.','']
 for row in s['completed']:
  p=Path(row['path'])/'packet_analysis.json'
  if not p.exists():continue
  d=json.loads(p.read_text());lines.append(row['name']+f": {d['complete_packets']} complete / {d['packets_created']} created")
  for k in ('copy_DATA','coherent_DATA'):
   if k in d['groups']:
    g=d['groups'][k];lines.append(f"  {k}: n={g['packets']}, mean ns={g['mean_ns']}; NI samples={g['ni_samples']}; router block samples={g['router_block_samples']}")
 lines+=['','No unique bottleneck or speedup claim is automatically inferred from samples.']
 (root/'REPORT.txt').write_text('\n'.join(lines)+'\n')
