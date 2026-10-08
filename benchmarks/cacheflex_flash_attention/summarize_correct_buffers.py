"""Buffer/protocol corroboration for the same-guest capacity intervention."""
from pathlib import Path
from collections import Counter
import csv,json,re,sys
root=Path(sys.argv[1]).resolve();runs=['cache_g4_h0','spm_g4_h0','spm_g4_h8'];result={}
for name in runs:
    p=root/name;r=json.loads((p/'qualified_removals.json').read_text())
    totals=Counter();strict=Counter();other_reuse=Counter();events=Counter();phases=Counter();sets=Counter();per_core=Counter()
    for key,count in r['counts'].items():
        cache,buf,event,reuse=key.split('|');buf=re.sub(r'^core\d+\.','',buf)
        if not re.fullmatch(r'system.ruby.l1_cntrl\d+.cache',cache):continue
        totals[buf]+=count;events[(buf,event,reuse)]+=count
        if reuse=='post_removal_request_and_retired_read_without_store':
            strict[buf]+=count;per_core[(cache,buf)]+=count
        if reuse=='retired_read_without_post_removal_request':other_reuse[buf]+=count
    with (p/'qualified_removals.csv').open() as f:
        for row in csv.DictReader(f):
            if not re.fullmatch(r'system.ruby.l1_cntrl\d+.cache',row['cache']):continue
            if row['reuse_class']!='post_removal_request_and_retired_read_without_store':continue
            buf=re.sub(r'^core\d+\.','',row['buffer']);phases[(buf,row['head_phase'],row['reuse_phase'])]+=1
            sets[(buf,'claimed_set_range_0_127' if ((int(row['addr'],16)>>6)&1023)<128 else 'other_sets')]+=1
    result[name]=dict(total_removals=dict(totals),read_reused_removals=dict(strict),read_without_request=dict(other_reuse),
        events={'|'.join(k):v for k,v in events.items()},phases={'|'.join(k):v for k,v in phases.items()},
        sets={'|'.join(k):v for k,v in sets.items()},per_core={'|'.join(k):v for k,v in per_core.items()},unresolved_protocol=r['unresolved_protocol'])
(root/'buffer_summary.json').write_text(json.dumps(result,indent=2))
keys=set().union(*(set(x['total_removals']) for x in result.values()))
order=sorted(keys,key=lambda k:result['spm_g4_h0']['read_reused_removals'].get(k,0)-result['spm_g4_h8']['read_reused_removals'].get(k,0),reverse=True)
text=['# Private-L2 buffer removal/reuse evidence','',
      'Only in-ROI retired ordinary reads with a post-removal request and no intervening observed store qualify below. SPMCP source reuse is excluded. Counts are NOT causal cycle savings.','',
      '| Buffer | Cache qualified removals | SPM | Separate-backing SPM | SPM minus separate |',
      '|---|---:|---:|---:|---:|']
for k in order:
    a,b,c=[result[n]['read_reused_removals'].get(k,0) for n in runs]
    text.append(f'| {k} | {a} | {b} | {c} | {b-c} |')
(root/'BUFFER_REPORT.md').write_text('\n'.join(text)+'\n');print('\n'.join(text))
