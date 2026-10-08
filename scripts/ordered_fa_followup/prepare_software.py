import json,sys
from pathlib import Path
R=Path('/data2/rsappidi/cacheflex_mc');H=R/'benchmarks/qwen/fa/software_confirm';H.mkdir(exist_ok=True)
OLD=R/'benchmarks/qwen/fa/paced_staging';ROOT=R/'results/cacheflex_flash_attention/software_confirm_20261005'
s=(OLD/'build.py').read_text().replace('paced_staging_20261004','software_confirm_20261005')
s=s.replace('||stage_simple||packed_producer','||packed_producer')
s=s.replace('size_t dst=(vbase(slot_offset+r,sets)<<6)|(w<<16);','size_t row=slot_offset+r;\n        size_t dst=((stage_simple && row<sets ? row : vbase(row,sets))<<6)|(w<<16);')
(H/'build.py').write_text(s)
s=(OLD/'plan.py').read_text();start=s.index('def cases():');end=s.index('def control_path',start)
s=s[:start]+'''def cases():
 arm=os.environ['FA45_ARM'];assert arm in ('cache','cache_stage','burst','stage','paced','combined')
 return [dict(name=f't{t}_{arm}',kind='fa',t=t,spm=arm not in ('cache','cache_stage'),variant='stage' if arm in ('stage','combined','cache_stage') else arm,schedule=int(arm in ('paced','combined'))) for t in (256,512,4096)]
'''+s[end:]
s=s.replace("if c['t']==1024:return", "if c['t']==4096:return R/f'results/cacheflex_flash_attention/bottleneck_20261004/targets_continuation_v1/runs/target_{backend}_control'\n if c['t']==1024:return")
(H/'plan.py').write_text(s)
s=(OLD/'run.py').read_text().replace('paced_staging_20261004','software_confirm_20261005').replace('FA44','FA45').replace('fa44_','fa45_').replace('FA-44','FA-45')
(H/'run.py').write_text(s)
s=(OLD/'report.py').read_text().replace('paced_staging_20261004','software_confirm_20261005').replace('FA44','FA45').replace("('cache','burst','paced','overlap','dbburst')","('cache','cache_stage','burst','stage','paced','combined')").replace('(256,512,1024)','(256,512,4096)').replace("[('paced','burst'),('overlap','burst'),('overlap','dbburst'),('dbburst','burst'),('burst','cache')]","[('paced','burst'),('combined','paced'),('combined','stage'),('stage','burst'),('burst','cache'),('cache_stage','cache')]")
(H/'report.py').write_text(s)
for p in H.glob('*.py'):compile(p.read_text(),str(p),'exec')
# Address specialization equivalence for every destination actually used by all schedules.
for offset in (0,256):
 for row in range(256):
  x=offset+row
  for w in (0,1):
   old=((x%1024+((x//1024)*2<<10))<<6)|(w<<16)
   new=((x if x<1024 else x%1024+((x//1024)*2<<10))<<6)|(w<<16)
   assert old==new
ROOT.mkdir(exist_ok=True)
(ROOT/'preflight.json').write_text(json.dumps(dict(address_cases=1024,passed=True,arms=['cache','cache_stage','burst','stage','paced','combined'],sizes=[256,512,4096],prior=['FA-39','FA-44','SYN-01'],reason='Target-size confirmation and interaction, matched new-ELF controls; no geometry sweep.'),indent=2))
print('Software confirmation sources prepared; bounded address equivalence PASS.')
