"""Run symbol-paired SPM tracing and independent-SPM-capacity ablation.

Usage: run_spm_capacity_probe.py NEW_OUT BUILT_SIMULATOR
Original guests are retained. New SPM production is stripped from its debug
side ELF, guaranteeing identical .text rather than borrowing mismatched PCs.
"""
from pathlib import Path
import concurrent.futures as cf
import hashlib,json,os,shutil,subprocess,sys
from validate_attribution import read
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
out=Path(sys.argv[1]).resolve();out.mkdir()
shutil.copy2(sys.argv[2],out/'gem5.opt')
old=ROOT/'results/cacheflex_flash_attention/clean_vl16_20260908/runs/guests'
debug=ROOT/'results/cacheflex_flash_attention/residency_symbols_20260909'
for variant in ('cache','spm'):
 shutil.copy2(old/variant,out/('legacy_'+variant))
 shutil.copy2(debug/(variant+'.debug'),out/(variant+'.debug'))
subprocess.run(['aarch64-linux-gnu-objcopy','--strip-debug',str(out/'spm.debug'),str(out/'spm')],check=True)
shutil.copy2(out/'legacy_cache',out/'cache')
for variant in ('cache','spm'):
 for name in (variant,variant+'.debug'):
  subprocess.run(['aarch64-linux-gnu-objcopy','--dump-section',f'.text={out/name}.text',str(out/name),str(out/(name+'.objcopy'))],check=True)
 assert (out/(variant+'.text')).read_bytes()==(out/(variant+'.debug.text')).read_bytes()
(out/'hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2))
for name in ('run_spm_capacity_probe.py','analyze_residency.py','analyze_spm_residency.py','run_8core.sh'):
 shutil.copy2(HERE/name,out/name)
state=dict(stage='starting',complete=[],failed=None);rows=[]
def status(stage):
 state['stage']=stage;(out/'status.json').write_text(json.dumps(state,indent=2));print(stage,flush=True)
def run(variant,t,separate,trace):
 name=f'{variant}_t{t}_separate{separate}_trace{trace}';path=out/name
 spm='spm' in variant
 env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
  CACHEFLEX_FLASH_VL='16',CACHEFLEX_FLASH_MATCHED_LRU='1',CACHEFLEX_FLASH_PROFILE='1',
  CACHEFLEX_FLASH_SPM_WAYS='4' if spm else '0',CACHEFLEX_FLASH_SPM_SEPARATE_BACKING=str(separate),
  CACHEFLEX_FLASH_BIN=str(out/variant),CACHEFLEX_GEM5_BIN=str(out/'gem5.opt'),CACHEFLEX_FLASH_OUT=str(path),
  CACHEFLEX_FLASH_DEBUG_FLAGS='CacheFlexResidency,ProtocolTrace' if trace else '')
 with (out/(name+'.host.log')).open('w') as log:
  assert (out/'run_8core.sh').read_bytes()==(HERE/'run_8core.sh').read_bytes(), 'runner changed during study'
  subprocess.run(['bash',str(HERE/'run_8core.sh'),'spm_konly_overlap' if spm else 'cache_pretouch',str(t),'64','128'],
                 env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 output,stats,cfg=read(path,16)
 expected=8*(t//128)*64*4 if spm else 0
 assert stats['m_spmcpRequestsIssued']==stats['m_spmcpRequestsCompleted']==expected
 for c in range(8):
  cache=cfg[f'system.ruby.l1_cntrl{c}.cache']
  assert cache['size']=='524288' and cache['assoc']=='8'
  assert cache['spm_separate_backing']==('true' if separate else 'false')
 return dict(variant=variant,t=t,separate=separate,trace=trace,path=str(path),output=output,counters=stats),cfg
def config(a,b):
 assert a.sections()==b.sections()
 for section in a.sections():
  assert set(a[section])==set(b[section])
  for k,x in a[section].items():
   y=b[section][k]
   if x==y:continue
   if section=='system.cpu0.workload' and k in ('cmd','executable'):continue
   assert section.startswith('system.ruby.l1_cntrl') and section.endswith('.cache') and k in ('num_spm_ways','spm_separate_backing'),(section,k,x,y)
def analyze(row):
 name=Path(row['path']).name;status('analyze_'+name)
 with (out/(name+'.analysis.log')).open('w') as log:
  subprocess.run([sys.executable,str(out/'analyze_spm_residency.py'),row['path'],str(out/(row['variant']+'.debug')),str(row['t']),row['variant']],
                 env=dict(os.environ,PYTHONPATH=str(HERE)),stdout=log,stderr=subprocess.STDOUT,check=True)
try:
 for t in (128,1024):
  status(f't{t}_controls')
  cases=[('cache',t,0,0),('spm',t,0,0),('spm',t,1,0),('legacy_spm',t,0,0)]
  with cf.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(lambda args:run(*args),cases))
  reference,refcfg=results[0]
  for row,cfg in results:
   assert row['output']['hash']==reference['output']['hash'];config(refcfg,cfg);rows.append(row)
  (out/'summary.json').write_text(json.dumps(rows,indent=2))
  # Trace each timed configuration on the very same executable/configuration.
  for control,cfg in results[:3]:
   traced,tracecfg=run(control['variant'],t,control['separate'],1)
   assert traced['output']==control['output'];config(cfg,tracecfg)
   for k in control['counters'].keys()-{'path'}:assert control['counters'][k]==traced['counters'][k],(k,'observer changed counter')
   rows.append(traced);(out/'summary.json').write_text(json.dumps(rows,indent=2));analyze(traced)
  state['complete'].append(f't{t}');status(f't{t}_complete')
 report=['# SPM separate-backing diagnostic','',
  'Extra SPM backing retains 512 KiB / 8 coherent ways plus four SPM-only ways (256 KiB extra per core).',
  'Set mapping, ISA addresses, ports, modeled latency and source coherence remain unchanged.',
  'This is a capacity-cost ablation, not a result for the area-matched CacheFlex implementation.','',
  '| T | Cache us | SPM shared us | SPM separate us | Shared/separate | Legacy SPM us |',
  '|---:|---:|---:|---:|---:|---:|']
 for t in (128,1024):
  r={(r['variant'],r['separate']):r['counters']['roi_us'] for r in rows if r['t']==t and not r['trace']}
  a,b,c,d=r['cache',0],r['spm',0],r['spm',1],r['legacy_spm',0]
  report.append(f'| {t} | {a:.4f} | {b:.4f} | {c:.4f} | {b/c:.4f} | {d:.4f} |')
 report+=['','SPM guest is a symbol-paired rebuild, not the legacy ELF. Legacy is retained as a separate timing/output control.',
  'Qualified removal files distinguish protocol reasons and ambiguous writes before later retired reads.',
  'These events and overlapping request service times are not additive causal penalties.']
 (out/'REPORT.md').write_text('\n'.join(report)+'\n');status('complete')
except BaseException as error:
 state['failed']=repr(error);status('failed');raise
