import ast,hashlib,json,os,sys,time
from pathlib import Path
R=Path('/data2/rsappidi/cacheflex_mc'); D=R/'results/ordered_fa_followup_20261005';D.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
# Host process identity, not historical namespace PIDs.
live=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  if p.stat().st_uid!=os.getuid():continue
  args=(p/'cmdline').read_bytes().split(b'\0');exe=args[0].decode() if args else ''
  if exe.endswith('gem5.opt') or any(a.endswith(b'/spm_isolation/run.py') or a.endswith(b'/iterative/run.py') for a in args):live.append((p.name,args))
 except (OSError,ProcessLookupError):pass
assert not live,live
sources={'fa40':R/'results/cacheflex_flash_attention/spm_isolation_20261004/experiment_v1','llama':R/'results/llama_fa_iterative_20261004/experiment_v1'}
receipt={}
for family,old in sources.items():
 state=read(old/'status.json');assert time.time()-state['updated']>300
 frozen=read(old/'inputs.json');bad=[p for p,h in frozen.items() if sha(Path(p))!=h];assert not bad,bad
 evidence={}
 for row in state['completed']:
  path=Path(row['path']);assert read(path/'accepted.json')==row
  if row.get('output_sha256'):assert sha(path/'output.fp16')==row['output_sha256']
  for f in path.iterdir():
   if f.is_file():evidence[str(f)]=sha(f)
 partial={str(f):sha(f) for a in state['active'].values() for f in Path(a['path']).iterdir() if f.is_file()}
 receipt[family]=dict(source=str(old),state=state,inputs=frozen,accepted_evidence=evidence,partial_evidence=partial)
 save(D/(family+'_prior_status.json'),state)
new=D/'fa40';new.mkdir(exist_ok=True)
old=sources['fa40'];src=R/'benchmarks/qwen/fa/spm_isolation/run.py';s=src.read_text()
s=s.replace('R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent',f"R=Path({str(R)!r});H=Path({str(src.parent)!r});sys.path.insert(0,str(H))")
s=s.replace("OUT=BASE/'experiment_v1'",f"OUT=Path({str(new)!r})")
s=s.replace('OUT.mkdir(parents=True,exist_ok=False);(OUT/\'runs\').mkdir();(OUT/\'references\').mkdir()',"OUT.mkdir(parents=True,exist_ok=True);(OUT/'runs').mkdir();(OUT/'references').mkdir()")
s=s.replace('plan=cases();state=',"prior=read(Path("+repr(str(old))+"));plan=cases();plan=[c for c in plan if c['name'] not in {r['name'] for r in prior['completed']}];state=".replace('read(Path('+repr(str(old))+'))','read(Path('+repr(str(old/'status.json'))+'))'))
s=s.replace('completed=[],failed=[])',"completed=prior['completed'],failed=[])")
s=s.replace("frozen={str(p):sha(p) for p in G.iterdir() if p.is_file()};",f"frozen=read(Path({str(old/'inputs.json')!r}));frozen.update({{str(p):sha(p) for p in G.iterdir() if p.is_file()}});")
s=s.replace("subprocess.run([sys.executable,str(H/'report.py')],cwd=R,env=ENV,check=True)","save(OUT/'accepted_summary.json',state['completed'])")
# Preserve exact numerical/config/resource validation body.
a=src.read_text().split("log=(path/'run.log').read_text();assert code==0",1)[1].split('unchanged();save',1)[0]
b=s.split("log=(path/'run.log').read_text();assert code==0",1)[1].split('unchanged();save',1)[0];assert a==b
compile(s,'fa40_run','exec');(D/'fa40_run.py').write_text(s)
# Prepare Llama continuation, keep its adaptive selection and all validation unchanged.
src=R/'benchmarks/llama/fa/iterative/run.py';s=src.read_text();old=sources['llama'];out=D/'llama'
s=s.replace('H=Path(__file__).resolve().parent',f"H=Path({str(src.parent)!r});sys.path.insert(0,str(H))")
s=s.replace("OUT=BASE/'experiment_v1'",f"OUT=Path({str(out)!r})")
s=s.replace("state=dict(pid=os.getpid(),stage='preparing',active={},pending=[],completed=[],failed=[],decisions=[],backend_status={})",f"state=read(Path({str(old/'status.json')!r}));state.update(pid=os.getpid(),stage='preparing',active={{}},pending=[],failed=[])")
s=s.replace("frozen=read(G/'frozen.json')",f"frozen=read(Path({str(old/'inputs.json')!r}));frozen.update(read(G/'frozen.json'))")
compile(s,'llama_run','exec');(D/'llama_run.py').write_text(s)
a=src.read_text().split("log=(path/'run.log').read_text();assert proc.returncode==0",1)[1].split('unchanged();save',1)[0];b=s.split("log=(path/'run.log').read_text();assert proc.returncode==0",1)[1].split('unchanged();save',1)[0];assert a==b
sys.path.insert(0,str(R/'results/parallel_overlap_20260930'));import broker
with broker.locked() as ledger:
 receipt['broker_before']=dict(ledger)
 for name in ('fa40_target_spm_control','fa41_t4096_cache_br16_bc128_r256'):
  assert ledger['leases'][name]['owner']==2;del ledger['leases'][name]
receipt['verified_at']=time.time();save(D/'recovery_verified.json',receipt)
print('Verified frozen sources, accepted evidence and preserved partials; released two orphaned leases. Continuations prepared.')
