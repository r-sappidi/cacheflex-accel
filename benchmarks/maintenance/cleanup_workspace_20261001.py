"""Explicit, dependency-checked cleanup; never recursively deletes research data."""
import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'archive/workspace_cleanup_20261001'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,d):(OUT/n).write_text(json.dumps(d,indent=2)+'\n')
def live_paths():
 paths=set()
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if proc.stat().st_uid!=os.getuid():continue
   for p in [proc/'exe',proc/'cwd',*list((proc/'fd').iterdir())]:
    try:paths.add(p.resolve())
    except OSError:pass
   for line in (proc/'maps').read_text().splitlines():
    s=line.split(maxsplit=5)
    if len(s)==6 and s[5].startswith('/'):paths.add(Path(s[5]).resolve())
   for s in (proc/'cmdline').read_bytes().decode().split('\0'):
    q=Path(s)
    if s and not s.startswith('-') and (q.is_absolute() or s.startswith('scripts/')):
     p=q if q.is_absolute() else ROOT/q
     if p.exists():paths.add(p.resolve())
  except (OSError,UnicodeError):pass
 return paths

def main():
 os.nice(10)
 OUT.mkdir(exist_ok=False)
 manifests=[ROOT/x for x in ['results/cacheflex_flash_attention/qwen_matrix_20260930/experiment_v4/inputs.json','results/cacheflex_l3_spm/qwen_long_context_20260928/experiment_v1/inputs.json','results/rumur_refresh_20261001/verification_v5/inputs.json']]
 protected={}
 for f in manifests:
  for name,h in json.loads(f.read_text()).items():
   p=Path(name).resolve()
   assert sha(p)==h,('frozen input mismatch',name)
   protected[p]=h
 save('protected.json',{str(p):h for p,h in protected.items()})
 cv=list(ROOT.glob('rsappidi_cv.*'))+list(ROOT.glob('pdflatex*.fls'))+[p for p in (ROOT/'build').iterdir() if p.is_file()]
 # Root build loose files were inspected: all are CV previews, tex, or latex logs.
 remove=cv+[ROOT/'--output',ROOT/'--results']
 retire=list((ROOT/'build/ARM_MESI_Three_Level_SPM').rglob('*'))+list((ROOT/'verification/rumur/build').rglob('*'))+list(ROOT.glob('*build.log'))+list(ROOT.glob('gem5-*.log'))
 retire += [ROOT/'scripts'/n for n in ['run_rumur_refresh.py','run_rumur_refresh_v2.py','run_rumur_refresh_v3.py','run_rumur_refresh_v4.py']]
 retire=sorted({p for p in retire if p.is_file() and not p.is_symlink()})
 tracked={ROOT/p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if p}
 opened=live_paths()
 for p in remove+retire:
  assert p.resolve()==p and p not in protected and p not in opened and p not in tracked,('protected candidate',str(p))
 rows=[dict(path=str(p.relative_to(ROOT)),size=p.stat().st_size,sha256=sha(p)) for p in retire]
 deleted=[dict(path=str(p.relative_to(ROOT)),size=p.stat().st_size,sha256=sha(p),reason='user-requested CV removal' if p in cv else 'empty accidental CLI output') for p in remove]
 save('archived_manifest.json',rows);save('deleted_manifest.json',deleted)
 archive=OUT/'retired.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as tf:
  for p in retire:tf.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
 with tarfile.open(archive,'r:gz') as tf:
  for r in rows:assert hashlib.sha256(tf.extractfile(r['path']).read()).hexdigest()==r['sha256']
 opened=live_paths()
 for r in rows+deleted:
  p=ROOT/r['path'];assert p not in opened and sha(p)==r['sha256'];p.unlink()
 for base in [ROOT/'build',ROOT/'verification/rumur/build']:
  for p in sorted([x for x in base.rglob('*') if x.is_dir()],key=lambda x:len(x.parts),reverse=True):
   try:p.rmdir()
   except OSError:pass
  try:base.rmdir()
  except OSError:pass
 for p,h in protected.items():assert sha(p)==h,('post-cleanup drift',str(p))
 result=dict(archived_files=len(rows),removed_cv_and_empty_files=len(deleted),protected_files_verified=len(protected),original_bytes=sum(x['size'] for x in rows+deleted),archive_bytes=archive.stat().st_size,net_bytes_reclaimed=sum(x['size'] for x in rows+deleted)-archive.stat().st_size,finished=time.time(),recovery=f'tar -xzf {archive.relative_to(ROOT)} -C .',note='Recovery restores archived build artifacts/logs/retired launchers, not deleted CV material.')
 save('complete.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
