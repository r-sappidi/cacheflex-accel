"""Archive verified inactive build artifacts, then remove their original files."""
from pathlib import Path
import hashlib,json,subprocess,tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'archive/build_cleanup_20260916'
TREES=['build/ARM_MESI_Three_Level_SPM','gem5/build/ARM_MESI_Three_Level']
BACKUPS=['gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt.pre-fault-pc-debug',
         'gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt.stale-20260824-1458']

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    assert not OUT.exists(),OUT
    targets=[]
    for tree in TREES:
        for p in (ROOT/tree).rglob('*'):
            if p.is_file() and not p.is_symlink() and p.suffix in ('.o','.pyo','.os','.d'):
                targets.append(p)
    targets += [ROOT/p for p in BACKUPS]
    assert all(p.is_file() and not p.is_symlink() for p in targets)
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    assert not any(str(p.relative_to(ROOT)) in tracked for p in targets)
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try: command=proc.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:continue
        if any(str(p) in command for p in targets) or 'scons ' in command:
            raise RuntimeError(f'Build or target in use: {proc}: {command}')
    OUT.mkdir(parents=True)
    rows=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)) for p in targets]
    manifest=dict(files=rows,original_bytes=sum(r['bytes'] for r in rows))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    archive=OUT/'artifacts.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in targets:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    # Verify every archived byte before removing any original.
    with tarfile.open(archive,'r:gz') as tar:
        for row in rows:
            f=tar.extractfile(row['path']);h=hashlib.sha256()
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            assert h.hexdigest()==row['sha256'],row['path']
    for row in rows:
        p=ROOT/row['path'];assert digest(p)==row['sha256'],p
    for row in rows:(ROOT/row['path']).unlink()
    manifest.update(archive_bytes=archive.stat().st_size,
        approximate_net_bytes_reclaimed=manifest['original_bytes']-archive.stat().st_size,
        removed_files=len(rows),archive_sha256=digest(archive),complete=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))

if __name__=='__main__':main()
