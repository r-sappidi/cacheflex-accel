"""Recoverable pruning of one completed ACL build's regenerable objects."""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
TREE=ROOT/'results/ai_workloads/mlp_acl_20260909/upstream/source/build'
OUT=ROOT/'results/build_cleanup_20260920'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    processes=subprocess.check_output(['ps','-eo','args'],text=True)
    assert not any('scons' in line and 'python' in line for line in processes.splitlines()),'build active'
    assert TREE.is_dir() and not TREE.is_symlink()
    files=sorted(p for p in TREE.rglob('*.o') if p.is_file() and not p.is_symlink())
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    assert all(str(p.relative_to(ROOT)) not in tracked for p in files)
    library=TREE/'libarm_compute-static.a';before=sha(library)
    OUT.mkdir();archive=OUT/'completed_acl_objects.tar.gz'
    manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        for r in manifest:assert hashlib.sha256(tar.extractfile(r['path']).read()).hexdigest()==r['sha256']
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for p,r in zip(files,manifest):
        assert p.is_relative_to(TREE) and p.suffix=='.o' and sha(p)==r['sha256']
        p.unlink()
    assert sha(library)==before
    result=dict(removed_objects=len(files),logical_bytes=sum(r['bytes'] for r in manifest),archive_bytes=archive.stat().st_size,library_sha256_unchanged=before,recovery='Extract completed_acl_objects.tar.gz from repository root; objects also regenerable. Sources, libraries, executables, traces/results, toolchain runtime objects and current Ruby build preserved.')
    (OUT/'completed.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
