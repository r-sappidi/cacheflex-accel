"""Reversibly archive only untracked old build intermediates in exact roots."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
import time

ROOTS = [Path('/data2/rsappidi/gem5/build/ARM'),
         Path('/data2/rsappidi/gem5_benchmark/build/ARM'),
         Path('/data2/rsappidi/cacheflex_micro/gem5/build/ARM')]
OUT = Path('/data2/rsappidi/cleanup_20260917')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()


def main():
    OUT.mkdir()
    summaries=[]
    for root in ROOTS:
        assert root.resolve()==root and root.is_dir()
        repo=root.parent.parent
        tracked=set(subprocess.check_output(['git','-C',str(repo),'ls-files','-z','build']).decode().split('\0'))
        selected=[]
        for p in root.rglob('*'):
            if p.suffix not in ('.o','.os','.pyc','.pyo') or p.is_symlink():continue
            s=p.stat()
            if not stat.S_ISREG(s.st_mode) or s.st_mtime>time.time()-7*86400:continue
            if str(p.relative_to(repo)) in tracked:continue
            selected.append((p,s))
        selected.sort()
        tag='micro' if 'cacheflex_micro' in root.parts else repo.name
        manifest=[dict(path=str(p),size=s.st_size,blocks=s.st_blocks,inode=s.st_ino,mtime_ns=s.st_mtime_ns,sha256=sha(p)) for p,s in selected]
        (OUT/f'{tag}.manifest.json').write_text(json.dumps(manifest,indent=2))
        if not manifest:continue
        # Refuse cleanup if a selected file is currently mapped/open.
        targets={str(p) for p,_ in selected}
        for process in Path('/proc').iterdir():
            if not process.name.isdigit():continue
            try:
                for fd in (process/'fd').iterdir():
                    assert os.readlink(fd) not in targets, ('open file',process,fd)
                maps=(process/'maps').read_text()
                assert not any(line.split()[-1] in targets for line in maps.splitlines() if line.split()), ('mapped file',process)
            except (PermissionError,FileNotFoundError,ProcessLookupError):pass
        archive=OUT/f'{tag}.intermediates.tar.gz'
        print('ARCHIVE',root,len(manifest),sum(r['size'] for r in manifest),flush=True)
        with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
            for p,_ in selected:tar.add(p,arcname=str(p.relative_to(repo)),recursive=False)
        # Verify every archived payload before unlinking any original.
        with tarfile.open(archive,'r:gz') as tar:
            members=tar.getmembers();assert len(members)==len(manifest)
            for member,row in zip(members,manifest):
                assert member.isfile() and member.name==str(Path(row['path']).relative_to(repo))
                h=hashlib.sha256();stream=tar.extractfile(member)
                for data in iter(lambda:stream.read(4*1024*1024),b''):h.update(data)
                assert h.hexdigest()==row['sha256'],member.name
        for row in manifest:
            p=Path(row['path']);s=p.lstat()
            assert stat.S_ISREG(s.st_mode) and (s.st_ino,s.st_size,s.st_mtime_ns)==(row['inode'],row['size'],row['mtime_ns'])
        for row in manifest:Path(row['path']).unlink()
        summary=dict(root=str(root),files=len(manifest),original_allocated_bytes=sum(r['blocks']*512 for r in manifest),archive_bytes=archive.stat().st_blocks*512,archive=str(archive),archive_sha256=sha(archive))
        summaries.append(summary)
        (OUT/'summary.json').write_text(json.dumps(summaries,indent=2))
        print('COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
