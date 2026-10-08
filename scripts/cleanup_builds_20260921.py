"""Explicit, audited cleanup of cancelled MICRO builds and obsolete objects.

Default writes a reviewable plan. --execute validates and removes only that plan.
Sources, toolchains, result logs, and active experiment dependencies are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time

ROOT = Path('/data2/rsappidi/cacheflex_mc')
OUT = ROOT / 'results/cleanup_builds_20260921'
FA = ROOT / 'results/cacheflex_flash_attention'
ACTIVE = [FA/'async_staging_20260921/experiment/inputs.json',
          ROOT/'results/ai_workloads/swiglu_model_pilot_v4_20260921/manifest.json']


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def live_files():
    paths = set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    paths.add(os.readlink(fd))
                except (FileNotFoundError, PermissionError):
                    pass
            for line in (proc/'maps').read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields)==6:
                    paths.add(fields[-1])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return paths


def dependencies():
    result = {}
    for manifest in ACTIVE:
        result.update(json.loads(manifest.read_text()))
    return result


def validate_dependencies(deps):
    for name, expected in deps.items():
        assert sha(Path(name)) == expected, ('active dependency changed', name)


def plan():
    OUT.mkdir(exist_ok=False)
    deps = dependencies()
    mapped = live_files()
    tracked = {str(ROOT/p) for p in subprocess.check_output(
        ['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if p}
    selected = {}
    # All MICRO study roots are cancelled. Preserve embedded source repositories
    # and installed cross-toolchain even if they contain ELF binaries.
    micro_roots = sorted(p for p in FA.iterdir() if p.is_dir() and p.name.startswith('micro_'))
    for root in micro_roots:
        for git in root.rglob('.git'):
            if git.is_dir():
                repo=git.parent
                tracked.update(str(repo/p) for p in subprocess.check_output(
                    ['git','-C',str(repo),'ls-files','-z']).decode().split('\0') if p)
        for p in root.rglob('*'):
            if not p.is_file() or p.is_symlink():
                continue
            rel=p.relative_to(root)
            if '.git' in rel.parts or 'runs' in rel.parts or any('toolchain' in x for x in rel.parts):
                continue
            with p.open('rb') as f:
                magic=f.read(4)
            if magic==b'\x7fELF':
                selected[p]='Cancelled MICRO generated ELF/object'
    # An earlier opt-in prototype is superseded by build_v2 used by the queue.
    selected[FA/'async_staging_20260921/build/gem5.opt']='Superseded prototype (active queue uses build_v2)'
    # Keep the current incremental SPM build for future work. Only old object
    # intermediates outside it are candidates, never result data or source.
    for root in (ROOT/'gem5/build',ROOT/'results'):
        for p in root.rglob('*'):
            if p.suffix not in ('.o','.os') or p.is_symlink() or not p.is_file():
                continue
            if 'ARM_MESI_Three_Level_SPM' in p.parts or p.stat().st_mtime>time.time()-86400:
                continue
            if not any(x=='build' or x.startswith('build_') or x=='builds' for x in p.relative_to(ROOT).parts):
                continue
            if any('toolchain' in x for x in p.parts):
                continue
            selected.setdefault(p,'Obsolete build object older than 24h outside current SPM build')
    rows=[]
    for p, reason in sorted(selected.items()):
        if not p.exists() or str(p) in deps or str(p) in tracked:
            continue
        assert str(p) not in mapped, ('currently open/mapped',p)
        assert p.resolve()==p and p.is_relative_to(ROOT)
        s=p.stat()
        rows.append(dict(path=str(p),reason=reason,size=s.st_size,allocated=s.st_blocks*512,
                         inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns))
    save(OUT/'plan.json',rows)
    save(OUT/'protected_dependencies.json',deps)
    print(json.dumps(dict(files=len(rows),GiB=sum(r['allocated'] for r in rows)/1024**3,
                          reasons={reason:sum(r['allocated'] for r in rows if r['reason']==reason)/1024**3
                                   for reason in sorted({r['reason'] for r in rows})}),indent=2))


def execute():
    assert not (OUT/'complete.json').exists()
    rows=json.loads((OUT/'plan.json').read_text())
    deps=json.loads((OUT/'protected_dependencies.json').read_text())
    validate_dependencies(deps)
    mapped=live_files()
    for row in rows:
        p=Path(row['path']);s=p.lstat()
        assert p.resolve()==p and p.is_relative_to(ROOT)
        assert stat.S_ISREG(s.st_mode) and str(p) not in deps and str(p) not in mapped
        assert (s.st_ino,s.st_dev,s.st_size,s.st_mtime_ns)==(row['inode'],row['device'],row['size'],row['mtime_ns'])
        row['sha256']=sha(p)
    save(OUT/'deleted_manifest.json',rows)
    # Recheck after hashing, before any deletion, to avoid active-file races.
    mapped=live_files()
    assert not any(r['path'] in mapped for r in rows)
    removed=[]
    for row in rows:
        p=Path(row['path']);s=p.lstat()
        assert (s.st_ino,s.st_dev,s.st_size,s.st_mtime_ns)==(row['inode'],row['device'],row['size'],row['mtime_ns'])
        p.unlink();removed.append(str(p))
    save(OUT/'removed_paths.json',removed)
    validate_dependencies(deps)
    save(OUT/'complete.json',dict(files=len(rows),allocated_bytes=sum(r['allocated'] for r in rows),
         active_dependency_hashes_verified=len(deps),completed=time.time(),
         recovery='Deleted build artifacts are not backed up; rebuild from retained source/toolchain. Exact binaries are not recoverable from this cleanup.'))
    print((OUT/'complete.json').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    if parser.parse_args().execute:execute()
    else:plan()
