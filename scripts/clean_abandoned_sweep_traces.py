"""Delete explicitly retired raw traces; retain results and diagnostic excerpts."""
import os
from pathlib import Path
import json
from run_fa_staging_trace import ROOT,save
B=ROOT/'results/cacheflex_flash_attention'
O=B/'retired_trace_cleanup_20260918';O.mkdir()
targets=[B/'scaled_diagnosis_20260918/experiment/runs'/f'{a}_t256_fetch17/run.log' for a in ('cache','spm')]
targets += sorted((B/'spm_capacity_probe_20260909').glob('*_trace1/debug.trace'))
assert len(targets)>=2
rows=[]
# Refuse removal if any process still has one of these files open.
inodes={(p.stat().st_dev,p.stat().st_ino) for p in targets}
for proc in Path('/proc').glob('[0-9]*'):
    try:
        for fd in (proc/'fd').iterdir():
            try:
                st=fd.stat()
                if (st.st_dev,st.st_ino) in inodes:raise RuntimeError(f'trace still open: {fd}')
            except (FileNotFoundError,PermissionError):pass
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
for i,p in enumerate(targets):
    assert p.is_relative_to(B) and not p.is_symlink() and p.name in ('run.log','debug.trace')
    st=p.stat();excerpt=O/f'{i:02d}.excerpt'
    with p.open('rb') as f,excerpt.open('wb') as out:
        out.write(f.read(65536));out.write(b'\n--- MIDDLE OMITTED ---\n')
        f.seek(max(0,st.st_size-65536));out.write(f.read())
    rows.append(dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,excerpt=str(excerpt)))
save(O/'manifest.json',dict(files=rows,recovery='Raw traces are not backed up; rerun saved commands to regenerate. Excerpts, stats, derived reports, configs, binaries preserved.'))
for p,row in zip(targets,rows):
    st=p.stat();assert (st.st_ino,st.st_size,st.st_mtime_ns)==(row['inode'],row['bytes'],row['mtime_ns'])
    p.unlink()
save(O/'completed.json',dict(files=len(rows),bytes_removed=sum(r['bytes'] for r in rows)))
print((O/'completed.json').read_text())
