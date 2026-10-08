"""Remove regenerable intermediate objects from completed Classic build only."""
import json
from pathlib import Path
import subprocess
from run_micro_fa_ruby_pilot import ROOT,sha,save

BASE=ROOT/'results/cacheflex_flash_attention'
TREE=BASE/'micro_upstream_20260918/gem5/build/ARM'
OUT=BASE/'build_cleanup_20260918'
OUT.mkdir(exist_ok=True)
assert TREE.is_dir() and (TREE/'gem5.opt').is_file()
processes=subprocess.check_output(['ps','-eo','args'],text=True)
# Do not clean while the dedicated MICRO builder is running. Other repository
# builds are separate and must be left untouched.
for line in processes.splitlines():
    if 'scons' in line and 'build/ARM/gem5.opt' in line:
        raise RuntimeError('Classic build may be active; refusing cleanup')
files=[p for p in TREE.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in ('.o','.pyo','.os')]
tracked=set(subprocess.check_output(['git','-C',str(BASE/'micro_upstream_20260918'),'ls-files'],text=True).splitlines())
assert not any(str(p.relative_to(BASE/'micro_upstream_20260918')) in tracked for p in files)
manifest=[dict(path=str(p),bytes=p.stat().st_size) for p in files]
before=sha(TREE/'gem5.opt')
save(OUT/'manifest.json',dict(scope=str(TREE),files=manifest,bytes=sum(x['bytes'] for x in manifest),simulator_sha256=before,
    recovery='Regenerate via SCons with saved Classic build environment. No backup of regenerable objects; source, build metadata, simulator, guests, configs, results, and logs preserved.'))
for p in files:
    assert p.is_relative_to(TREE) and p.suffix in ('.o','.pyo','.os')
    p.unlink()
assert sha(TREE/'gem5.opt')==before
save(OUT/'completed.json',dict(files=len(files),bytes=sum(x['bytes'] for x in manifest),simulator_sha256_unchanged=True))
print((OUT/'completed.json').read_text())
