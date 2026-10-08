"""Capture pre-edit inputs without touching any active frozen experiment."""
import json, shutil, time
from pathlib import Path
from build_access_timing import sha, save
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_l3_spm/initial_20260928'
def main():
    out=BASE/'preserved';out.mkdir(parents=True)
    files=[ROOT/'gem5/src/mem/request.hh',ROOT/'gem5/src/cpu/o3/lsq_unit.cc',
           ROOT/'gem5/src/mem/ruby/network/Network.cc']
    for sub in ('protocol','structures','system'):
        files += [p for p in (ROOT/'gem5/src/mem/ruby'/sub).iterdir() if p.is_file()]
    manifest={}
    for p in files:
        dst=out/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dst);manifest[str(p)]=sha(p)
    save(out/'manifest.json',dict(created=time.time(),files=manifest))
    print(out)
if __name__=='__main__':main()
