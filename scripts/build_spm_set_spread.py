"""Build an isolated set-placement experiment from the current finite model."""
import json
import shutil
import sys
from pathlib import Path
import build_finite_micro as build
from build_access_timing import sha, save

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/cacheflex_flash_attention/spm_set_spread_20260927'
FILES = ['mem/ruby/structures/' + f for f in
         ('CacheMemory.cc', 'CacheMemory.hh', 'RubyCache.py')]

if __name__ == '__main__':
    if '--preserve' in sys.argv:
        for rel in FILES:
            dst = BASE / 'preserved' / rel
            assert not dst.exists(), dst
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / 'gem5/src' / rel, dst)
        save(BASE / 'preserved/hashes.json',
             {rel: sha(ROOT / 'gem5/src' / rel) for rel in FILES})
    else:
        build.BASE = BASE
        build.BUILD = BASE / 'build'
        build.OLD = ROOT / 'results/cacheflex_flash_attention/finite_model_20260926/build_v3'
        build.main()
