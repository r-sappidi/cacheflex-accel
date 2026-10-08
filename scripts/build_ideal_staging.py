"""Freeze an isolated oracle simulator, keeping every previous executable."""
import os
from pathlib import Path
import shutil
os.environ['CACHEFLEX_TIMING_ROOT'] = 'results/cacheflex_flash_attention/ideal_staging_20260923'
os.environ['CACHEFLEX_TIMING_BUILD'] = os.environ.get('CACHEFLEX_IDEAL_BUILD', 'build_v2')
import build_access_timing as build

if __name__ == '__main__':
    build.main()
    # Use the already validated timing configuration, not the older resource
    # adapter copied by the shared build helper.
    source = build.ROOT/'results/cacheflex_flash_attention/timing_alignment_20260922/fa_v3/configs'
    shutil.copytree(source, build.BUILD/'configs', dirs_exist_ok=True)
    shutil.copy2(Path(__file__).with_name('ideal_staging_se.py'),
                 build.BUILD/'configs/deprecated/example/ideal_staging_se.py')
    manifest = {str(p): build.sha(p) for p in (build.BUILD/'sources').rglob('*') if p.is_file()}
    manifest.update({str(p): build.sha(p) for p in (build.BUILD/'configs').rglob('*.py')})
    manifest[str(build.BUILD/'gem5.opt')] = build.sha(build.BUILD/'gem5.opt')
    build.save(build.BUILD/'frozen.json', manifest)
