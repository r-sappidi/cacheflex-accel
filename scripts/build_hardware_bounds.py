"""Preserve sources and build an isolated finite-resource model."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/cacheflex_flash_attention/hardware_bounds_20260921'
ADAPTER = ROOT / 'scripts/bounded_se.py'
FILES = ['cpu/o3/BaseO3CPU.py', 'cpu/o3/lsq_unit.hh', 'cpu/o3/lsq_unit.cc',
         'mem/ruby/system/Sequencer.py', 'mem/ruby/system/Sequencer.hh',
         'mem/ruby/system/Sequencer.cc',
         'mem/ruby/protocol/MESI_Three_Level_SPM-dir.sm',
         'mem/ruby/protocol/MESI_Three_Level_SPM-L1cache.sm',
         'mem/ruby/protocol/MESI_Three_Level_SPM-L0cache.sm']

def main():
    OUT.mkdir(exist_ok=True)
    if '--preserve' in sys.argv:
        dest = OUT / 'preserved'; dest.mkdir()
        for rel in FILES:
            target = dest / rel; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / 'gem5/src' / rel, target)
        (dest / 'git.diff').write_bytes(subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT))
        return
    build_name = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--build-name=')), 'build')
    assert '/' not in build_name and build_name.startswith('build')
    build = OUT / build_name; build.mkdir(exist_ok='--resume' in sys.argv)
    def save(value):
        (build / 'status.json').write_text(json.dumps(dict(value, updated=time.time(), pid=os.getpid()), indent=2))
    env = dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
        PYTHON_CONFIG='/data2/rsappidi/conda-envs/cacheflex-gem5/bin/python3-config',
        PROTOC='/usr/bin/protoc', CCFLAGS_EXTRA='-gdwarf-2', GEM5PY_CCFLAGS_EXTRA='-gdwarf-2',
        LINKFLAGS_EXTRA='-L/usr/lib64', CACHEFLEX_LINK_MANIFEST=str(build / 'link_manifest.json'))
    save(dict(stage='building'))
    try:
        with (build / 'build.log').open('w') as log:
            subprocess.run(['/data2/rsappidi/conda-envs/cacheflex-gem5/bin/scons',
                'build/ARM_MESI_Three_Level_SPM/gem5.opt', '-j8', '--linker=gold'],
                cwd=ROOT / 'gem5', env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        # A second dependency pass also catches edits made while the first
        # incremental compilation was running. Freeze sources after it.
        with (build / 'build_final.log').open('w') as log:
            subprocess.run(['/data2/rsappidi/conda-envs/cacheflex-gem5/bin/scons',
                'build/ARM_MESI_Three_Level_SPM/gem5.opt', '-j8', '--linker=gold'],
                cwd=ROOT / 'gem5', env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest = json.loads((build / 'link_manifest.json').read_text())
        cmd = manifest['argv']; cmd[cmd.index('-o') + 1] = str(build / 'gem5.opt')
        with (build / 'link.log').open('w') as log:
            subprocess.run(cmd, cwd=ROOT / 'gem5/build/ARM_MESI_Three_Level_SPM',
                env=dict(manifest['environment'], LD_LIBRARY_PATH=env['LD_LIBRARY_PATH']),
                stdout=log, stderr=subprocess.STDOUT, check=True)
        shutil.copytree(ROOT / 'gem5/configs', build / 'configs', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(ADAPTER, build / 'configs/deprecated/example/bounded_se.py')
        digest = hashlib.sha256((build / 'gem5.opt').read_bytes()).hexdigest()
        (build / 'simulator.sha256').write_text(digest + '\n')
        sources = {}
        for rel in FILES:
            source = ROOT / 'gem5/src' / rel
            target = build / 'sources' / rel; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            sources[rel] = hashlib.sha256(source.read_bytes()).hexdigest()
        (build / 'validated_build_ready.json').write_text(json.dumps(dict(
            sha256=digest, source_hashes=sources,
            note='Build ready for validation, NOT correctness accepted'), indent=2))
        save(dict(stage='built_requires_validation', sha256=digest))
    except BaseException as exc:
        save(dict(stage='failed', error=repr(exc))); raise

if __name__ == '__main__':
    main()
