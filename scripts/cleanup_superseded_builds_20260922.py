"""Archive verified obsolete simulator binaries, then remove only those files.

Default: reviewable plan. --execute: hash-verified recoverable cleanup.
No source, result, trace, current build, guest or toolchain deletion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

from cleanup_builds_20260921 import ROOT, FA, live_files, sha, save

OUT = ROOT / 'results/cleanup_superseded_builds_20260922'
GROUPS = {
    'access_timing_20260922': ('build', 'build_v2'),
    'access_aggregate_20260922': ('build', 'build_v2'),
    'hardware_bounds_20260921': ('build', 'build_v2', 'build_v3', 'build_v4'),
    'micro_resources_20260921': ('build', 'build_v3', 'build_v4'),
}
ACTIVE = [FA/'ingress_bypass_20260922/experiment/inputs.json',
          ROOT/'results/ai_workloads/swiglu_model_pilot_v4_20260921/manifest.json']


def dependencies():
    deps = {}
    for p in ACTIVE:
        deps.update(json.loads(p.read_text()))
    return deps


def validate(deps):
    for name, digest in deps.items():
        assert sha(Path(name)) == digest, ('active dependency changed', name)


def identity(p):
    s = p.lstat()
    assert p.is_file() and not p.is_symlink() and p.resolve() == p
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def plan():
    OUT.mkdir(exist_ok=False)
    deps = dependencies()
    mapped = live_files()
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT)
                  .decode().split('\0'))
    rows = []
    for study, builds in GROUPS.items():
        for build in builds:
            p = FA/study/build/'gem5.opt'
            assert p.exists(), p
            assert str(p) not in deps and str(p) not in mapped
            assert str(p.relative_to(ROOT)) not in tracked
            with p.open('rb') as f:
                assert f.read(4) == b'\x7fELF'
            rows.append(dict(path=str(p), member=str(p.relative_to(ROOT)),
                             identity=identity(p), allocated=p.stat().st_blocks*512,
                             reason='Superseded diagnostic/prototype simulator; latest validated build retained'))
    save(OUT/'plan.json', rows)
    save(OUT/'protected_dependencies.json', deps)
    print(json.dumps(dict(files=len(rows), allocated_GiB=sum(r['allocated'] for r in rows)/2**30), indent=2))


def execute():
    assert not (OUT/'complete.json').exists()
    rows = json.loads((OUT/'plan.json').read_text())
    deps = json.loads((OUT/'protected_dependencies.json').read_text())
    assert deps == dependencies()
    validate(deps)
    mapped = live_files()
    for row in rows:
        p = Path(row['path'])
        assert identity(p) == row['identity'] and str(p) not in mapped
        row['sha256'] = sha(p)
    save(OUT/'manifest.json', rows)
    archive = OUT/'superseded_simulators.tar.zst'
    assert not archive.exists()
    subprocess.run(['tar', '-I', 'zstd -1 -T1', '-cf', str(archive),
                    '-C', str(ROOT), '--'] + [r['member'] for r in rows], check=True)
    expected = {r['member']:r['sha256'] for r in rows}
    seen = set()
    proc = subprocess.Popen(['zstd', '-dc', str(archive)], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode='r|') as tf:
        for member in tf:
            assert member.isfile() and member.name in expected
            h = hashlib.sha256()
            stream = tf.extractfile(member)
            for chunk in iter(lambda: stream.read(4*1024*1024), b''):
                h.update(chunk)
            assert h.hexdigest() == expected[member.name], member.name
            seen.add(member.name)
    # Drain the tar end padding before waiting for the decompressor.
    while proc.stdout.read(1024*1024):
        pass
    assert proc.wait() == 0 and seen == set(expected)
    validate(deps)
    mapped = live_files()
    for row in rows:
        p = Path(row['path'])
        assert identity(p) == row['identity'] and str(p) not in mapped
    for row in rows:
        Path(row['path']).unlink()
    validate(deps)
    archive_bytes = archive.stat().st_blocks*512
    save(OUT/'complete.json', dict(files=len(rows),
         original_allocated_bytes=sum(r['allocated'] for r in rows),
         archive_allocated_bytes=archive_bytes,
         net_reclaimed_bytes=sum(r['allocated'] for r in rows)-archive_bytes,
         archive_sha256=sha(archive), active_dependency_hashes_verified=len(deps),
         completed=time.time(), recovery=f'tar -I zstd -xf {archive} -C {ROOT}',
         preserved='All source, configs, logs, traces, numerical results, guests, toolchains, latest validated and active builds'))
    print((OUT/'complete.json').read_text())


def purge_archive():
    """Reclaim the temporary archive too; all current binaries remain intact."""
    result = json.loads((OUT/'complete.json').read_text())
    archive = OUT/'superseded_simulators.tar.zst'
    assert archive.resolve() == archive and not archive.is_symlink()
    assert sha(archive) == result['archive_sha256']
    assert str(archive) not in live_files()
    assert not any(Path(r['path']).exists()
                   for r in json.loads((OUT/'manifest.json').read_text()))
    for study, latest in [('access_timing_20260922', 'build_v3'),
                          ('access_aggregate_20260922', 'build_v3'),
                          ('hardware_bounds_20260921', 'build_v5'),
                          ('micro_resources_20260921', 'build_v5')]:
        assert (FA/study/latest/'gem5.opt').is_file()
    deps = dependencies()
    validate(deps)
    archive.unlink()
    result.update(archive_removed=True, archive_allocated_bytes=0,
                  net_reclaimed_bytes=result['original_allocated_bytes'],
                  recovery='Obsolete binaries deleted without backup. Source snapshots, hashes and build manifests retained for rebuilding; exact binary reproduction not guaranteed.')
    save(OUT/'complete.json', result)
    validate(deps)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--purge-archive', action='store_true')
    args = ap.parse_args()
    if args.purge_archive:
        purge_archive()
    elif args.execute:
        execute()
    else:
        plan()
