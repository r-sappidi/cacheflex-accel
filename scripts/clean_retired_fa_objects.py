"""Archive and remove only regenerable objects in explicitly retired FA studies."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/cacheflex_flash_attention'
OUT = ROOT / 'results/build_cleanup_20260921'
NAMES = ('optuna_20260912', 'best_corrected_20260912',
         'vl16_deeper_20260908', 'vl16_remaining_20260908',
         'pv_improvements_20260913', 'aligned_driver_causes_20260911',
         'driver_causes_20260911', 'query_groups_retry_20260911',
         'query_groups_20260911', 'correct_fa_20260911')
ACTIVE = ROOT / 'results/ai_workloads/swiglu_model_pilot_v4_20260921/manifest.json'


def stream_digest(f):
    h = hashlib.sha256()
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
    return h.hexdigest()


def digest(p):
    with p.open('rb') as f:
        return stream_digest(f)


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'manifest.json').exists(), 'Do not overwrite an earlier inventory'
    protected = json.loads(ACTIVE.read_text())
    assert all(digest(Path(p)) == h for p, h in protected.items())
    processes = subprocess.check_output(['ps', '-eo', 'args'], text=True)
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT,
                                         text=True).splitlines())
    candidates = []
    for name in NAMES:
        directory = BASE / name
        assert directory.is_dir() and directory.resolve() == directory
        assert not any(str(directory) in line and ('gem5.opt' in line or
                       'scons' in line or 'python' in line)
                       for line in processes.splitlines())
        for p in sorted(directory.rglob('*')):
            if not p.is_file() or p.is_symlink() or p.suffix not in ('.o', '.os', '.pyo'):
                continue
            assert p.resolve().is_relative_to(directory)
            assert str(p) not in protected and str(p.relative_to(ROOT)) not in tracked
            s = p.stat()
            candidates.append(dict(path=str(p.relative_to(ROOT)), bytes=s.st_size,
                                   inode=s.st_ino, mtime_ns=s.st_mtime_ns, sha256=digest(p)))
    (OUT / 'manifest.json').write_text(json.dumps(candidates, indent=2) + '\n')
    archive = OUT / 'retired_objects.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for row in candidates:
            tar.add(ROOT / row['path'], arcname=row['path'], recursive=False)
    with tarfile.open(archive, 'r:gz') as tar:
        assert tar.getnames() == [r['path'] for r in candidates]
        for row in candidates:
            with tar.extractfile(row['path']) as f:
                assert stream_digest(f) == row['sha256']
    for row in candidates:
        p = ROOT / row['path']
        s = p.stat()
        assert (s.st_ino, s.st_mtime_ns, s.st_size) == (row['inode'], row['mtime_ns'], row['bytes'])
        assert digest(p) == row['sha256']
        p.unlink()
    assert all(digest(Path(p)) == h for p, h in protected.items())
    report = dict(files=len(candidates), removed_bytes=sum(r['bytes'] for r in candidates),
                  archive_bytes=archive.stat().st_size, archive_sha256=digest(archive),
                  active_dependencies_unchanged=True,
                  preserved='All sources, executables, libraries, configs, stats, traces, references; main build and active queues untouched.',
                  recovery=f'tar -xzf {archive} -C {ROOT}')
    (OUT / 'completed.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
