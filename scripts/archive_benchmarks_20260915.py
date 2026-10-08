"""Recoverable, explicitly scoped benchmark cleanup; --restore reverses it."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'archive/benchmarks_cleanup_20260915'
TARGETS = [
    'benchmarks/acl_full_stack_gemm/w7_fp16_acl',
    'benchmarks/acl_sve_gemm/bin_gem5',
    'benchmarks/ai_workloads/bin',
    'benchmarks/ai_workloads/__pycache__',
    'benchmarks/cacheflex_flash_attention/bin',
    'benchmarks/cacheflex_flash_attention/__pycache__',
    'benchmarks/cacheflex_spm_gemm/bin_gem5',
    'benchmarks/ai_workloads/after_spm_probe.py',
    'benchmarks/cacheflex_flash_attention/queue_residency_validation.py',
    'benchmarks/cacheflex_flash_attention/run_dma_solution_sweep.sh',
    'benchmarks/cacheflex_flash_attention/run_software_best_8core.sh',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    restore = sys.argv[1:] == ['--restore']
    if sys.argv[1:] and not restore:
        raise SystemExit('Usage: archive_benchmarks_20260915.py [--restore]')
    if restore:
        manifest = json.loads((ARCHIVE / 'manifest.json').read_text())
        for entry in manifest['files']:
            source = ARCHIVE / entry['path']
            target = ROOT / entry['path']
            if target.exists() or target.is_symlink():
                raise SystemExit(f'Refusing to overwrite {target}')
            if digest(source) != entry['sha256']:
                raise SystemExit(f'Archive hash mismatch: {source}')
        for entry in manifest['files']:
            target = ROOT / entry['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ARCHIVE / entry['path']), str(target))
        print(f"Restored {len(manifest['files'])} files")
        return
    if ARCHIVE.exists():
        raise SystemExit(f'Archive already exists: {ARCHIVE}')
    files = []
    for relative in TARGETS:
        path = ROOT / relative
        if not path.exists():
            raise SystemExit(f'Missing audited target: {path}')
        for item in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if item.is_symlink():
                raise SystemExit(f'Unexpected symlink: {item}')
            if item.is_file():
                files.append(dict(path=str(item.relative_to(ROOT)),
                                  bytes=item.stat().st_size, sha256=digest(item)))
    # Check live command lines before moving anything; running queues themselves
    # were audited separately for future build dependencies.
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = proc.read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (OSError, PermissionError):
            continue
        if any(relative in command for relative in TARGETS):
            raise SystemExit(f'Target referenced by live process: {proc}: {command}')
    ARCHIVE.mkdir(parents=True)
    manifest = dict(targets=TARGETS, files=files,
                    total_bytes=sum(entry['bytes'] for entry in files))
    (ARCHIVE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for relative in TARGETS:
        target = ARCHIVE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(ROOT / relative), str(target))
    for entry in files:
        assert digest(ARCHIVE / entry['path']) == entry['sha256'], entry['path']
    print(json.dumps(dict(files=len(files), bytes=manifest['total_bytes'],
                          archive=str(ARCHIVE))))


if __name__ == '__main__':
    main()
