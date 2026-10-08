"""Replay the failed four-core case unchanged, with a bounded diagnostic window.

This is not a benchmark result or a repair. Never overwrite the failed run.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/cacheflex_flash_attention/micro_head_scaling_20260921'
PRIOR = BASE / 'runs/cache_parallel_c4_t4096'
OUT = BASE / 'diagnosis/functional_read_v2'
START = 116_850_000_000
STOP = 116_950_000_000


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    state = dict(coordinator=os.getpid(), stage='verifying_inputs')

    def status(stage, **kw):
        state.update(stage=stage, updated=time.time(), **kw)
        save(OUT / 'status.json', state)

    proc = None
    try:
        status('verifying_inputs')
        manifest = json.loads((BASE / 'inputs.json').read_text())

        def frozen():
            for path, digest in manifest.items():
                assert sha(Path(path)) == digest, ('Changed frozen input', path)

        frozen()
        save(OUT / 'inputs.json', manifest)
        shutil.copy2(__file__, OUT / Path(__file__).name)
        original = json.loads((PRIOR / 'command.json').read_text())
        cmd = [f'--outdir={OUT}' if x.startswith('--outdir=') else x
               for x in original]
        assert not any(x.startswith(('--debug-', '--abs-max-tick')) for x in cmd)
        cmd[1:1] = [
            '--debug-flags=ProtocolTrace,RubySystem,RubyPort,SyscallBase,SyscallVerbose',
            f'--debug-start={START}', f'--debug-end={STOP}',
            '--debug-file=diagnostic.trace']
        cmd += [f'--abs-max-tick={STOP}']
        save(OUT / 'command.json', cmd)
        save(OUT / 'plan.json', dict(
            original=str(PRIOR), target_address='0x1b16840',
            debug_start=START, stop_tick=STOP, max_trace_bytes=2 * 1024**3,
            max_host_seconds=4 * 3600, numerical_acceptance=False,
            changes=['output directory', 'debug flags/window', 'stop tick'],
            hardware_and_guest_unchanged=True))
        begin = time.monotonic()
        env = dict(os.environ,
                   LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        with (OUT / 'run.log').open('w') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
            while True:
                trace = OUT / 'diagnostic.trace'
                size = trace.stat().st_size if trace.exists() else 0
                elapsed = time.monotonic() - begin
                assert size <= 2 * 1024**3, 'Trace exceeds 2GiB budget'
                assert elapsed < 4 * 3600, 'Replay timeout'
                assert shutil.disk_usage(OUT).free > 50 * 1024**3
                status('replaying', simulator=proc.pid,
                       host_seconds=elapsed, trace_bytes=size)
                try:
                    rc = proc.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    pass
        frozen()
        log = (OUT / 'run.log').read_text(errors='backslashreplace')
        reproduced = 'Ruby functional read failed for address 0x1b16840' in log
        save(OUT / 'exit.json', dict(returncode=rc, reproduced=reproduced))
        status('failure_reproduced' if reproduced else 'inspect_replay',
               returncode=rc, host_seconds=time.monotonic() - begin)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        status('diagnostic_failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
