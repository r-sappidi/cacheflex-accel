"""Restart only the three jobs interrupted by the lost tmux session.

Reuse the original runner validation code and frozen input manifests. Preserve
partial runs by rename; completed cases are skipped. No model/input edits.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/data2/rsappidi/cacheflex_mc')
JOURNAL = ROOT / 'results/interrupted_restart_20260928'
SUITES = {
    'swiglu': (
        ROOT / 'scripts/run_swiglu_retry_transfer.py',
        ROOT / 'results/ai_workloads/swiglu_retry_transfer_20260927',
        {'pilot_cache_t512_d2048_f8192_mp2_kc128_g0',
         'pilot_spm_t512_d2048_f8192_mp2_kc128_g0'},
    ),
    'llama': (
        ROOT / 'benchmarks/cacheflex_flash_attention/llama_cpu/run.py',
        ROOT / 'results/cacheflex_flash_attention/llama_cpu_20260925/experiment_export_safe_20260926',
        {'upstream_sve_ready_t16384'},
    ),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def once(source, old, new):
    assert source.count(old) == 1, ('runner structure changed', old[:100])
    return source.replace(old, new, 1)


def adapted(kind, runner):
    source = runner.read_text()
    if kind == 'swiglu':
        source = once(source, 'ROOT=Path(__file__).resolve().parents[1]', f'ROOT=Path({str(ROOT)!r})')
        source = once(source,
            "    OUT.mkdir();(OUT/'runs').mkdir()\n    state=dict(pid=os.getpid(),stage='preparing',active={},completed=[])",
            "    state=json.loads((OUT/'status.json').read_text())\n    state.update(pid=os.getpid(),stage='restarting_interrupted',active={})\n    state['recovery_journal']=str(_recovery)")
        start = source.index('    frozen={str(p):sha(p)')
        end = source.index('    def unchanged():', start)
        source = source[:start] + "    frozen=json.loads((OUT/'inputs.json').read_text())\n" + source[end:]
        source = once(source,
            "        source=Path(old['path']);name=source.name;path=OUT/'runs'/name;path.mkdir()",
            "        source=Path(old['path']);name=source.name;path=OUT/'runs'/name\n"
            "        if any(Path(r['path']).name==name for r in state['completed']):return\n"
            "        assert name in _targets\n        path.mkdir()")
        source = once(source,
            "            save(path/'command.json',cmd);save(path/'case.json',dict(source=str(source),case=old['case']))",
            "            assert cmd==json.loads((_recovery/'partials'/name/'command.json').read_text()),'command changed'\n"
            "            save(path/'command.json',cmd);save(path/'case.json',dict(source=str(source),case=old['case']))")
        source = once(source, "[('gates',24),('t512_transfer',512)]", "[('t512_transfer',512)]")
    else:
        source = once(source, 'HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]',
                      f'HERE=Path({str(runner.parent)!r});ROOT=HERE.parents[2]')
        start = source.index("    OUT.mkdir();(OUT/'runs').mkdir();(OUT/'references').mkdir()")
        end = source.index('    def unchanged():', start)
        source = source[:start] + """    state=json.loads((OUT/'status.json').read_text())
    state.update(pid=os.getpid(),stage='restarting_interrupted',active={})
    state['recovery_journal']=str(_recovery)
    lock=threading.RLock();halt=threading.Event()
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state)
    frozen=json.loads((OUT/'inputs.json').read_text())
    cmd0=json.loads((CONTROL/'command.json').read_text())
""" + source[end:]
        source = once(source,
            "        name=f'{mode}_t{t}';path=OUT/'runs'/name;path.mkdir();proc=None;start=time.monotonic()",
            "        name=f'{mode}_t{t}';path=OUT/'runs'/name\n"
            "        assert name in _targets\n        path.mkdir();proc=None;start=time.monotonic()")
        source = once(source, "            save(path/'command.json',cmd)",
            "            assert cmd==json.loads((_recovery/'partials'/name/'command.json').read_text()),'command changed'\n"
            "            save(path/'command.json',cmd)")
        source = once(source,
            "            status(f'reference_t{t}');refs[t]=dense(t);np.save(OUT/'references'/f't{t}.npy',refs[t])",
            "            refs[t]=np.load(OUT/'references'/f't{t}.npy')")
        source = once(source,
            "                assert not halt.is_set()\n                run(mode,t,refs[t])",
            "                if any(r['mode']==mode and r['t']==t for r in state['completed']):continue\n"
            "                assert not halt.is_set()\n                run(mode,t,refs[t])")
    compile(source, str(runner), 'exec')
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', choices=SUITES)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    runner, out, targets = SUITES[args.suite]
    source = adapted(args.suite, runner)
    manifest = json.loads((out/'inputs.json').read_text())
    for filename, expected in manifest.items():
        assert sha(filename) == expected, ('frozen input mismatch', filename)
    state = json.loads((out/'status.json').read_text())
    assert set(state['active']) == targets, ('unexpected active jobs', state['active'])
    assert not state.get('error'), state.get('error')
    for pid in [state['pid']] + [r['pid'] for r in state['active'].values()]:
        assert not Path(f'/proc/{pid}').exists(), ('PID exists; inspect before restart', pid)
    for name in targets:
        path = out/'runs'/name
        assert not (path/'accepted.json').exists() and not (path/'validated.json').exists()
        assert not (path/'failed.json').exists()
    assert shutil.disk_usage(out).free > 50 * 2**30
    print(f'{args.suite}: {len(manifest)} frozen hashes verified; {len(targets)} interrupted jobs; adapter compiles', flush=True)
    if args.check:
        return
    journal = JOURNAL/args.suite
    journal.mkdir(parents=True, exist_ok=True)
    with (journal/'coordinator.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (journal/'launch.json').exists(), 'recovery already launched'
        (journal/'partials').mkdir()
        shutil.copy2(out/'status.json', journal/'status_before.json')
        shutil.copy2(runner, journal/'original_runner.py')
        shutil.copy2(__file__, journal/'recovery_runner.py')
        (journal/'adapted_runner.py').write_text(source)
        preserved = {}
        for name in sorted(targets):
            path = out/'runs'/name
            preserved[name] = {p.name:sha(p) for p in path.iterdir() if p.is_file()}
            path.rename(journal/'partials'/name)
        save(journal/'partial_hashes.json', preserved)
        save(journal/'launch.json', dict(pid=os.getpid(),started=time.time(),suite=args.suite,
            targets=sorted(targets),manifest_entries=len(manifest),original_runner_sha256=sha(runner),
            reason='User requested recovery after tmux session was killed',
            ledger=['AI-06','AI-09','FA-30'],restart_from_beginning=True))
        sys.path[:0] = [str(runner.parent), str(ROOT/'scripts')]
        namespace = dict(__name__='__main__',__file__=str(runner),_recovery=journal,_targets=targets)
        exec(compile(source, str(runner), 'exec'), namespace)


if __name__ == '__main__':
    main()
