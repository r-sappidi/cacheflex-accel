"""Frozen cache/V-SPM controls on repaired Ruby; no kernel changes or search.

Four T512 numerical/recorder gates precede two concurrent T16384 controls.
Existing results are never overwritten. Failure stops the queue and its peers.
"""
import argparse
import concurrent.futures as cf
import configparser
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

import numpy as np
from fa_export_patch import verify_dump
from single_head_reference import check
from validate_attribution import read

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'results/cacheflex_flash_attention/single_head_20260915'
BUILD = ROOT / 'results/ruby_getx_fix_20260916_build_v3'
CASES = {'cache': (32, 2), 'spm': (16, 4)}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def prior_path(arch, tokens):
    br, group = CASES[arch]
    return OLD / 'runs' / f'{arch}_t{tokens}_br{br}_bc128_g{group}_p1'


def command(original, simulator, run, entries):
    assert not any(x.startswith('--debug-') or x.startswith('--ruby-flight-recorder') for x in original)
    assert sum(x.startswith('--outdir=') for x in original) == 1
    result = [f'--outdir={run}' if x.startswith('--outdir=') else x for x in original]
    result[0] = str(simulator)
    return result + [f'--ruby-flight-recorder-entries={entries}']


def config_audit(prior, current, entries):
    old = configparser.ConfigParser(interpolation=None, strict=False)
    old.read(prior / 'config.ini')
    assert set(old.sections()) == set(current.sections())
    additions = []
    for section in old.sections():
        for key, value in old[section].items():
            assert current[section][key] == value.replace(str(prior), '<RESULT>'), (section, key, value, current[section].get(key))
        for key in current[section].keys() - old[section].keys():
            expected = {'flight_recorder_entries': str(entries),
                        'flight_recorder_messages': 'true', 'spm_retain_llc': 'false'}
            assert key in expected and current[section][key] == expected[key], (section, key, current[section][key])
            additions.append([section, key, current[section][key]])
    controllers = [s for s in current.sections() if 'flight_recorder_entries' in current[s]]
    assert len(controllers) == 32
    assert all(current[s]['flight_recorder_entries'] == str(entries) for s in controllers)
    return {'all_prior_fields_equal': True, 'new_default_fields': additions}


def counter_delta(old, new):
    return {k: {'old': old[k], 'new': new[k]} for k in old.keys() | new.keys()
            if k != 'path' and old.get(k) != new.get(k)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir()  # Exclusive creation: never silently reuse incomplete runs.
    (out / 'runs').mkdir()
    state = dict(pid=os.getpid(), stage='freezing', completed=[], updated=time.time())
    stop = threading.Event()

    def status(stage):
        state.update(stage=stage, updated=time.time())
        save(out / 'status.json', state)
        print(stage, flush=True)

    try:
        status('freezing')
        assert sha(BUILD / 'gem5.opt') == (BUILD / 'simulator.sha256').read_text().strip()
        shutil.copy2(BUILD / 'gem5.opt', out / 'gem5.opt')
        watched = [out / 'gem5.opt', Path(__file__).resolve()]
        watched += list((ROOT / 'gem5/configs').rglob('*.py'))
        watched += [Path(__file__).with_name(n) for n in
                    ('validate_attribution.py', 'analyze_traffic.py', 'fa_export_patch.py', 'single_head_reference.py')]
        for arch in CASES:
            for tokens in (512, 16384):
                prior = prior_path(arch, tokens)
                assert (prior / 'accepted.json').exists(), prior
                cmd = json.loads((prior / 'command.json').read_text())
                watched += [prior / 'command.json', prior / 'config.ini', prior / 'accepted.json',
                            Path(cmd[cmd.index('--cmd') + 1])]
        watched += [OLD / 'references' / f't{t}_bc128.npy' for t in (512, 16384)]
        manifest = {str(p): sha(p) for p in watched}
        save(out / 'manifest.json', manifest)
        save(out / 'plan.json', dict(cases=CASES, tokens=16384, heads=1, d=64, cores=8,
             vl=16, l1i=65536, l1d=65536, private_l2=65536, llc_total=4194304,
             simulator_source=str(BUILD), target_recorder_entries=0,
             stages=['T512 cache/SPM recorder off', 'T512 cache/SPM recorder on; exact observer gates',
                     'T16384 cache/SPM concurrently; full independent output validation'],
             note='Prior pilots, not globally tuned winners. Existing guest paths and canonical argv preserved. No DMA/multicast/batch/LLC retention.'))

        def verify_inputs():
            for path, expected in manifest.items():
                assert sha(path) == expected, ('frozen input changed', path)

        env = dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

        def run(arch, tokens, entries):
            verify_inputs()
            prior = prior_path(arch, tokens)
            path = out / 'runs' / f'{arch}_t{tokens}_rec{entries}'
            path.mkdir()
            original = json.loads((prior / 'command.json').read_text())
            cmd = command(original, out / 'gem5.opt', path, entries)
            save(path / 'command.json', cmd)
            start = time.monotonic()
            proc = None
            try:
                with (path / 'run.log').open('w') as log:
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                    while True:
                        text = (path / 'run.log').read_text(errors='replace')
                        events = re.findall(r'[^\n]*progress event[^\n]*', text)
                        save(path / 'progress.json', dict(stage='running', pid=proc.pid,
                             updated=time.time(), host_elapsed_s=time.monotonic()-start,
                             recent_cpu_events=events[-8:], kernel_percentage=None))
                        if stop.is_set():
                            raise RuntimeError('Peer failed; stopping dependent comparison')
                        try:
                            rc = proc.wait(timeout=10)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                elapsed = time.monotonic() - start
                save(path / 'exit.json', dict(returncode=rc, host_elapsed_s=elapsed))
                assert rc == 0, (path, rc)
                verify_inputs()
                output, counters, cfg = read(path, 16)
                audit = config_audit(prior, cfg, entries)
                assert output['Hq'] == output['Hkv'] == '1' and output['T'] == str(tokens)
                assert (path / 'output.fp16').stat().st_size == tokens * 64 * 2
                br, group = CASES[arch]
                copies = 0 if arch == 'cache' else 8 * math.ceil((tokens // 8 // br) / group) * tokens * 2
                assert counters['m_spmcpRequestsIssued'] == counters['m_spmcpRequestsCompleted'] == copies
                files = list(path.glob('*.flight.bin'))
                assert len(files) == (32 if entries else 0)
                assert all(p.stat().st_size == 64 + entries * 1024 for p in files)
                row = dict(path=str(path), arch=arch, T=tokens, recorder_entries=entries,
                           output=output, counters=counters, host_elapsed_s=elapsed,
                           config_audit=audit, guest_sha256=sha(Path(cmd[cmd.index('--cmd')+1])))
                verify_dump(row)
                ref = OLD / 'references' / f't{tokens}_bc128.npy'
                row['validation'] = check(path / 'output.fp16', np.load(ref))
                row['reference_sha256'] = manifest[str(ref)]
                archived = json.loads((prior / 'accepted.json').read_text())
                row['archived_comparison'] = dict(path=str(prior),
                    same_output=archived['output'] == output,
                    roi_change_percent=100*(counters['roi_us']/archived['counters']['roi_us']-1),
                    counter_changes=counter_delta(archived['counters'], counters))
                save(path / 'accepted.json', row)
                save(path / 'progress.json', dict(stage='accepted', roi_us=counters['roi_us'], host_elapsed_s=elapsed))
                return row
            except BaseException as exc:
                stop.set()
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait()
                save(path / 'progress.json', dict(stage='failed', error=repr(exc)))
                raise

        def pair(tokens, entries):
            rows = {}
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                futures = {pool.submit(run, a, tokens, entries): a for a in CASES}
                for future in cf.as_completed(futures):
                    arch = futures[future]
                    row = future.result()
                    rows[arch] = row
                    state['completed'].append(dict(arch=arch, T=tokens, recorder_entries=entries,
                         roi_us=row['counters']['roi_us'], host_elapsed_s=row['host_elapsed_s']))
                    status(state['stage'])
            return rows

        status('t512_recording_off')
        off = pair(512, 0)
        status('t512_recording_on')
        on = pair(512, 8192)
        for arch in CASES:
            assert off[arch]['output'] == on[arch]['output'], arch
            assert not counter_delta(off[arch]['counters'], on[arch]['counters']), arch
        save(out / 'gates_passed.json', dict(extracted_roi_counters_equal=True, guest_outputs_equal=True,
             full_numerical_checks=True, off=off, on=on))
        status('t16384_recording_off')
        rows = pair(16384, 0)
        speedup = rows['cache']['counters']['roi_us'] / rows['spm']['counters']['roi_us']
        save(out / 'summary.json', dict(rows=rows, cache_over_spm_speedup=speedup))
        lines = ['# Clean repaired-simulator FA controls', '',
                 'Single head, T16384, d64, 8 cores, VL16; transaction recorder disabled.', '',
                 '| Kernel | ROI (us) | Host hours | ROI change vs archived |',
                 '|---|---:|---:|---:|']
        for arch, row in rows.items():
            lines.append(f"| {arch} | {row['counters']['roi_us']:.4f} | {row['host_elapsed_s']/3600:.3f} | {row['archived_comparison']['roi_change_percent']:+.4f}% |")
        lines += ['', f'Cache/SPM speedup: {speedup:.6f}x.', '',
                  'All outputs independently checked; guest/export hashes, copy counts and prior configuration fields audited.',
                  'These are frozen pilot controls, not a completed best-to-best tuning search.']
        (out / 'REPORT.md').write_text('\n'.join(lines)+'\n')
        status('complete')
    except BaseException as exc:
        stop.set()
        state['error'] = repr(exc)
        status('failed')
        raise


if __name__ == '__main__':
    main()
