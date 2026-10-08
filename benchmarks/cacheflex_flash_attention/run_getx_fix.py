"""Validate the release/GETX repair, then retry the frozen failed T16K guest."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from validate_attribution import read
from fa_export_patch import verify_dump
from single_head_reference import check

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/cacheflex_flash_attention'
OLD = BASE/'q_resident_20260915_v4'
BUILD = ROOT/'results/ruby_getx_fix_20260916_build_v3'
OUT = BASE/'q_getx_fix_20260916_v3'


def main():
    OUT.mkdir()
    state = dict(pid=os.getpid(), completed=[])
    def status(stage, **kwargs):
        state.update(stage=stage, updated=time.time(), **kwargs)
        p = OUT/'status.tmp'
        p.write_text(json.dumps(state, indent=2)); p.replace(OUT/'status.json')
    try:
        status('waiting_for_build')
        while not (BUILD/'simulator.sha256').exists():
            time.sleep(10)
        sim = BUILD/'gem5.opt'
        assert hashlib.sha256(sim.read_bytes()).hexdigest() == (BUILD/'simulator.sha256').read_text().strip()
        status('release_getx_stress_gate')
        subprocess.run([str(ROOT/'.venv-fa-optuna/bin/python'), '-B',
            str(ROOT/'scripts/run_spm_release_getx.py'), str(OUT/'stress'),
            '--simulator', str(sim)], check=True)
        # Validate recorder on/off too, using the unchanged T512 guest.
        status('t512_observer_and_numerical_gate')
        subprocess.run([str(ROOT/'.venv-fa-optuna/bin/python'), '-B',
            str(ROOT/'scripts/validate_ruby_flight.py'), str(sim), str(OUT/'t512')], check=True)
        for t in (2048, 16384):
            prior = OLD/'runs'/f'spm_t{t}_br16_bc128_g4_p1'
            run = OUT/f't{t}'
            run.mkdir()
            cmd = json.loads((prior/'command.json').read_text())
            cmd = [f'--outdir={run}' if x.startswith('--outdir=') else x for x in cmd]
            cmd[0] = str(sim)
            cmd += ['--ruby-flight-recorder-entries=8192']
            # Bounded records throughout; detailed transition evidence only
            # around the known late race, avoiding a full-run text trace.
            if t == 16384:
                cmd[1:1] = ['--debug-flags=ProtocolTrace', '--debug-start=73534000000',
                            '--debug-end=73560000000', '--debug-file=race.log']
            (run/'command.json').write_text(json.dumps(cmd, indent=2))
            guest = Path(cmd[cmd.index('--cmd')+1])
            (run/'guest.sha256').write_text(hashlib.sha256(guest.read_bytes()).hexdigest()+'\n')
            start = time.time()
            with (run/'run.log').open('w') as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
                while proc.poll() is None:
                    status(f't{t}_running', child_pid=proc.pid, host_elapsed_s=time.time()-start)
                    time.sleep(15)
            assert proc.returncode == 0, (t, proc.returncode)
            output, counters, cfg = read(run, 16)
            _, _, previous_cfg = read(prior, 16) if t == 2048 else (None, None, None)
            # Full config comparison also works against a failed prior run.
            if previous_cfg is None:
                import configparser
                previous_cfg = configparser.ConfigParser(interpolation=None, strict=False)
                previous_cfg.read(prior/'config.ini')
                for section in previous_cfg.sections():
                    for key, value in list(previous_cfg[section].items()):
                        previous_cfg[section][key] = value.replace(str(prior), '<RESULT>')
            for section in previous_cfg.sections():
                for key, value in previous_cfg[section].items():
                    assert cfg[section][key] == value, (section, key)
            assert counters['m_spmcpRequestsIssued'] == counters['m_spmcpRequestsCompleted'] == t*2
            row = dict(path=str(run), output=output, counters=counters)
            verify_dump(row)
            ref_path = (OLD if t == 2048 else BASE/'single_head_20260915')/'references'/f't{t}_bc128.npy'
            row['reference_sha256'] = hashlib.sha256(ref_path.read_bytes()).hexdigest()
            row['validation'] = check(run/'output.fp16', np.load(ref_path))
            (run/'accepted.json').write_text(json.dumps(row, indent=2))
            state['completed'].append(dict(T=t, roi_us=counters['roi_us']))
        status('complete', child_pid=None)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
