#!/usr/bin/env python3
"""Small frozen Q-resident FA off/on observer gate; never touches active replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from decode_ruby_flight import decode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks/cacheflex_flash_attention'))
from validate_attribution import read
from fa_export_patch import verify_dump
from single_head_reference import reference, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('simulator', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir()
    original = ROOT/'results/cacheflex_flash_attention/q_resident_20260915_v4/runs/spm_t512_br16_bc128_g4_p0'
    command = json.loads((original/'command.json').read_text())
    rows, configs, elapsed = {}, {}, {}
    def status(**data):
        temp = out/'status.tmp'
        temp.write_text(json.dumps(data, indent=2))
        temp.replace(out/'status.json')
    try:
        for name, entries in [('off', 0), ('on', 8192)]:
            run = out/name
            run.mkdir()
            cmd = [f'--outdir={run}' if x.startswith('--outdir=') else x for x in command]
            cmd[0] = str(args.simulator.resolve())
            cmd.append(f'--ruby-flight-recorder-entries={entries}')
            (run/'command.json').write_text(json.dumps(cmd, indent=2))
            start = time.time()
            with (run/'run.log').open('w') as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
                status(stage=name, pid=proc.pid)
                rc = proc.wait()
            assert rc == 0, (name, rc)
            elapsed[name] = time.time() - start
            output, counters, configs[name] = read(run, 16)
            row = dict(path=str(run), output=output, counters=counters)
            verify_dump(row)
            row['validation'] = check(run/'output.fp16', reference(512, 128))
            rows[name] = row
        assert rows['off']['output'] == rows['on']['output']
        for key, value in rows['off']['counters'].items():
            if key != 'path':
                assert value == rows['on']['counters'][key], key
        for section in configs['off'].sections():
            for key, value in configs['off'][section].items():
                if key != 'flight_recorder_entries':
                    assert value == configs['on'][section][key], (section, key)
        assert not list((out/'off').glob('*.flight.bin'))
        files = list((out/'on').glob('*.flight.bin'))
        controllers = [s for s in configs['on'].sections()
                       if configs['on'][s].get('flight_recorder_entries') == '8192']
        assert len(files) == len(controllers) == 32, (len(files), len(controllers))
        summary = {}
        for path in files:
            records, rejected = decode(path)
            assert records and not rejected, (path, rejected)
            assert path.stat().st_size == 64 + 8192*1024
            summary[path.name] = dict(records=len(records), latest_sequence=max(r[2] for r in records))
        (out/'passed.json').write_text(json.dumps(dict(rows=rows, files=summary,
            host_elapsed_seconds=elapsed, note='Off/on matched rebuilt simulator, not protocol-correctness proof'), indent=2))
        status(stage='passed')
    except BaseException as exc:
        status(stage='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
