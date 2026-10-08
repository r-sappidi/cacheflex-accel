"""Two-thread release/store stress on old and repaired simulators."""
import json
import os
from pathlib import Path
import subprocess
import argparse

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/cacheflex_flash_attention/q_resident_20260915_v4/runs/spm_t512_br16_bc128_g4_p0'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--simulator', type=Path, default=ROOT/'results/ruby_getx_fix_20260916_build_v2/gem5.opt')
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir()
    command = json.loads((BASE/'command.json').read_text())
    for name, sim in [('old', ROOT/'results/ruby_flight_20260916/build_v2/gem5.opt'),
                      ('fixed', args.simulator.resolve())]:
        run = OUT/name
        run.mkdir()
        cmd = [f'--outdir={run}' if x.startswith('--outdir=') else x for x in command]
        cmd = [x.replace('sve_vl_se=16', 'sve_vl_se=4') for x in cmd]
        cmd = ["system.cpu[0].workload[0].cmd=['spm-release-getx']"
               if x.startswith('system.cpu[0].workload[0].cmd=') else x for x in cmd]
        cmd[cmd.index('--cmd')+1] = str(ROOT/'results/ruby_getx_race_guest_20260916/guest')
        cmd[0] = str(sim)
        cmd[1:1] = ['--debug-flags=ProtocolTrace', '--debug-file=protocol.log']
        (run/'command.json').write_text(json.dumps(cmd, indent=2))
        with (run/'run.log').open('w') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
        text = (run/'run.log').read_text()
        transitions = (run/'protocol.log').read_text()
        row = dict(returncode=result.returncode, functional_pass='SPM_RELEASE_GETX PASS errors=0' in text,
                   buffered_transfer=transitions.count('MX_A>IX_A'),
                   stale_release=transitions.count('SPM_PUT_old'))
        (run/'result.json').write_text(json.dumps(row, indent=2))
        print(name, row, flush=True)
        if name == 'fixed':
            assert row['returncode'] == 0 and row['functional_pass'], row
            assert row['buffered_transfer'] > 0 and row['stale_release'] > 0, row


if __name__ == '__main__':
    main()
