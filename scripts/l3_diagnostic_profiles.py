"""Timing-only profiles and conservative host admission for MOD-10."""
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path('/data2/rsappidi/cacheflex_mc')
BASE = ROOT/'results/cacheflex_l3_spm/diagnostics_20260929'
DIRECT = ROOT/'results/cacheflex_l3_spm/direct_link_20260928/experiment_v2'
PROFILES = {
    'default': {},
    'read_fast': {'read': 1},
    'reply_fast': {'reply': 1},
    'both_fast': {'read': 1, 'reply': 1},
    'lanes8': {'arithmetic': 13, 'convert': 10, 'interval': 8},
    'lanes4': {'arithmetic': 21, 'convert': 18, 'interval': 16},
    'pipeline': {'pipeline': True},
}
OPTIONS = {'read': '--diag-reduction-read', 'reply': '--diag-reduction-reply',
           'arithmetic': '--cacheflex-l3-reduction-arithmetic',
           'convert': '--cacheflex-l3-reduction-convert',
           'interval': '--cacheflex-l3-reduction-interval',
           'pipeline': '--cacheflex-l3-reduction-pipeline'}
FIELDS = {'read': 'l3_reduction_read_delay', 'reply': 'l3_reduction_reply_delay',
          'arithmetic': 'l3_reduction_arithmetic', 'convert': 'l3_reduction_convert',
          'interval': 'l3_reduction_interval', 'pipeline': 'l3_reduction_pipeline'}

def resources():
    def cpu():
        return list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    a = cpu(); time.sleep(.5); b = cpu()
    delta = [y-x for x, y in zip(a, b)]
    mem = {k: int(v.split()[0]) for k, v in
           (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    sims = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            cmd = (p/'cmdline').read_bytes().split(b'\0')
            if cmd and Path(os.fsdecode(cmd[0])).name in ('gem5.opt', 'gem5.debug', 'gem5.fast'):
                sims.append(int(p.name))
        except OSError: pass
    return dict(cpu_idle_pct=100*delta[3]/max(1, sum(delta)),
                available_gib=mem['MemAvailable']/2**20,
                swap_used_gib=(mem['SwapTotal']-mem['SwapFree'])/2**20,
                free_disk_gib=shutil.disk_usage(BASE).free/2**30, simulators=sims)

def can_launch(r, simulator=True):
    return (r['cpu_idle_pct'] >= 50 and r['available_gib'] >= 128 and
            r['free_disk_gib'] >= 100 and r['swap_used_gib'] == 0 and
            (not simulator or len(r['simulators']) < 6))

def command(source, build, path, profile, trace=None):
    cmd = json.loads((Path(source)/'command.json').read_text())
    cmd[0] = str(build/'gem5.opt'); cmd[1] = '--outdir='+str(path)
    cmd[2] = str(build/'configs/deprecated/example/l3_diagnostics_se.py')
    for key, value in PROFILES[profile].items():
        option = OPTIONS[key]
        assert value is True or option not in cmd, ('unexpected separate option form', option)
        cmd = [x for x in cmd if x != option and not x.startswith(option+'=')]
        cmd.append(option if value is True else f'{option}={value}')
    if trace:
        start, end = trace
        assert 0 <= start < end
        cmd[2:2] = ['--debug-flags=CacheFlexReductionTrace',
                    '--debug-file=reduction_trace.log',
                    f'--debug-start={start}', f'--debug-end={end}']
    return cmd

def allowed_config(profile):
    values = {'read': 0, 'reply': 0, **PROFILES[profile]}
    return {f'system.ruby.l2_cntrl{i}.{FIELDS[k]}': v
            for i in range(8) for k, v in values.items()}

def trace_window(source):
    """Observe 100us starting 2us after the last pre-ROI progress timestamp."""
    tick = None
    import re
    with (Path(source)/'run.log').open() as stream:
        for line in stream:
            if 'KV_ROI_BEGIN' in line: break
            m = re.match(r'(\d+): .*progress event', line)
            if m: tick = int(m[1])
        else: raise ValueError('No ROI start marker')
    assert tick is not None
    return tick+2_000_000, tick+102_000_000
