"""Retirement-boundary intervals, not instruction service-time sums."""
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess

def markers(elf):
    result={}
    for line in subprocess.check_output(['aarch64-linux-gnu-nm','-n',str(elf)],text=True).splitlines():
        w=line.split()
        if len(w)==3 and w[2].startswith('cf_boundary_'):
            phase=re.sub(r'_\d+$','',w[2].removeprefix('cf_boundary_'))
            pc=int(w[0],16)
            assert pc not in result or result[pc]==phase,('ambiguous boundary',pc)
            result[pc]=phase
    assert result
    return result

def analyze(path,elf):
    lookup=markers(elf);stats={}
    for line in (path/'stats.txt').read_text().split('End Simulation Statistics')[0].splitlines():
        w=line.split()
        if len(w)>1:
            try:stats[w[0]]=float(w[1])
            except ValueError:pass
    end=int(stats['finalTick']);start=end-int(stats['simTicks'])
    events=defaultdict(list)
    for line in (path/'boundaries.trace').read_text().splitlines():
        m=re.search(r'(\d+): system.cpu(\d+)\.commit: CFBOUND pc=(0x[0-9a-f]+)',line)
        if m:events[int(m[2])].append((int(m[1]),lookup[int(m[3],16)]))
    assert set(events)==set(range(8))
    cores={};arrivals={}
    for c,seq in events.items():
        last=start;phase='unlabeled_start';bins=defaultdict(int)
        for tick,next_phase in seq:
            if tick<start:phase=next_phase;continue
            if tick>end:break
            bins[phase]+=tick-last;last=tick;phase=next_phase
            if phase=='final_sync':arrivals[c]=tick
        bins[phase]+=end-last
        assert sum(bins.values())==end-start
        cores[c]={k:v/1e6 for k,v in bins.items()}
    assert set(arrivals)==set(range(8))
    critical=max(arrivals,key=arrivals.get)
    result=dict(start_tick=start,end_tick=end,per_core_us=cores,
        last_arriving_core=critical,last_arriving_core_phases_us=cores[critical],
        terminal_arrival_spread_us=(max(arrivals.values())-min(arrivals.values()))/1e6,
        warning='Intervals between retiring labeled instructions include intervening stalls and loop control; not execution-unit occupancy or a causal saved-cycle partition.')
    (path/'boundary_analysis.json').write_text(json.dumps(result,indent=2))
    return result
