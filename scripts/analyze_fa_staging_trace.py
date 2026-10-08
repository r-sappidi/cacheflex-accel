#!/usr/bin/env python3
"""Reconstruct observed request intervals around the selected staging load.

Begin/Done matching gives a lower bound, not general exact occupancy: aliases
and requests predating the trace can be absent. Sixteen distinct observed
in-flight requests nevertheless prove saturation of a sixteen-entry queue.
Reads debug logs as bytes because packet dumps need not be valid UTF-8.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re


def analyze(root):
    source = root / 'trace/diagnostic.trace'
    seq = re.compile(rb'^\s*(\d+)\s+0\s+Seq\s+(Begin|Done)\s+>\s+'
                     rb'\[(0x[0-9a-f]+), line (0x[0-9a-f]+)\]\s+(.*)')
    pending = collections.defaultdict(collections.deque)
    intervals, evidence, attempts = [], [], []
    selected = json.loads((root / 'window.json').read_text())
    # This analysis is deliberately restricted to the diagnosed trace/instance.
    sn = b'158524822'
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for line in stream:
            digest.update(line)
            match = seq.match(line)
            if match:
                tick, event, _, address, detail = match.groups()
                tick = int(tick)
                if event == b'Begin':
                    pending[address].append((tick, detail.decode().strip()))
                elif pending[address]:
                    start, kind = pending[address].popleft()
                    intervals.append(dict(start=start, end=tick, kind=kind,
                                          line=address.decode()))
            if b'[sn:' + sn + b']' in line or b'CFBOUND pc=0x401e14 sn=' + sn in line:
                evidence.append(line.decode('utf-8', errors='backslashreplace').rstrip())
                if b'Memory request (pkt:' in line:
                    attempts.append(dict(tick=int(line.split(b':', 1)[0]),
                                         accepted=b'was sent (' in line))
    # Only completed, paired intervals are used. No invented start times.
    for attempt in attempts:
        tick = attempt['tick']
        active = [x for x in intervals if x['start'] <= tick < x['end']]
        attempt['observed_inflight_end_of_tick'] = len(active)
        attempt['types'] = dict(collections.Counter(x['kind'] for x in active))
        attempt['intervals'] = active
    assert attempts and attempts[0]['types'] == {'SPMCP_fetch': 16}
    assert attempts[-1]['accepted'] and attempts[-1]['tick'] == 27178811200
    result = dict(trace_sha256=digest.hexdigest(), selected_window=selected,
                  core=0, pc='0x401e14', sn=int(sn), ticks_per_cycle=400,
                  first_attempt=attempts[0]['tick'], accepted=attempts[-1]['tick'],
                  admission_wait_cycles=(attempts[-1]['tick']-attempts[0]['tick'])//400,
                  attempts=attempts, instruction_evidence=evidence,
                  caveat='Observed paired intervals are a lower bound; end-of-tick '
                         'occupancy does not resolve within-tick ordering.')
    (root / 'admission_analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['trace_sha256', 'admission_wait_cycles']}))
    print([(x['tick'], x['accepted'], x['types']) for x in attempts])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
