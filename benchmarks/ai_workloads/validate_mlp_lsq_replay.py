"""Verify the captured duplicate-wakeup failure and exact repaired replay."""
import json
from pathlib import Path
import re
import sys

old, fixed = map(Path, sys.argv[1:3])
def events(path):
    with (path / 'debug.trace').open() as stream:
        return [line.rstrip() for line in stream
                if 'system.cpu4.' in line and
                ('SN=1125645' in line or '[sn:1125645]' in line)]
a, b = events(old), events(fixed)
def count(rows, text):
    return sum(text in line for line in rows)
assert count(a, 'Waking blocked load') == 4
assert count(a, 'Load Issue to Cache') == 2
assert count(b, 'Still waiting:') == 3
assert count(b, 'Waking blocked load') == 1
assert count(b, 'Load Issue to Cache') == 1
assert "Assertion `_numOutstandingPackets == 0' failed" in (old / 'run.log').read_text()
log = (fixed / 'run.log').read_text()
assert 'exiting with last active thread context' in log
assert not re.search(r'Assertion|panic:|fatal:', log)
checks = [dict(re.findall(r'(\w+)=([^\s]+)', line)) for line in log.splitlines()
          if line.startswith('MLP_RESULT')]
assert len(checks) == 2 and all(c['errors'] == c['nonfinite'] == '0' for c in checks)
assert {c['hash'] for c in checks} == {'1d6a606583d58f42'}
from sweep_mlp_study import epochs
timing = epochs(fixed, 1)[0]
assert timing['copies_issued'] == timing['copies_completed'] == 147456
result = dict(passed=True, old_events=a, fixed_events=b, checks=checks, roi=timing)
(fixed / 'validation.json').write_text(json.dumps(result, indent=2))
print(json.dumps(dict(passed=True, roi=timing), indent=2))
