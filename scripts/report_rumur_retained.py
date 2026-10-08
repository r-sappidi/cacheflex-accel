"""Accepted-only bounded verification summary; no hardware-proof inference."""
import json
from pathlib import Path
BASE=Path('/data2/rsappidi/cacheflex_mc/results/rumur_refresh_20261001/verification_v6')
s=json.loads((BASE/'status.json').read_text())
lines=['VER-01 Rumur refresh — bounded results','Stage: '+s['stage'],'','Positive passes and expected negative counterexamples:']
for r in s['accepted']:
 lines.append(f"{r['name']}: {r['outcome']}; states/rules={r['states']}")
lines+=['','Unexpected/unaccepted outcomes:']+[f"{r['name']}: returncode {r['returncode']}; inspect {r['path']}/run.log" for r in s['failed']]
lines+=['','Pending: '+', '.join(s['pending']),'','Private model is one line/one CPU op per core, with explicit operation bounds.','Network check is a directed dependency witness, not a mesh proof.','L3 check is symbolic ordering, not FP/RTL/full LSQ verification.','No all-bounds success or implementation-refinement claim.']
(BASE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
