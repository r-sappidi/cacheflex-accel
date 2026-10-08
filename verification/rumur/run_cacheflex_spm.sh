#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 - <<'PY'
from pathlib import Path
import json,sys
p=Path('results/rumur_refresh_20261001/verification_v6/status.json')
if p.exists():
 s=json.loads(p.read_text())
 print('Existing verification:',p,'stage:',s['stage'],'PID:',s['pid'])
 print('Preserving existing attempt. Inspect status/logs; use a new versioned attempt for a rerun.')
 sys.exit(0)
import subprocess
sys.exit(subprocess.call(['python3','scripts/run_rumur_refresh_v6.py']))
PY
