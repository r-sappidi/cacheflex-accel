"""Keep the human progress log/report current without mutating frozen runs."""
import json, os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = os.environ.get('CACHEFLEX_L3_EXPERIMENT', 'experiment_v7')
OUT = ROOT / 'results/cacheflex_l3_spm/initial_20260928' / EXPERIMENT
LOG = ROOT / 'docs/experiments/L3_SPM_PROGRESS.md'

def main():
    initial = json.loads((OUT / 'status.json').read_text())
    seen = {r['name'] for r in initial['completed']}
    previous = None
    with LOG.open('a') as f:
        f.write(f'\n## Automatic queue journal: {EXPERIMENT}\n\n')
    while True:
        try:
            state = json.loads((OUT / 'status.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(5)
            continue
        lines = []
        if state['stage'] != previous:
            lines.append('Queue stage: `' + state['stage'] + '`.')
            previous = state['stage']
        for r in state['completed']:
            if r['name'] not in seen:
                lines.append(f"Accepted `{r['name']}`: ROI {r['roi_us']:.4f} us; "
                             'all configured numerical/config/resource checks passed.')
                seen.add(r['name'])
        if lines:
            now = datetime.now(timezone.utc).isoformat(timespec='seconds')
            with LOG.open('a') as f:
                f.write('\n' + now + '\n\n' + '\n'.join('- '+s for s in lines) + '\n')
            env = dict(os.environ, CACHEFLEX_L3_EXPERIMENT=EXPERIMENT)
            subprocess.run(['python3', str(ROOT/'scripts/analyze_l3_spm.py')],
                           cwd=ROOT, env=env, check=True)
        if state['stage'] in ('failed', 'completed_requires_analysis'):
            if state['stage'] == 'failed':
                with LOG.open('a') as f:
                    f.write('\nQueue stopped on a failed gate; see status.json and failed.json. '
                            'Failed timing is not an accepted result.\n')
            break
        time.sleep(15)

if __name__ == '__main__':
    main()
