"""Summarize accepted DSB observer records, never infer additive speedup."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[4];P=R/'results/cacheflex_flash_attention/dsb_diag_20261003/experiment_v1'
d=json.loads((P/'status.json').read_text())
lines=['FA-37 accepted DSB counter summary',f"Stage: {d['stage']}",'',
 'All values below are sums across cores, not wall-clock cycles saved.',
 'Rename/commit samples overlap. Dependencies may be speculative or have other blockers.',
 'Coherent-only means the existing commit drain condition; not a proof that all SPM work is absent.','']
for row in d['completed']:
 if not row['observer_enabled']:continue
 c=row['counters'];total=lambda suffix:sum(v for k,v in c.items() if k.endswith(suffix))
 lines += [row['name']+f" ROI {row['roi_us']:.4f} us (exact prior stats/output)"]
 for suffix in ['.arrivals','.retired','.headSamples','.storeWaitSamples','.widthDeferredSamples',
                '.waitMask::0','.waitMask::1','.waitMask::2','.waitMask::3','.waitMask::4','.waitMask::5','.waitMask::6','.waitMask::7',
                '.renameWaitSamples','.renameCoherentFrontSamples','.renameCoherentFrontUnique',
                '.youngerCoherentDependencies','.youngerSpmDependencies','.youngerBridgeDependencies']:
  lines.append(f'  {suffix[1:]}: {total(suffix):.0f}')
 lines.append('')
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
