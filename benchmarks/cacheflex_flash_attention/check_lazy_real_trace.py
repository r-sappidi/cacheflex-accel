"""Exact full-result comparison against the archived small residency trace."""
from pathlib import Path
import json,shutil,sys
import analyze_residency as a
if '--fast-events' in sys.argv:
    from fast_residency_events import events
    a.events=events
ROOT=Path(__file__).resolve().parents[2]
old=ROOT/'results/cacheflex_flash_attention/spm_capacity_probe_20260909'
source=old/'cache_t128_separate0_trace1';out=Path(sys.argv[1]).resolve();out.mkdir()
(out/'debug.trace').symlink_to(source/'debug.trace');shutil.copy2(source/'run.log',out/'run.log')
actual=a.analyze(out,old/'cache.debug',128,fast_occupancy=True)
expected=json.loads((source/'residency.json').read_text())
assert actual==expected,'Lazy integration changed real trace results'
assert (out/'evictions.csv').read_bytes()==(source/'evictions.csv').read_bytes()
(out/'PASS.json').write_text(json.dumps({'exact_residency_json':True,'byte_identical_evictions_csv':True},indent=2))
print('PASS: exact residency JSON and eviction CSV',flush=True)
