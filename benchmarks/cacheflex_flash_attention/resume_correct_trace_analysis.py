"""Resume only host analysis after the verified lazy-integration optimization."""
from pathlib import Path
import concurrent.futures as cf
import json,subprocess,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
out=Path(sys.argv[1]).resolve()
gate=ROOT/'results/cacheflex_flash_attention/fast_residency_check_20260911/PASS.json'
assert json.loads(gate.read_text())=={'exact_residency_json':True,'byte_identical_evictions_csv':True}
rows=[json.loads(p.read_text()) for p in out.glob('*/observer_gate.json')];assert len(rows)==3
def status(s,**kw):(out/'status.json').write_text(json.dumps(dict(stage=s,**kw),indent=2));print(s,flush=True)
def run(row):
    p=Path(row['path'])
    with (p/'analysis_fast.log').open('w') as log:subprocess.run(['python3',str(HERE/'analyze_correct_residency.py'),str(p),str(row['group'])],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(p.name,'analyzed',flush=True);return row
try:
    status('host_analysis_fast_resume',simulation_reruns=0)
    with cf.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,rows))
    (out/'summary.json').write_text(json.dumps(rows,indent=2));status('complete',simulation_reruns=0)
except BaseException as e:status('failed',error=repr(e));raise
