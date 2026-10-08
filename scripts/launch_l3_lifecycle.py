"""Wait for the explicitly selected frozen build, then run its gated queue."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_l3_spm/lifecycle_20260928'
def main():
    build=os.environ['CACHEFLEX_L3_LIFECYCLE_BUILD']
    experiment=os.environ['CACHEFLEX_L3_LIFECYCLE_EXPERIMENT']
    assert not (BASE/experiment).exists(), 'Refusing to overwrite an experiment'
    print('Waiting for',build,'then',experiment,flush=True)
    while True:
        path=BASE/build/'status.json'
        if path.exists():
            state=json.loads(path.read_text())
            assert state['stage']!='failed',state
            if state['stage']=='built_requires_validation':break
        time.sleep(5)
    with (BASE/(experiment+'.host.log')).open('w') as log:
        subprocess.run(['python3',str(ROOT/'scripts/run_l3_lifecycle.py')],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main()
