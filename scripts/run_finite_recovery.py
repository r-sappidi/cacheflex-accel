"""Build-gated finite validation, pilots, then finite-only analytical tuning."""
import json,os,subprocess,time
from pathlib import Path
from build_access_timing import save
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_flash_attention/finite_model_20260926'
BUILD=os.environ.get('CACHEFLEX_FINITE_BUILD','build_v2')
EX=os.environ.get('CACHEFLEX_FINITE_EXPERIMENT','experiment_v4')

def main():
    record=BASE/(EX+'_coordinator.json')
    assert not record.exists(),record
    def status(stage,**kw):save(record,dict(stage=stage,pid=os.getpid(),updated=time.time(),build=BUILD,experiment=EX,**kw))
    status('waiting_for_build')
    while True:
        p=BASE/BUILD/'status.json'
        if p.exists():
            s=json.loads(p.read_text())
            if s['stage']=='failed':status('build_failed',error=s.get('error'));return
            if s['stage']=='built_requires_validation':break
        time.sleep(5)
    env=dict(os.environ,CACHEFLEX_FINITE_BUILD=BUILD,CACHEFLEX_FINITE_EXPERIMENT=EX)
    for key in ('CACHEFLEX_FINITE_REUSE_DEFAULTS','CACHEFLEX_FINITE_REUSE_RUNS'):env.pop(key,None)
    with (BASE/(EX+'.host.log')).open('w') as log:
        proc=subprocess.Popen(['python3','scripts/run_finite_micro.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        status('validation_and_pilots',child_pid=proc.pid)
        rc=proc.wait()
    if rc:status('validation_failed',returncode=rc);return
    env['CACHEFLEX_FINITE_SOURCE_EXPERIMENT']=EX
    with (BASE/(EX+'_analytical.host.log')).open('w') as log:
        proc=subprocess.Popen(['python3','scripts/run_finite_analytical_tuning.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        status('analytical_tuning',child_pid=proc.pid)
        rc=proc.wait()
    status('completed' if rc==0 else 'tuning_failed',returncode=rc)

if __name__=='__main__':main()
