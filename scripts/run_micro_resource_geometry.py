"""After translated controls pass, test MICRO-sized caches separately."""
import json
import time
from pathlib import Path
import run_micro_resources as runner

def main():
    parent=runner.ROOT/'results/cacheflex_flash_attention/micro_resources_20260921'
    prerequisite=parent/'experiment_v6/status.json'
    print('Waiting for translated-control correctness, stress and guard gates',flush=True)
    while True:
        if prerequisite.exists():
            state=json.loads(prerequisite.read_text())
            if state['stage']=='completed_requires_analysis': break
            if state['stage']=='failed': raise RuntimeError(state)
        time.sleep(15)
    runner.BUILD=parent/'build_v5'
    runner.EX=parent/'experiment_paper_geometry_v2'
    original=runner.profile
    def profile(old,bounded,cap,small=False):
        params=original(old,bounded,cap,small)
        if bounded:
            for i in range(8):
                params[f'system.ruby.l1_cntrl{i}.cache.size']='524288'
                params[f'system.ruby.l2_cntrl{i}.spm_retain_llc']=False
        return params
    runner.profile=profile
    runner.save(parent/'geometry_driver_inputs_v2.json', {
        str(Path(__file__)):runner.sha(Path(__file__)),
        str(Path(runner.__file__)):runner.sha(Path(runner.__file__))})
    runner.main()

if __name__=='__main__': main()
