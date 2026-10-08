"""Use one simulator slot for wide mapping regressions before FA targets.

Temporarily pause ONLY our placement coordinator (not its simulators), allow
its current children to finish, run two short tests, then resume in finally.
This preserves the two-new-simulator resource limit and all frozen inputs.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from build_access_timing import sha,save
from run_access_timing import stats
from test_micro_timing import check_global_guard
import run_finite_micro as finite

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_flash_attention/spm_set_spread_20260927'
OUT=BASE/'wide_validation_v2'


def alive(pid):
    try:
        # A stopped parent cannot reap completed children; zombies use no CPU.
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0] != 'Z'
    except FileNotFoundError:return False


def main():
    OUT.mkdir()
    state=json.loads((BASE/'experiment/status.json').read_text())
    parent=state['pid']
    parent_live=alive(parent)
    if parent_live:
        assert b'scripts/run_spm_set_spread.py' in Path(f'/proc/{parent}/cmdline').read_bytes()
    children=[r['pid'] for r in state['active'].values()]
    if parent_live:os.kill(parent,signal.SIGSTOP)
    # Re-read after stop to include any launch racing the first status read.
    state=json.loads((BASE/'experiment/status.json').read_text())
    children=list(set(children+[r['pid'] for r in state['active'].values()]))
    def status(stage,**kw):save(OUT/'status.json',dict(stage=stage,pid=os.getpid(),coordinator=parent,updated=time.time(),**kw))
    try:
        status('waiting_for_existing_children',children=children)
        while any(alive(pid) for pid in children):time.sleep(5)
        guest=BASE/'wide_guest/guest';sim=BASE/'build/gem5.opt'
        seed=ROOT/'results/cacheflex_flash_attention/finite_model_20260926/experiment_v5/runs/finite_kv0_spm1'
        frozen={str(p):sha(p) for p in [guest,sim,Path(__file__),seed/'command.json',
            ROOT/'benchmarks/spm_smoke/spm_set_spread.cpp',ROOT/'benchmarks/spm_smoke/build_set_spread.sh']}
        save(OUT/'inputs.json',frozen)
        for stride in (0,512):
            p=OUT/f'stride{stride}';p.mkdir()
            cmd=json.loads((seed/'command.json').read_text())
            cmd[0]=str(sim);cmd[1]='--outdir='+str(p)
            cmd[cmd.index('--cmd')+1]=str(guest);cmd[cmd.index('--options')+1]=''
            cmd=["system.cpu[0].workload[0].cmd=['spm-set-spread-wide']" if x.startswith('system.cpu[0].workload[0].cmd=') else x for x in cmd]
            for i in range(8):cmd+=['-P',f'system.ruby.l1_cntrl{i}.cache.spm_set_stride={stride}']
            save(p/'command.json',cmd);status('running',stride=stride)
            with (p/'run.log').open('w') as log:
                subprocess.run(cmd,cwd=ROOT,env=finite.ENV,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
            text=(p/'run.log').read_text()
            assert 'SPM_SET_SPREAD PASS errors=0' in text and 'Simulated exit code not 0!' not in text,text[-3000:]
            cfg=json.loads((p/'config.json').read_text())
            for i in range(8):
                assert cfg['system']['cpu'][i]['isa'][0]['sve_vl_se']==16
                assert cfg['system']['ruby'][f'l1_cntrl{i}']['cache']['spm_set_stride']==stride
            for s in stats(p):check_global_guard(s)
            for f,h in frozen.items():assert sha(f)==h,f
            save(p/'accepted.json',dict(stride=stride,vl=16,first_way=1,ways=4,
                sets=[0,511,512,1023],rounds=16,raw_without_wb_barrier=True))
        status('passed')
    except BaseException as e:status('failed',error=repr(e));raise
    finally:
        # Resume even on validation failure; this standalone result must be
        # checked before declaring the broader mapping implementation valid.
        if parent_live:os.kill(parent,signal.SIGCONT)

if __name__=='__main__':main()
