"""Cross-check online timing against the previously hand-joined single-copy trace.

Waits for the all-copy queue, so this adds at most one simulator afterward.
Only the selected copy's admission tick is observed; no verbose trace is written.
"""
import json
import os
from pathlib import Path
import subprocess
import time
from build_access_timing import ROOT, sha, save
from run_access_timing import stats

def main():
    root=ROOT/'results/cacheflex_flash_attention/access_aggregate_20260922'
    experiment=root/'experiment_v3';build=root/'build_v3'
    out=root/'trace_crosscheck';out.mkdir()
    proc=None
    def status(stage,**kw):save(out/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**kw))
    try:
        status('waiting_for_t4096')
        while True:
            stage=json.loads((experiment/'status.json').read_text())['stage']
            assert stage!='failed','Main aggregate queue failed'
            if stage=='completed_validated':break
            time.sleep(10)
        old=ROOT/'results/cacheflex_flash_attention/micro_resources_20260921/experiment_v6/runs/spm_bounded_t512'
        evidence=ROOT/'results/cacheflex_flash_attention/access_timing_20260922/experiment_v3'
        target=json.loads((evidence/'target.json').read_text())
        cmd=json.loads((old/'command.json').read_text())
        cmd=[f'--outdir={out}' if a.startswith('--outdir=') else a for a in cmd]
        cmd[0]=str(build/'gem5.opt');cmd[2]=str(build/'configs/deprecated/example/bounded_se.py')
        save(out/'command.json',cmd)
        files=[Path(__file__),build/'gem5.opt',Path(cmd[cmd.index('--cmd')+1]),
               old/'stats.txt',old/'output.fp16',evidence/'target.json',evidence/'breakdown.json']
        files+=list((build/'configs').rglob('*.py'))
        frozen={str(p):sha(p) for p in files};save(out/'inputs.json',frozen)
        tick=str(target['begin']['tick'])
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
                 CACHEFLEX_AGGREGATE_PATH=str(out/'aggregate.json'),
                 CACHEFLEX_AGGREGATE_START=tick,CACHEFLEX_AGGREGATE_END=tick)
        with (out/'run.log').open('w') as log:
            proc=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status('running',simulator=proc.pid)
            proc.wait(timeout=600)
        assert proc.returncode==0,proc.returncode
        assert stats(old)==stats(out),'Timing drift'
        assert sha(old/'output.fp16')==sha(out/'output.fp16'),'Output drift'
        for p,h in frozen.items():assert sha(p)==h,('Changed input',p)
        d=json.loads((out/'aggregate.json').read_text())
        assert d['begun']==d['completed']==1
        a=d['groups']['all'];c=a['category_ticks']
        assert a['total_ticks']==284500
        expected={
            'CPU_issue_to_execute':400,'CPU_execute_to_store_eligible':5600,
            'CPU_store_eligible_to_Ruby_acceptance':1600,'L1_ingress_FIFO_wait':198800,
            'L1_forward_block:(*m_writeCredits_ptr)':3600,
            'L1_forward_block:(*m_downstreamWriteCredits_ptr)':33200,
            'private_L2_input_wait':2100,'network_message_wait_before_NI':12000,
            'network_NI_serialization_flow_control':6500,
            'network_router_buffer_arbitration':10500,'network_pipeline_links':6000}
        for k,v in expected.items():assert c.get(k)==v,(k,c.get(k),v)
        assert sum(c.values())==a['total_ticks'] and d['overlap_ticks']==0
        save(out/'validated.json',dict(single_copy_matches_prior_trace=True,
            total_core_cycles=711.25,explicit_category_tick_matches=expected,
            exact_all_nonhost_stats=True,exact_output=True,
            note='LLC unmatched-event count includes intentionally unobserved copies outside this one-tick admission window; not a whole-ROI coverage test'))
        status('validated')
    except BaseException as exc:
        if proc and proc.poll() is None:proc.terminate();proc.wait(timeout=10)
        status('failed',error=repr(exc));raise

if __name__=='__main__':main()
