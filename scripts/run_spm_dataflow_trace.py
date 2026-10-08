"""Follow the proven admission-stall blockers without changing hardware."""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import re
import subprocess
import time
from run_fa_staging_trace import ROOT,save,sha,stats

B=ROOT/'results/cacheflex_flash_attention'
OLD=B/'staging_trace_20260917'
OUT=B/'spm_dataflow_trace_20260918'

def main():
    OUT.mkdir();state=dict(pid=os.getpid(),stage='preparing')
    def status(stage):state.update(stage=stage,updated=time.time());save(OUT/'status.json',state);print(stage,flush=True)
    try:
        diagnosis=json.loads((OLD/'admission_analysis.json').read_text())
        targets=sorted({r['line'] for attempt in diagnosis['attempts'] for r in attempt['intervals'] if r['kind']=='SPMCP_fetch'})
        start=diagnosis['selected_window']['begin']-2_000_000
        end=diagnosis['selected_window']['end']+200_000
        prior=Path(json.loads((OLD/'plan.json').read_text())['prior'])
        cmd0=json.loads((prior/'command.json').read_text())
        assert (prior/'accepted.json').exists()
        deps=[Path(cmd0[0]),Path(cmd0[cmd0.index('--cmd')+1]),prior/'command.json',prior/'config.ini']+list((ROOT/'gem5/configs').rglob('*.py'))
        manifest={str(p):sha(p) for p in deps};save(OUT/'inputs.json',manifest)
        save(OUT/'plan.json',dict(prior=str(prior),T=16384,Hq=8,Hkv=1,cores=8,VL=16,
            configuration='Exact retained-SPM multi-head setup from the proven111-cycle stall; NOT current single-head or GCC15 experiment',
            begin=start,end=end,stop=end+4000,targets=targets,max_simulators=2,max_trace_bytes=1024**3,
            purpose='Trace outstanding SPMCP lifetimes through queues/controllers/network; then identify one evidence-backed intervention at unchanged storage',
            limits='Worst-gap-selected window, not representative full-runtime latency fraction; partial replay inherits prior whole-run correctness validation'))
        # Preserve every available controller transition for the blocking lines.
        with (OLD/'trace/diagnostic.trace').open('rb') as src,(OUT/'existing_target_context.log').open('wb') as dst:
            needles=[x.encode() for x in targets]
            for line in src:
                if any(n in line for n in needles):dst.write(line)
        def run(name,trace):
            p=OUT/name;p.mkdir();cmd=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in cmd0]
            if trace:
                cmd[1:1]=['--debug-flags=RubySequencer,RubyPort,ProtocolTrace,RubyQueue,RubyNetwork,RubySlicc,RubyResourceStalls,LSQ,SPM_CPU',f'--debug-start={start}',f'--debug-end={end}','--debug-file=dataflow.trace']
            cmd+=['--abs-max-tick='+str(end+4000)]
            save(p/'command.json',cmd)
            env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib')
            start_host=time.monotonic()
            with (p/'run.log').open('w') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
                try:
                    while child.poll() is None:
                        tracefile=p/'dataflow.trace';size=tracefile.stat().st_size if tracefile.exists() else 0
                        save(p/'progress.json',dict(pid=child.pid,host_elapsed_s=time.monotonic()-start_host,trace_bytes=size,updated=time.time()))
                        if size>1024**3:raise RuntimeError('1GiB trace cap')
                        if time.monotonic()-start_host>12*3600:raise RuntimeError('12h timeout')
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:pass
                finally:
                    if child.poll() is None:child.terminate();child.wait()
            assert child.returncode==0,(name,child.returncode)
            return stats(p)
        status('replaying_trace_and_observer_control')
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            a=pool.submit(run,'control',False);b=pool.submit(run,'trace',True)
            control,traced=a.result(),b.result()
        assert control==traced,'observer changes non-host counters'
        for p,digest in manifest.items():assert sha(p)==digest,p
        save(OUT/'observer_equivalence.json',dict(passed=True,counters=sum(map(len,control))))
        with (OUT/'trace/dataflow.trace').open('rb') as src,(OUT/'target_dataflow.log').open('wb') as dst:
            needles=[x.encode() for x in targets]
            for line in src:
                if any(n in line for n in needles):dst.write(line)
        status('trace_ready_for_dependency_analysis')
    except BaseException as e:state['error']=repr(e);status('failed');raise

if __name__=='__main__':main()
