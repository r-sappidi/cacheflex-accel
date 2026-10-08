"""Frozen failed VL4 prefix with bounded syscall/functional/controller trace."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'results/cacheflex_flash_attention'
OLD=B/'micro_gcc15_export_revalidation_v2_20260919/runs/ruby_spm_vl4'
OUT=B/'gcc15_vl4_functional_diagnosis_20260920'
def save(p,obj):p.write_text(json.dumps(obj,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir();state=dict(pid=os.getpid(),original=str(OLD))
    def status(stage):state.update(stage=stage,updated=time.time());save(OUT/'status.json',state);print(stage,flush=True)
    cmd=json.loads((OLD/'command.json').read_text())
    cmd=[f'--outdir={OUT}' if x.startswith('--outdir=') else x for x in cmd]
    cmd[1:1]=['--debug-flags=SyscallVerbose,PseudoInst,RubySystem,RubySlicc,RubyPort,Exec',
              '--debug-start=66500000000','--debug-end=66600000000','--debug-file=failure.trace']
    cmd+=['--abs-max-tick=66600000400']
    deps={x:sha(x) for x in (cmd[0],cmd[cmd.index('--cmd')+1])}
    save(OUT/'command.json',cmd);save(OUT/'inputs.json',deps)
    status('replaying_failed_prefix')
    start=time.monotonic()
    env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib')
    try:
        with (OUT/'run.log').open('w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            state['child_pid']=child.pid
            try:
                while child.poll() is None:
                    trace=OUT/'failure.trace';size=trace.stat().st_size if trace.exists() else 0
                    state.update(host_elapsed_s=time.monotonic()-start,trace_bytes=size)
                    status('replaying_failed_prefix')
                    if size>512*1024**2 or time.monotonic()-start>3600:raise RuntimeError('diagnostic bound exceeded')
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:pass
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=30)
        state['returncode']=child.returncode
        assert all(sha(p)==s for p,s in deps.items()),'input changed'
        log=(OUT/'run.log').read_text(errors='replace')
        assert 'Ruby functional read failed for address 0x9d480' in log,'did not reproduce exact failure'
        status('failure_reproduced_trace_ready')
    except BaseException as e:state['error']=repr(e);status('diagnostic_failed');raise
if __name__=='__main__':main()
