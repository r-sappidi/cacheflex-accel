"""Scheduling-only adapter: run the ORIGINAL acceptance tail without edits.

The original coordinator files remain frozen. Each worker gets an isolated
namespace, bounded address space and single-thread numerical libraries. Existing
simulators are observed through their stopped parent's zombie exit status.
"""
import ast
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import textwrap
import threading
import time
import traceback

ROOT = Path('/data2/rsappidi/cacheflex_mc')
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'benchmarks/ai_workloads'),
               str(ROOT/'benchmarks/cacheflex_flash_attention')]
MODULES = dict(diagnostics='run_l3_diagnostics', long='run_qwen_long_context',
               ffn='run_qwen_swiglu', projection='run_qwen_projection_v2')
GIB = 2**30


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    p = Path(p)
    tmp = p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def process(pid):
    try:
        x = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        return dict(state=x[0], ppid=int(x[1]), start_ticks=int(x[19]),
                    wait_status=int(x[49]), cpu_ticks=int(x[11])+int(x[12]))
    except FileNotFoundError:
        return None


def acceptance_function(module):
    """Extract only the unmodified statements AFTER simulation completion."""
    tree = ast.parse(Path(module.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='main')
    run = next(n for n in ast.walk(main) if isinstance(n, ast.FunctionDef) and n.name=='run')
    body = next(n.body for n in run.body if isinstance(n, ast.Try)) if module.__name__ in (
        'run_qwen_swiglu', 'run_qwen_projection_v2') else run.body
    # The launch with-statement is followed by the log read and ALL acceptance
    # checks. Never rewrite or omit any statement in this suffix.
    idx = next(i for i,n in enumerate(body) if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('log','text')
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
               and n.value.func.attr=='read_text')
    tail = copy.deepcopy(body[idx:])
    assert isinstance(tail[-1], ast.Return)
    original_dump = ast.dump(ast.Module(body=body[idx:], type_ignores=[]))
    assert original_dump == ast.dump(ast.Module(body=tail, type_ignores=[]))
    fn = ast.FunctionDef(name='accept_original', args=ast.arguments(posonlyargs=[], args=[],
            kwonlyargs=[], kw_defaults=[], defaults=[]), body=[ast.Global(names=['proc'])]+tail, decorator_list=[])
    compiled = compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                       module.__file__+':original_acceptance_tail', 'exec')
    return compiled, hashlib.sha256(original_dump.encode()).hexdigest()


def constrain(pid, cpus, address_bytes):
    # Soft limits are reversible by the same UID; hard limits remain unchanged.
    hard = resource.prlimit(pid, resource.RLIMIT_AS)[1]
    resource.prlimit(pid, resource.RLIMIT_AS, (address_bytes, hard))
    for task in Path(f'/proc/{pid}/task').iterdir():
        try:
            os.sched_setaffinity(int(task.name), cpus)
        except ProcessLookupError:
            pass


class Adopted:
    def __init__(self, row):
        self.pid = row['pid']
        self.ident = row['identity']
        self.returncode = None

    def poll(self):
        now = process(self.pid)
        assert now and now['start_ticks']==self.ident['start_ticks'], 'lost adopted simulator'
        if now['state']=='Z':
            self.returncode = os.waitstatus_to_exitcode(now['wait_status'])
        return self.returncode


def main():
    spec = read(sys.argv[1])
    journal = Path(spec['journal'])
    journal.mkdir(exist_ok=True)
    cpus = set(spec['cpus'])
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                      PYTHONDONTWRITEBYTECODE='1')
    constrain(os.getpid(), cpus, int(1.5*GIB))
    family = spec['family']
    module = importlib.import_module(MODULES[family])
    out = Path(spec['out'])
    frozen = read(spec['manifest'])
    for p,h in frozen.items():
        assert sha(p)==h, ('changed frozen input',p)
    def freeze(p):
        p = str(p); h = sha(p)
        assert p not in frozen or frozen[p]==h, ('changed frozen input',p)
        frozen[p] = h
    def unchanged():
        for p,h in frozen.items():
            assert sha(p)==h, ('changed frozen input',p)
    def status(*args):
        pass  # Only the supervisor writes the authoritative queue status.
    def local_save(p, value):
        if Path(p).name=='accepted.json':
            save(journal/'validated.json', value)
        elif Path(p)==out/'inputs.json':
            save(journal/'inputs.json', value)
        else:
            save(p,value)

    if spec.get('reference'):
        ref = spec['reference']; path=Path(ref['path'])
        assert not path.exists(), ('reference already exists',path)
        path.parent.mkdir(exist_ok=True)
        if family=='long':
            # Original oracle, same chunking, precision, and generator entrypoint.
            import qwen_long_reference
            sys.argv = [qwen_long_reference.__file__, '--t', str(ref['t']), '--bc',
                        str(ref['bc']), '--out',str(path)]
            try:
                qwen_long_reference.main()
            finally:
                sys.argv = [__file__, spec['_spec_path']]
        elif family=='ffn':
            module.generate(path,ref['t'],896,4864,Path(ref['library']))
        else:
            module.generate(path,ref['t'],ref['op'],Path(ref['library']))
        unchanged()
        save(journal/'done.json',dict(reference=str(path),sha256=sha(path)))
        return

    job = spec['job']; name = spec['name']; path = out/'runs'/name
    replay = spec.get('verify_only',False)
    adopted = spec.get('adopted')
    assert replay or adopted or not path.exists(), ('duplicate run',path)
    if not path.exists(): path.mkdir()
    ns = dict(vars(module))
    state = dict(completed=read(out/'status.json')['completed'],active={},pending=[])
    ns.update(OUT=out,state=state,freeze=freeze,unchanged=unchanged,status=status,
              save=local_save,lock=threading.RLock(),name=name,path=path,
              frozen=frozen,start=time.monotonic(),stop=threading.Event())
    if family in ('diagnostics','long'):
        source=Path(job['source']); prior=read(source/'accepted.json')
        c=dict(prior['case']) if prior.get('case') else None
        for n in ('accepted.json','command.json','config.json','stats.txt','run.log','output.fp16'):
            if (source/n).exists(): freeze(source/n)
        oldcmd=read(source/'command.json');freeze(oldcmd[oldcmd.index('--cmd')+1])
        if family=='diagnostics':
            profile=job['profile'];trace=job.get('trace',False)
            build=Path(spec['build']);cmd=module.command(source,build,path,profile,
                                         module.trace_window(source) if trace else None)
            network=read(source/'config.json')['system']['ruby']['network']
            direct=network['cacheflex_direct_reduction'];width=network['cacheflex_direct_bytes']
            job=dict(job,trace_window=module.trace_window(source) if trace else None)
            ns.update(profile=profile,trace=trace)
        else:
            c['t']=job['t'];direct=job['direct'];width=16
            cmd=module.make_command(oldcmd,path,c['t'],direct)
            job=dict(job,width=16,latency=2)
            ns.update(t=c['t'],latency=2)
        refname=f"t{c['t']}_bc{c['bc']}_q{c['hq']}_kv{c['hkv']}_causal{int(c['causal'])}.npy"
        ref=next(p/'references'/refname for p in (out,module.QWEN,module.TUNED,module.CACHE)
                 if (p/'references'/refname).exists())
        assert sha(ref)==read(ref.with_suffix('.json'))['sha256']
        freeze(ref);freeze(ref.with_suffix('.json'))
        expected=module.np.load(ref)
        ns.update(source=source,prior=prior,c=c,job=job,exact=job['exact'],
                  direct=direct,width=width,expected=expected)
    else:
        c=job;inp=Path(c['inp']);repeats=2
        guest=(module.BASE/'guests'/f"custom_g{c['g']}"/c['arm'] if family=='ffn'
               else module.BASE/'guest_v2'/c['arm'])
        for p in (inp,inp.with_suffix('.json'),inp.with_suffix('.stages.npz' if family=='ffn' else '.npz')):
            freeze(p)
        cmd,args=module.command(path,guest,inp,c,repeats)
        ns.update(c=c,inp=inp,guest=guest,repeats=repeats,args=args)
    if adopted or replay:
        assert read(path/'command.json')==cmd, 'adopted/replay command differs'
    else:
        save(path/'command.json',cmd)
        save(path/('case.json' if family in ('ffn','projection') else 'job.json'),job)
    save(journal/'inputs.json',frozen)
    proc=None
    if replay:
        from types import SimpleNamespace
        proc=SimpleNamespace(returncode=0)
    elif adopted:
        proc=Adopted(adopted)
        constrain(proc.pid,cpus,int(4.5*GIB))
        if process(proc.pid)['state']=='T': os.kill(proc.pid,signal.SIGCONT)
    else:
        def preexec():
            constrain(os.getpid(),cpus,int(4.5*GIB))
        with (path/'run.log').open('x') as log:
            proc=subprocess.Popen(['nice','-n','10',*cmd],cwd=ROOT,env=module.ENV,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=preexec)
    if not replay:
        ident=process(proc.pid)
        assert ident is not None
        save(journal/'worker_status.json',dict(stage='simulating',pid=os.getpid(),
             simulator_pid=proc.pid,simulator_identity=ident,updated=time.time()))
        start=time.monotonic()
        while proc.poll() is None:
            assert time.monotonic()-start<96*3600,'96h simulation timeout'
            assert __import__('shutil').disk_usage(out).free>75*GIB,'disk guard'
            for logname in ('run.log','reduction_trace.log'):
                lp=path/logname
                assert not lp.exists() or lp.stat().st_size<128*2**20,'log/trace guard'
            time.sleep(3)
        save(journal/'exit.json',dict(returncode=proc.returncode,pid=proc.pid,
             adopted=bool(adopted),identity=ident))
    ns['proc']=proc
    # Refresh peers immediately before original validation. Supervisor repeats
    # peer checks under its single-writer publication order for concurrent cases.
    state['completed']=read(out/'status.json')['completed']
    code,digest=acceptance_function(module)
    save(journal/'acceptance_source.json',dict(source=module.__file__,sha256=sha(module.__file__),
         original_tail_ast_sha256=digest,unmodified=True))
    save(journal/'worker_status.json',dict(stage='validating',pid=os.getpid(),updated=time.time()))
    exec(code,ns)
    row=ns['accept_original']()
    unchanged();save(journal/'inputs.json',frozen)
    save(journal/'done.json',dict(name=name,roi_us=row.get('roi_us'),first_us=row.get('first_us'),
                                output_sha256=row.get('output_sha256'),verify_only=replay))


if __name__=='__main__':
    try:
        main()
    except BaseException as e:
        if len(sys.argv)>1 and Path(sys.argv[1]).exists():
            spec=read(sys.argv[1]);save(Path(spec['journal'])/'failed.json',
                  dict(error=repr(e),traceback=traceback.format_exc()))
        raise
