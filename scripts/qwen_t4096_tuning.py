"""L3-11: isolated five-case completion, original numerical acceptance intact."""
import ast
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import threading
import time

ROOT = Path('/data2/rsappidi/cacheflex_mc')
sys.path.insert(0, str(ROOT/'scripts'))
from build_access_timing import save, sha
from parallel_budget_scheduler import owned

BASE = ROOT/'results/cacheflex_l3_spm/qwen_t4096_tuning_20260929'
PRIOR = ROOT/'results/cacheflex_l3_spm/qwen_cache_analytical_20260928'
JOURNAL = ROOT/'results/parallel_budget_20260929'
REMAINDER = ROOT/'results/ai_workloads/qwen_remainder_20260929/experiment_v1/status.json'
GIB = 2**30
KEYS = ('t', 'arm', 'r', 'bc', 'br', 'hq', 'hkv', 'causal', 'depth')


def read(path):
    return json.loads(Path(path).read_text())


def slots_for(remaining, references, budget):
    old = min(budget['max_worker_slots'], remaining + references)
    reserved = budget['outside_reserve'] + (old + 1)*6*GIB + 2*GIB
    slots = min(4, max(0, (budget['ram_limit'] - reserved)//(8*GIB)))
    return int(slots), reserved, old


class BudgetGuard:
    """Bound new children without changing any existing scheduler or worker."""
    def __init__(self, out):
        self.out = Path(out)
        self.budget = read(JOURNAL/'budget.json')
        self.lock = threading.RLock()
        self.children = {}
        self.previous = None
        self.finished = threading.Event()
        self.error = None
        self.lockfile = (self.out/'coordinator.lock').open('w')
        fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.sched_setaffinity(0, self.budget['cpu_set'])
        resource.setrlimit(resource.RLIMIT_AS,
                           (2*GIB, resource.getrlimit(resource.RLIMIT_AS)[1]))
        self.sample()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()

    def sample(self):
        with self.lock:
            old = read(JOURNAL/'status.json')
            old_metric = read(JOURNAL/'resources.json')
            rem = read(REMAINDER)
            # No second independently filling queue may share our spare slots.
            assert rem['stage'] in ('failed', 'completed_validated') and not rem['active'], 'remainder queue became active'
            rem_proc = Path('/proc')/str(rem['pid'])
            assert not rem_proc.exists(), 'remainder coordinator still exists'
            procs = owned()
            now = time.monotonic()
            cpu = None
            if self.previous is not None:
                then, ticks = self.previous
                delta = sum(max(0, p['cpu_ticks']-ticks.get((pid,p['start_ticks']),p['cpu_ticks']))
                            for pid,p in procs.items())
                cpu = 100*delta/os.sysconf('SC_CLK_TCK')/(now-then)/os.cpu_count()
            self.previous = now, {(pid,p['start_ticks']):p['cpu_ticks'] for pid,p in procs.items()}
            rss = sum(p['rss'] for p in procs.values())
            mem = {s.split(':')[0]:int(s.split()[1])*1024
                   for s in Path('/proc/meminfo').read_text().splitlines()}
            refs = sum(x['name'].startswith('reference_') for x in old['active'])
            slots, reserved, old_slots = slots_for(sum(old['remaining'].values()), refs, self.budget)
            stale = time.time()-old['updated'] > 180 or time.time()-old_metric['updated'] > 180
            disk = shutil.disk_usage(self.out).free
            pressure = (stale or rss >= self.budget['ram_limit']-4*GIB or
                        mem['MemAvailable'] < 64*GIB or disk < 100*GIB or
                        (cpu is not None and cpu > 20) or old_metric.get('pressure',False))
            sample = dict(updated=time.time(),pid=os.getpid(),cpu_pct_of_host=cpu,
                user_rss_bytes=rss,user_ram_pct=100*rss/mem['MemTotal'],
                available_bytes=mem['MemAvailable'],disk_free_bytes=disk,
                cpu_set=self.budget['cpu_set'],cpu_cap_pct=100*len(self.budget['cpu_set'])/os.cpu_count(),
                ram_limit_bytes=self.budget['ram_limit'],old_future_slots=old_slots,
                reserved_before_new_simulators=reserved,new_simulator_as=8*GIB,
                allowed_simulators=0 if pressure else slots,pressure=pressure,
                prior_failed_families=list(old.get('failed',{})),children=list(self.children))
            save(self.out/'resource_status.json', sample)
            return sample

    def can_launch(self, active):
        assert self.error is None, self.error
        sample = self.sample()
        return active < sample['allowed_simulators'], sample

    def register(self, proc):
        with self.lock:
            self.children[proc.pid] = (proc, False)

    def unregister(self, proc):
        if proc is not None:
            with self.lock:
                self.children.pop(proc.pid, None)

    def watch(self):
        while not self.finished.wait(5):
            try:
                sample = self.sample()
                with self.lock:
                    pause = sample['pressure'] or len(self.children)>sample['allowed_simulators']
                    for pid,(proc,paused) in list(self.children.items()):
                        if proc.poll() is not None:
                            continue
                        if pause != paused:
                            try:
                                proc.send_signal(signal.SIGSTOP if pause else signal.SIGCONT)
                            except ProcessLookupError:
                                continue
                            self.children[pid] = proc,pause
            except BaseException as exc:
                self.error = repr(exc)
                # The runner observes this and terminates its own peers.
                with self.lock:
                    for proc,_ in self.children.values():
                        if proc.poll() is None:
                            try: proc.send_signal(signal.SIGCONT)
                            except ProcessLookupError: pass
                return

    def close(self):
        self.finished.set()
        self.thread.join(timeout=10)


def fn(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def run_parts(tree):
    body = next(n.body for n in fn(tree,'run').body if isinstance(n,ast.Try))
    start = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and
                 any(isinstance(t,ast.Name) and t.id=='logtext' for t in n.targets))
    launch = next(i for i,n in enumerate(body) if isinstance(n,ast.With))
    return body[:launch], body[start:]


def prepare():
    assert not BASE.exists(), 'Do not duplicate or overwrite this experiment'
    state = read(PRIOR/'experiment_v1/status.json')
    assert state['stage']=='completed_validated'
    selected = [dict(t=4096,arm=a,r=r,bc=b,br=16,hq=14,hkv=2,causal=True,depth=4)
                for a,r,b in [('cache',256,256),('private',128,256),('private',256,512),
                              ('l3',128,256),('l3',256,512)]]
    for c in selected:
        assert not any(all(row['case'][k]==c[k] for k in KEYS) for row in state['completed']), c
    target = [row for row in state['completed'] if row['case']['t']==4096 and row['case']['hq']==14]
    assert len(target)==4
    for arm in ('cache','private','l3'):
        geometries={(r['case']['r'],r['case']['bc']) for r in target if r['case']['arm']==arm}
        geometries.update((c['r'],c['bc']) for c in selected if c['arm']==arm)
        assert geometries=={(128,256),(256,256),(256,512)}
    # Preserve/verify the already accepted experiment's full frozen dependency set.
    frozen=read(PRIOR/'experiment_v1/inputs.json')
    for p,h in frozen.items():
        assert sha(p)==h, ('changed prior input',p)
    BASE.mkdir(parents=True)
    save(BASE/'prior_manifest_verified.json',dict(time=time.time(),files=len(frozen),manifest_sha256=sha(PRIOR/'experiment_v1/inputs.json')))
    plan=dict(ledger='L3-11',related=['L3-06','L3-07','L3-08','L3-09'],selected=selected,
        reused_t4096=target,objective='lowest validated complete T4096 FA ROI per backend',
        geometry_set=[[128,256],[256,256],[256,512]],br=16,max_parallel=4,
        l3_transport='existing NoC; FP16 contribution plus first initialization',
        policy='same frozen guest/model; all original acceptance checks; equal target-size coverage; no global optimum claim')
    save(BASE/'selection.json',plan)
    save(BASE/'model.json',plan)
    save(BASE/'preexisting_queue_state.json',dict(scheduler=read(JOURNAL/'status.json'),remainder=read(REMAINDER)))
    source=PRIOR/'selected_runner.py'
    original=source.read_text()
    s=original
    def replace(a,b):
        nonlocal s
        assert s.count(a)==1, a
        s=s.replace(a,b)
    replace("BASE=ROOT/'results/cacheflex_l3_spm/qwen_cache_analytical_20260928'", "BASE=ROOT/'results/cacheflex_l3_spm/qwen_t4096_tuning_20260929'")
    replace("TUNED=ROOT/'results/cacheflex_l3_spm/qwen_analytical_20260928/experiment_v1'", "TUNED=ROOT/'results/cacheflex_l3_spm/qwen_cache_analytical_20260928/experiment_v1'")
    replace('from run_l3_kernel_tuning import resources', 'from run_l3_kernel_tuning import resources\nfrom qwen_t4096_tuning import BudgetGuard')
    replace("OUT.mkdir(parents=True);(OUT/'runs').mkdir();(OUT/'references').mkdir()", "OUT.mkdir(parents=True);(OUT/'runs').mkdir();(OUT/'references').mkdir()\n    guard=BudgetGuard(OUT)")
    replace('Cache tuning added to completed private/L3 search; same frozen guest and hardware.', 'T4096 equal-candidate completion; same frozen guest and hardware. New five cases plus reused accepted controls.')
    replace('All three backends have bounded analytical candidate searches; rows include accepted reused controls.', 'All three backends get R128/Bc256, R256/Bc256, R256/Bc512 at Br16. L3 uses the original NoC transport.')
    replace("ROOT/'scripts/qwen_cache_tuning.py',ROOT/'docs/experiments/QWEN_CACHE_TUNING.md']:", "ROOT/'scripts/qwen_cache_tuning.py',ROOT/'docs/experiments/QWEN_CACHE_TUNING.md',\n                  ROOT/'scripts/qwen_t4096_tuning.py',ROOT/'docs/experiments/QWEN_T4096_TUNING.md',\n                  ROOT/'scripts/parallel_budget_scheduler.py',ROOT/'scripts/parallel_budget_worker.py',\n                  TUNED/'inputs.json',BASE/'acceptance_audit.json',\n                  ROOT/'results/parallel_budget_20260929/budget.json']:")
    replace("assert not stop.is_set();r=resources();save(path/'resource_launch.json',r)\n                            if r['cpu_idle_pct']>=50 and r['available_gib']>=128 and r['free_disk_gib']>=100 and len(r['simulators'])<6:break", "assert not stop.is_set();admit,r=guard.can_launch(len(state['active']));save(path/'resource_launch.json',r)\n                            if admit:break")
    replace("proc=subprocess.Popen(['nice','-n','10',*cmd],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)", "proc=subprocess.Popen(['prlimit',f'--as={8*2**30}:','--','nice','-n','10',*cmd],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)\n                        guard.register(proc)")
    replace("assert not stop.is_set(),'peer failed'", "assert not stop.is_set(),'peer failed'\n                        assert guard.error is None,guard.error")
    replace("assert time.monotonic()-start<12*3600,'12h timeout'", "assert time.monotonic()-start<96*3600,'96h host timeout including resource waits'")
    replace("if proc and proc.poll() is None:\n                    proc.terminate()", "if proc and proc.poll() is None:\n                    proc.send_signal(__import__('signal').SIGCONT)\n                    proc.terminate()")
    replace("with lock:state['active'].pop(name,None);status()", "guard.unregister(proc)\n                with lock:state['active'].pop(name,None);status()")
    replace('with cf.ThreadPoolExecutor(max_workers=1) as pool:', 'with cf.ThreadPoolExecutor(max_workers=5) as pool:')
    start=s.index("        selected=json.loads((BASE/'selection.json').read_text())['selected']")
    end=s.index("        status('completed_validated')",start)
    s=s[:start]+'''        selected=json.loads((BASE/'selection.json').read_text())['selected']
        save(OUT/'reused_cases.json',[r['name'] for r in baseline_rows])
        save(OUT/'gates_passed.json',dict(reused_baseline_cases=len(baseline_rows),
             full_math=True,exact_output=True,reason='unchanged validated guest/model and previously validated geometries'))
        phase('t4096_equal_candidate_completion',selected)
'''+s[end:]
    replace('**L3-08 completion update:** all-backend accepted comparison with cache tuning, ', '**L3-11 completion update:** equal three-geometry T4096 comparison, ')
    replace("state['error']=repr(exc);status('failed');raise", "state['error']=repr(exc);status('failed');raise\n    finally:\n        guard.close()")
    before,after=ast.parse(original),ast.parse(s)
    checks={name:ast.dump(fn(before,name))==ast.dump(fn(after,name)) for name in ('work','key','label','get_reference')}
    for i,name in enumerate(('command_construction','entire_acceptance_tail')):
        a,b=run_parts(before)[i],run_parts(after)[i]
        checks[name]=ast.dump(ast.Module(body=a,type_ignores=[]))==ast.dump(ast.Module(body=b,type_ignores=[]))
    assert all(checks.values()), checks
    dest=BASE/'selected_runner.py'
    compile(s,str(dest),'exec')
    dest.write_text(s)
    save(BASE/'acceptance_audit.json',dict(source=str(source),source_sha256=sha(source),derived=str(dest),
         derived_sha256=sha(dest),identical=checks,scheduling='fresh shared budget reservations; 8GiB simulator cap; at most4 active; original acceptance unchanged'))
    print(json.dumps(dict(root=str(BASE),cases=selected,checks=checks),indent=2))


if __name__=='__main__':
    assert sys.argv[1:]==['--prepare']
    prepare()
