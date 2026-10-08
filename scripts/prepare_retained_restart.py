"""Derive versioned runners: same numerical checks, new binary and host admission."""
from pathlib import Path
import ast,json,hashlib
R=Path('/data2/rsappidi/cacheflex_mc');B=R/'results/retained_protocol_restart_20261002';G=B/'build'
s=(R/'scripts/run_qwen_fa_matrix_v4.py').read_text()
s=s.replace("OUT=BASE/os.environ.get('MATRIX_EXPERIMENT','experiment_v4')","OUT=ROOT/'results/retained_protocol_restart_20261002/fa_matrix'")
s=s.replace("resource.RLIMIT_AS,(2*GIB,","resource.RLIMIT_AS,(4*GIB,")
s=s.replace("frozen=read(GUEST/'frozen.json');", "frozen=read(ROOT/'results/retained_protocol_restart_20261002/build/frozen.json');frozen.update(read(GUEST/'frozen.json'));")
s=s.replace("cmd[1]='--outdir='+str(path)","cmd[0]=str(ROOT/'results/retained_protocol_restart_20261002/build/gem5.opt');cmd[1]='--outdir='+str(path)")
s=s.replace('>=8 or','>=3 or').replace('broker.total(b)+8*GIB','broker.total(b)+24*GIB').replace('dict(bytes=8*GIB','dict(bytes=24*GIB').replace('simcap=6*GIB','simcap=22*GIB').replace('max_workers=8','max_workers=3').replace('max_new_simulators=8','max_new_simulators=3').replace('max_parallel=8','max_parallel=3')
# Remove old-platform numbers from new winners metadata; don't imply new-model ratios.
s=s.replace(',prior_private_us=9473.3184,prior_cache_us=10509.4604','')
(B/'run_matrix.py').write_text(s)
s=(R/'scripts/run_qwen_long_context.py').read_text()
s=s.replace("BUILD=PARENT.parent/'build_v4'","BUILD=ROOT/'results/retained_protocol_restart_20261002/build'\nGATES=ROOT/'results/retained_protocol_restart_20261002/gates'\nsys.path.insert(0,str(ROOT/'results/parallel_overlap_20260930'))\nimport broker,resource,signal\nfrom parallel_budget_scheduler import owned\nGIB=2**30")
s=s.replace("OUT=BASE/'experiment_v1'","OUT=ROOT/'results/retained_protocol_restart_20261002/long_context'")
s=s.replace("name=f'exact_{a}'","name=f'revalidate_{a}'").replace('exact=True) for a in ARMS','exact=False) for a in ARMS')
s=s.replace("    OUT.mkdir(parents=True);", "    budget=json.loads((ROOT/'results/parallel_budget_20260929/budget.json').read_text());os.sched_setaffinity(0,budget['cpu_set']);resource.setrlimit(resource.RLIMIT_AS,(8*GIB,resource.RLIM_INFINITY))\n    simcap=48*GIB\n    OUT.mkdir(parents=True);")
s=s.replace("(PARENT/'complete.json')","(GATES/'complete.json')").replace("(PARENT/'status.json')","(GATES/'status.json')")
a=s.index('        def headroom(path):');z=s.index('        def get_reference(c):',a)
s=s[:a]+'''        def headroom(path):
            while True:
                with broker.locked() as b:
                    old=b['leases'].get('retained_long',{}).get('bytes',0)
                    if not b['hold_new'] and time.time()-b['heartbeat']<20 and broker.total(b)-old+simcap+8*GIB<=b['cap']:
                        b['leases']['retained_long']=dict(bytes=simcap+8*GIB,owner=os.getpid(),family='retained long');return
                status('waiting_for_memory_reservation');time.sleep(5)

        def watch(start):
            nonlocal simcap
            assert time.monotonic()-start<96*3600,'96h timeout'
            ps=owned().get(proc.pid);hold=False
            if ps and ps['vms']>=simcap-2*GIB:
                with broker.locked() as b:
                    if broker.total(b)+4*GIB<=b['cap']:
                        b['leases']['retained_long']['bytes']+=4*GIB;simcap+=4*GIB
                        resource.prlimit(proc.pid,resource.RLIMIT_AS,(simcap,resource.RLIM_INFINITY))
                    else:hold=True
            rss=sum(x['rss'] for x in owned().values());avail=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
            hold|=rss>budget['ram_limit']-2*GIB or avail<64*GIB+max(0,budget['ram_limit']-rss)
            if ps and ps['state']!='Z':
                if hold and ps['state']!='T':proc.send_signal(signal.SIGSTOP)
                elif not hold and ps['state']=='T':proc.send_signal(signal.SIGCONT)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:status()

'''+s[z:]
s=s.replace('(OUT,QWEN,TUNED,CACHE)','(OUT,BASE/\'experiment_v1\',QWEN,TUNED,CACHE)')
s=s.replace("proc=subprocess.Popen(['nice','-n','10',*cmd],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)","proc=subprocess.Popen(['prlimit',f'--as={simcap}:','--','nice','-n','10',*cmd],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)")
(B/'run_long.py').write_text(s)
# Verify the arithmetic validation calls themselves remain identical AST subtrees.
for old,new in [('scripts/run_qwen_fa_matrix_v4.py','run_matrix.py'),('scripts/run_qwen_long_context.py','run_long.py')]:
 def calls(text):return [ast.dump(n) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('validate','export_hash','check_global_guard')]
 assert calls((R/old).read_text())==calls((B/new).read_text())
(B/'runner_audit.json').write_text(json.dumps(dict(original_numerical_calls_identical=True,delta='new binary/output roots; same guests/oracles/checks; revalidate rather than old-protocol exact timing; bounded validator8GiB and memory admission',matrix_concurrency=3,long_concurrency=1),indent=2))
print('Prepared new-model matrix and long-context runners.')
