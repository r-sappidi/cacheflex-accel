"""OPS-23 follow-up: preserve all active jobs and remove verified retired reserve."""
from pathlib import Path
import ast,json,sys,time
ROOT=Path('/data2/rsappidi/cacheflex_mc');EXT=ROOT/'results/parallel_overlap_20260930'
sys.path.insert(0,str(ROOT/'scripts'))
from parallel_budget_scheduler import owned
from parallel_budget_worker import sha,save,read
assert not any(('llama_cpu_20260925' in r['cmd'] and r['name'].startswith('gem5') and r['state']!='Z') or (r['name'].startswith('python') and len(r['cmd'].split())>2 and r['cmd'].split()[1].endswith('/restart_interrupted_20260928.py') and r['cmd'].split()[2]=='llama' and r['state']!='Z') for r in owned().values())
llama=ROOT/'results/cacheflex_flash_attention/llama_cpu_20260925/experiment_export_safe_20260926/status.json'
assert '48h timeout' in read(llama).get('error','')
s=(EXT/'supervisor.py').read_text();original=s
changes=[
 ('return int(20.5*GIB)  # outside8 + llama9.5 + supervisor1 + AI-12 coordinator2','return int(11*GIB)  # outside8 + supervisor1 + AI-12 coordinator2; retiredllama reserve released'),
 ("lock=(EXT/'supervisor.lock').open('w')","lock=(EXT/'supervisor_v2.lock').open('w')"),
 ("assert not (extension/'committed.json').exists(),'Do not duplicate handoff'","assert not (extension/'committed_v2.json').exists(),'Do not duplicate handoff'"),
 ("assert not (REMAINDER_OUT/'status.json').exists(),'Remainder already started'","assert (REMAINDER_OUT/'status.json').exists(),'Remainder must already be active'"),
 ("        broker.save(EXT/'broker.json',dict(cap=cfg['ram_limit'],old_reserved=old_reserved,leases={},hold_new=False,heartbeat=time.time(),updated=time.time()))","        with broker.locked() as budget_state:\n            budget_state['old_reserved']=old_reserved\n            budget_state['heartbeat']=time.time()"),
 ("save(extension/'committed.json',dict(","save(extension/'committed_v2.json',dict(")]
for a,b in changes:assert s.count(a)==1,a;s=s.replace(a,b)
a="""    with (EXT/'remainder.log').open('x') as log:
        remainder_proc=subprocess.Popen(['/usr/bin/python3',str(EXT/'remainder_runner.py')],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(EXT/'remainder_launch.json',dict(pid=remainder_proc.pid,time=time.time()))"""
b="""    rempid=read(EXT/'remainder_launch.json')['pid'];remidentity=process(rempid)
    assert remidentity and remidentity['state']!='Z'
    remainder_proc=Adopted(dict(pid=rempid,identity=remidentity))
    save(EXT/'remainder_adopted_v2.json',dict(pid=rempid,identity=remidentity,time=time.time()))"""
assert s.count(a)==1;s=s.replace(a,b)
compile(s,str(EXT/'supervisor_v2.py'),'exec');(EXT/'supervisor_v2.py').write_text(s)
# Main loop, including all admission/publication/resource checks, identical.
start='    failed={};published='
assert s[s.index(start):]==original[original.index(start):]
save(EXT/'release_llama_audit.json',dict(old=sha(EXT/'supervisor.py'),new=sha(EXT/'supervisor_v2.py'),main_loop_identical=True,remainder_not_restarted=True,llama_status_sha256=sha(llama),llama_error=read(llama)['error'],released_gib=9.5,time=time.time()))
print('Prepared same-loop adoption, release9.5GiB retired reservation.')
