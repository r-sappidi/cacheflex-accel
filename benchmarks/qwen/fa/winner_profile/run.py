"""FA-53 exact-winner passive replay, with shared admission and full acceptance."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import fcntl,json,re,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from prepare import R,H,OUT,G,CONTROL,B,sha,save
sys.path.insert(0,str(R/'benchmarks/qwen/fa/softmax_tune'))
import tune as T
import numpy as np

NAME='t4096_spm_bc256_r256_f15_cr16_profile'
REF=T.ROOT/'references/t4096_bc256.npy'
TICKET='fa53_winner_profile'

def phase_checks(path,stats,c):
    log=(path/'run.log').read_text();phases={i:{} for i in range(8)};details={i:{} for i in range(8)};pending={}
    for i,p,t,n in re.findall(r'PV_PHASE core=(\d+) phase=(\d+) ticks=(\d+) entries=(\d+)',log):
        i,p=int(i),int(p);assert p not in phases[i];phases[i][p]=dict(ticks=int(t),entries=int(n))
    timeline={int(i):dict(start=int(a),end=int(b),final_arrival=int(f)) for i,a,b,f in re.findall(r'PV_TIMELINE core=(\d+) start=(\d+) end=(\d+) final_arrival=(\d+)',log)}
    assert set(timeline)==set(range(8))
    tasks,updates,copies=T.work(c['t'],c['bc'],c['r'])
    for i,rows in phases.items():
        assert set(rows)==set(range(26)),i
        assert sum(v['ticks'] for v in rows.values())==timeline[i]['end']-timeline[i]['start'],i
        assert timeline[i]['start']<timeline[i]['final_arrival']<=timeline[i]['end'],i
        for phase in (5,6,7,11,22):assert rows[phase]['entries']==updates[i]//16,(i,phase,rows[phase],updates[i])
        for phase in (3,23):assert rows[phase]['entries']==tasks[i],(i,phase,rows[phase],tasks[i])
        for phase in (1,2,12,24):assert rows[phase]['entries']==1,(i,phase,rows[phase])
        assert rows[4]['entries']==copies[i]//(2*c['bc']),(i,rows[4],copies[i])
    for line in log.splitlines():
        if line.startswith(('FA_DETAIL ','FA_PENDING ')):
            v={k:int(n) for k,n in re.findall(r'(\w+)=(\d+)',line)};i=v.pop('core')
            if line.startswith('FA_DETAIL '):
                p=v.pop('phase');assert p not in details[i];details[i][p]=v
            else:assert i not in pending;pending[i]=v
    assert set(pending)==set(range(8));clock=T.read(path/'config.json')['system']['cpu_clk_domain']['clock'][0]
    for i,d in details.items():
        assert set(d)==set(range(26)),i
        totals={k:sum(p[k] for p in d.values()) for k in next(iter(d.values()))}
        assert totals['copy_issue']==totals['copy_done']==copies[i],(i,totals,copies[i])
        assert totals['copy_area']==totals['copy_latency'],i
        assert totals['raw_start']==totals['raw_done']+totals['raw_squash'],i
        assert totals['raw_start']==stats[f'system.cpu{i}.spmLsq0.spmRawBlockedUniqueLoads'],i
        assert totals['raw_latency']/clock==stats[f'system.cpu{i}.spmLsq0.spmRawWaitCycles'],i
        assert totals['spm_port_block']==stats[f'system.cpu{i}.spmLsq0.spmPortBlockedEvents'],i
        assert pending[i]['copies']==pending[i]['raw']==pending[i]['spm']==0,(i,pending[i])
        assert max(pending[i][k] for k in ('copy_max','raw_max','cache_max','spm_max'))<=256,(i,pending[i])
        for kind in ('cache','spm'):
            assert totals[kind+'_issue']==totals[kind+'_done']+pending[i][kind],(i,kind)
        for p,v in d.items():
            for kind in ('copy','raw','cache','spm'):
                assert 0<=v[kind+'_busy']<=phases[i][p]['ticks'],(i,p,kind)
                assert v[kind+'_area']>=v[kind+'_busy'],(i,p,kind)
    return dict(phases=phases,timeline=timeline,phase_details=details,pending_at_end=pending)

def accept(path,c,cmd):
    log=(path/'run.log').read_text()
    assert 'because exiting with last active thread context' in log
    assert not re.search(r'panic:|fatal:|\bFAIL\b',log)
    assert 'SOFTMAX_OPT flags=15' in log and 'BOTTLENECK stage_simple=0 packed_producer=0' in log
    delta=T.config_check(path,c,cmd)
    old=dict(T.leaves(T.read(CONTROL/'config.json')))
    expected={k:v.replace(str(CONTROL),str(path)) if isinstance(v,str) else v for k,v in old.items()}
    assert expected==dict(T.leaves(T.read(path/'config.json'))),'Config mismatch vs exact winner'
    blocks=T.stats(path);baseline=T.stats(CONTROL)
    if blocks!=baseline:
        differences=[]
        for bi,(a,b) in enumerate(zip(blocks,baseline)):
            differences.extend(dict(block=bi,key=k,observed=a.get(k),expected=b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k))
        save(path/'stat_differences.json',dict(observed_blocks=len(blocks),expected_blocks=len(baseline),differences=differences))
        raise AssertionError('Passive replay changed non-host statistics')
    s=blocks[0]
    for block in blocks:
        T.check_global_guard(block)
        for i in range(8):
            for domain in ('regular','spm'):
                for kind,cap in [('Load',32),('Store',48)]:assert block[f'system.cpu{i}.{domain}Lsq0.boundedMax{kind}Occupancy']<=cap
            assert block[f'system.ruby.l0_cntrl{i}.dataMissCredits.maxLive']<=c['credits']
    T.export_hash(path/'output.fp16',log)
    arithmetic=T.validate(path/'output.fp16',np.load(REF),4096,14,2,True,dense=False)
    tasks,updates,copies=T.work(4096,256,256)
    actual={int(i):(int(t),int(u)) for i,t,u in re.findall(r'GQA_WORK core=(\d+) tasks=(\d+) row_updates=(\d+)',log)}
    assert actual=={i:(tasks[i],updates[i]) for i in range(8)},actual
    issued=sum(v for k,v in s.items() if k.endswith('.m_spmcpRequestsIssued'))
    assert issued==sum(copies)==sum(v for k,v in s.items() if k.endswith('.m_spmcpRequestsCompleted'))
    assert sum(v for k,v in s.items() if k.endswith(('.reductionAccepted','.l3ReductionStarts')))==0
    digest=sha(path/'output.fp16');assert digest==T.read(CONTROL/'accepted.json')['output_sha256']==sha(CONTROL/'output.fp16')
    profile=phase_checks(path,s,c)
    return dict(name=NAME,case=c,path=str(path),outcome='full_fa_pass',roi_us=s['simTicks']/1e6,arithmetic=arithmetic,copies=issued,output_sha256=digest,config_delta=delta,observer_equivalence=dict(peer=str(CONTROL),all_nonhost_stats_exact=True,all_output_bits_exact=True,guest_unchanged=True),**profile)

def main():
    lock=(OUT/'coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'Existing queue; do not duplicate'
    c=T.read(CONTROL/'case.json');assert c==T.case(4096,True,256,256,15,16,'fa51c')
    cpus=T.cpus(0);assert set(cpus)<=os.sched_getaffinity(0)
    os.sched_setaffinity(0,cpus);os.nice(10)
    state=dict(pid=os.getpid(),stage='preparing',active={},pending=[NAME],completed=[],failed=[])
    def status(stage=None):
        if stage:state['stage']=stage
        state['updated']=time.time();T.save(OUT/'status.json',state)
    def stop(sig,frame):raise KeyboardInterrupt('termination requested')
    signal.signal(signal.SIGTERM,stop)
    status()
    frozen=T.read(CONTROL.parent.parent/'inputs.json')
    frozen.update(T.read(B/'link_inputs.json'))
    for p in [*H.glob('*.py'),H/'README.md',OUT/'phase_map.txt',OUT/'phase_map_audit.json',OUT/'plan.json',REF,B/'gem5.opt',B/'ready.json',B/'observer.hh',B/'phase_api.hh',B/'hook.hh',B/'pseudo_inst.cc',B/'lsq_unit.cc',B/'Sequencer.cc',*CONTROL.glob('*.json'),CONTROL/'stats.txt',CONTROL/'output.fp16']:
        frozen[str(p)]=sha(p)
    T.save(OUT/'inputs.json',frozen)
    def unchanged():
        for p,h in frozen.items():assert sha(p)==h,('frozen drift',p)
    proc=None;leased=False;path=OUT/'runs'/NAME
    try:
        unchanged();path.mkdir();cmd=T.read(CONTROL/'command.json');cmd[0]=str(B/'gem5.opt');cmd[1]='--outdir='+str(path)
        expected=list(T.read(CONTROL/'command.json'));assert cmd[2:]==expected[2:]
        T.save(path/'command.json',cmd);T.save(path/'case.json',c)
        while True:
            status('waiting_for_reservation')
            with T.broker.locked() as b:
                if time.time()-b['heartbeat']<20 and not b['hold_new'] and not T.pressure() and T.broker.total(b)+10*T.GIB<=b['cap']:
                    assert TICKET not in b['leases'];b['leases'][TICKET]=dict(bytes=10*T.GIB,owner=os.getpid(),family='FA-53');leased=True;break
            time.sleep(5)
        limit=8*T.GIB
        def child():resource.setrlimit(resource.RLIMIT_AS,(limit,resource.RLIM_INFINITY))
        env=dict(T.ENV,CACHEFLEX_PV_PHASE_MAP=str(OUT/'phase_map.txt'))
        for key in list(env):
            if key.startswith('CACHEFLEX_') and key!='CACHEFLEX_PV_PHASE_MAP':del env[key]
        T.save(path/'environment.json',{k:v for k,v in env.items() if k.startswith('CACHEFLEX_') or k=='LD_LIBRARY_PATH'})
        with (path/'run.log').open('w') as log:proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=child)
        start=last=time.monotonic();sig=None;paused=False
        state['active']={NAME:dict(pid=proc.pid,cpus=cpus,started=time.time(),paused=False,simulator_as=limit)};status('running')
        while proc.poll() is None:
            assert time.monotonic()-start<6*3600,'six-hour host timeout'
            with (path/'run.log').open('rb') as f:
                f.seek(max(0,(path/'run.log').stat().st_size-16384));tail=f.read().decode(errors='replace')
            now=tuple(sorted(re.findall(r'system.cpu(\d+) progress event, total committed:(\d+)',tail)))
            if paused or now!=sig:last=time.monotonic();sig=now
            assert time.monotonic()-last<900,'no committed-instruction progress for 900s'
            hold=T.pressure();info=T.owned().get(proc.pid)
            if info and info['vms']>=limit-T.GIB:
                with T.broker.locked() as b:
                    if not hold and T.broker.total(b)+4*T.GIB<=b['cap']:
                        b['leases'][TICKET]['bytes']+=4*T.GIB;limit+=4*T.GIB;resource.prlimit(proc.pid,resource.RLIMIT_AS,(limit,resource.RLIM_INFINITY))
                    else:hold=True
            if hold!=paused:proc.send_signal(signal.SIGSTOP if hold else signal.SIGCONT);paused=hold
            state['active'][NAME].update(paused=paused,simulator_as=limit)
            state['guest_progress']=dict(retired=dict(now),last_change=time.time()-(time.monotonic()-last));status()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
        assert proc.returncode==0,('simulator exit',proc.returncode)
        state['active']={};status('validating');row=accept(path,c,cmd);row['host_seconds']=time.monotonic()-start
        unchanged();T.save(path/'accepted.json',row);state['completed']=[row];state['pending']=[]
        import analyze
        analyze.main()
        status('complete')
        with (R/'docs/experiments/LEDGER.md').open('a') as f:f.write('\n**FA-53 complete:** exact FA-51c Qwen SPM winner replay accepted; all non-host stats and output bits match. Full oracle/config/work/copy/resource/frozen-input and phase checks pass. See results/cacheflex_flash_attention/winner_profile_20261007/REPORT.md and FINAL_VERIFICATION.json.\n')
    except BaseException as e:
        state['failed'].append(dict(name=NAME,error=repr(e),traceback=traceback.format_exc()));status('failed')
        with (R/'docs/experiments/LEDGER.md').open('a') as f:f.write('\n**FA-53 failed/unaccepted:** '+repr(e)+'; see winner_profile_20261007/status.json. No phase performance claim.\n')
        raise
    finally:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGCONT);proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        if leased:T.broker.release(TICKET)
        state['active']={};status()
if __name__=='__main__':main()
