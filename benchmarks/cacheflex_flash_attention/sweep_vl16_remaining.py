"""Dependency-ordered, correctness-gated software search. Four simulations max.

Starts grouping, then placement/buffering, then native assembly scheduling,
then token-count validation of independently selected cache/SPM finalists.
"""
import concurrent.futures as cf
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading

from validate_attribution import read

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent

def main():
    out=Path(sys.argv[1]); assert out.is_absolute(); out.mkdir()
    (out/'builds').mkdir(); (out/'runs').mkdir()
    old=ROOT/'results/cacheflex_flash_attention/clean_vl16_20260908/runs'
    shutil.copy2(old/'gem5.opt',out/'gem5.opt')
    shutil.copy2(old/'guests/cache',out/'cache_original')
    watched=[HERE/'spm_multicore.cpp',HERE/'cache_qk_wrapper.cpp',
             HERE/'make_search_kernels.py',HERE/'build_vl16_search.sh',HERE/'run_8core.sh',
             ROOT/'gem5/configs/ruby/MESI_Three_Level.py']
    watched += list((ROOT.parent/'cacheflex_micro/experiments/end2end/src').glob('*.hpp'))
    watched += [ROOT.parent/'cacheflex_micro/experiments/end2end/src/cache_flash_attn.cpp',
                ROOT.parent/'cacheflex_micro/kernels/attention_kernels_spm/11_flash_attn_kv_spm_vl4.cpp']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    for p in watched[:5]: shutil.copy2(p,out/p.name)
    lock=threading.Lock(); rows=[]; errors=[]; skipped=[]; events=[]; builds={}
    references={}
    original=dict(id='original',placement='cache',group=0,overlap=0,u=1,order=0,
                  guest=str(out/'cache_original'))
    advanced=len(sys.argv)>2 and sys.argv[2]=='advanced'
    rich_reference=None

    def save():
        (out/'progress.json').write_text(json.dumps(dict(events=events,completed=rows,errors=errors,skipped=skipped),indent=2))

    def event(s):
        with lock:
            events.append(s); save(); print(s,flush=True)

    def checked_process(cmd,env,log,timeout):
        with log.open('w') as stream:
            p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try: code=p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try: p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL); p.wait()
                raise RuntimeError('timeout '+str(log))
            if code: raise RuntimeError(f'exit {code}: {log}')

    def config_check(cfg,ref):
        assert set(cfg.sections())==set(ref.sections())
        for section in cfg.sections():
            assert set(cfg[section])==set(ref[section])
            for key,x in cfg[section].items():
                y=ref[section][key]
                if x==y or (section=='system.cpu0.workload' and key in ('cmd','executable')):
                    continue
                assert key=='num_spm_ways' and x=='4' and y=='0',(section,key,x,y)

    def simulate(desc,t,br,bc,stage,reference=None):
        banks=1+desc['overlap']
        k=desc['placement'] in ('k','kv'); v=desc['placement'] in ('v','kv')
        sets=banks*((bc//128)*64*k+bc*v)
        if sets>1024:
            with lock: skipped.append(dict(stage=stage,variant=desc['id'],br=br,bc=bc,reason='SPM set capacity',sets=sets));save()
            return None
        name=f'{stage}_{desc["id"]}_t{t}_br{br}_bc{bc}'
        path=out/'runs'/name
        env=dict(PATH=os.environ['PATH'],HOME=os.environ['HOME'],CACHEFLEX_FLASH_VL='16',
                 CACHEFLEX_FLASH_MATCHED_LRU='1',CACHEFLEX_FLASH_SPM_WAYS='4' if k or v else '0',
                 CACHEFLEX_FLASH_BIN=desc['guest'],CACHEFLEX_GEM5_BIN=str(out/'gem5.opt'),CACHEFLEX_FLASH_OUT=str(path))
        checked_process(['bash',str(HERE/'run_8core.sh'),'spm_konly_overlap' if k or v else 'cache_pretouch',str(t),str(br),str(bc)],
                        env,out/(name+'.host.log'),7200)
        output,counters,cfg=read(path,expected_vl=16)
        for key,val in dict(T=t,Br=br,Bc=bc,d=64,Hq=8,Hkv=1,cores=8).items(): assert output[key]==str(val),(key,output[key],val)
        n=math.ceil(t/br); groups=math.ceil(n/desc['group']) if desc['group'] else 1
        expected=groups*8*(t//128*64*4*k+t*2*v)
        assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==expected,(name,expected,counters['m_spmcpRequestsCompleted'])
        for core in range(8):
            assert cfg[f'system.cpu{core}.isa']['sve_vl_se']=='16'
            assert cfg[f'system.ruby.l1_cntrl{core}.cache.replacement_policy']['type']=='LRURP'
        if reference:
            assert output['hash']==reference['output']['hash'],(name,'hash',output['hash'],reference['output']['hash'])
            _,_,refcfg=read(Path(reference['path']),expected_vl=16)
            config_check(cfg,refcfg)
        row=dict(stage=stage,desc=desc,t=t,br=br,bc=bc,path=str(path),output=output,counters=counters)
        with lock: rows.append(row);save();print(name,counters['roi_us'],'PASS',flush=True)
        return row

    def reference(t,bc):
        key=(t,bc)
        if key not in references:
            references[key]=simulate(original,t,32,bc,'reference')
        return references[key]

    def build(placement,group,overlap,u=1,order=0,wide=0,pipeline=0,fusion=0):
        nonlocal rich_reference
        name=f'{placement}_g{group}_b{1+overlap}_u{u}_o{order}'
        if advanced: name+=f'_r{wide}_p{pipeline}_f{fusion}'
        if name in builds: return builds[name]
        for p,h in hashes.items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,('source changed',p)
        target=out/'builds'/name
        try:
            args=[placement,str(group),str(overlap),str(u),str(order),str(wide),str(pipeline),str(fusion)]
            checked_process(['bash',str(HERE/'build_vl16_search.sh'),str(target)]+args,
                            os.environ.copy(),out/(name+'.build.log'),300)
            desc=dict(id=name,placement=placement,group=group,overlap=overlap,u=u,order=order,wide=wide,pipeline=pipeline,fusion=fusion,guest=str(target/'guest'))
            simulate(desc,256,32,128,'smoke',reference(256,128))
            if advanced:
                if rich_reference is None:
                    rt=out/'builds/rich_reference'
                    checked_process(['bash',str(HERE/'build_vl16_search.sh'),str(rt),'cache','1','0','1','0','0','0','0','1'],
                                    os.environ.copy(),out/'rich_reference.build.log',300)
                    rd=dict(original,id='rich_reference',group=1,guest=str(rt/'guest'))
                    rich_reference=simulate(rd,256,32,128,'rich_reference')
                rt=out/'builds'/(name+'_rich')
                checked_process(['bash',str(HERE/'build_vl16_search.sh'),str(rt)]+args+['1'],
                                os.environ.copy(),out/(name+'.rich.build.log'),300)
                simulate(dict(desc,id=name+'_rich',guest=str(rt/'guest')),256,32,128,'rich_smoke',rich_reference)
            builds[name]=desc
        except Exception as e:
            with lock: errors.append(dict(stage='build/smoke',variant=name,error=repr(e)));save()
            builds[name]=None
        return builds[name]

    def batch(jobs,stage):
        # All references/builds prepared before launching worker threads.
        for _,_,bc in jobs: reference(1024,bc)
        accepted=[]
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(simulate,d,1024,br,bc,stage,references[(1024,bc)]):(d,br,bc) for d,br,bc in jobs if d}
            for f in cf.as_completed(futures):
                try:
                    row=f.result()
                    if row: accepted.append(row)
                except Exception as e:
                    d,br,bc=futures[f]
                    with lock: errors.append(dict(stage=stage,variant=d['id'],br=br,bc=bc,error=repr(e)));save()
        return accepted

    def best(candidates,arch,n=1):
        selected=[r for r in candidates if (r['desc']['placement']=='cache')==(arch=='cache')]
        return sorted(selected,key=lambda r:r['counters']['roi_us'])[:n]

    if advanced:
        event('ADVANCED RETILING: original cache, grouped cache and K+V at Br16/32/64/128 x Bc128/256/512, T1024')
        dc=build('cache',4,0,2,0); ds=build('kv',4,0,4,1)
        assert dc and ds,'Retiling seeds failed correctness'
        jobs=[(d,br,bc) for d in (original,dc,ds) for br in (16,32,64,128) for bc in (128,256,512)]
        candidates=batch(jobs,'advanced_tiles')
        event('ADVANCED KERNELS: top grouped-cache/K+V layout, QK rows8/16/24 x pipeline off/on x rescale fusion off/on')
        jobs=[]
        for arch in ('cache','spm'):
            seed=best([r for r in candidates if r['desc']['id']!='original'],arch)[0]
            d=seed['desc']
            for wide in (8,16,24):
                for pipeline in (0,1):
                    for fusion in (0,1):
                        variant=build(d['placement'],d['group'],d['overlap'],d['u'],d['order'],wide,pipeline,fusion)
                        jobs.append((variant,seed['br'],seed['bc']))
        candidates+=batch(jobs,'advanced_kernels')
        event('PER-SIZE RETUNING: shortlist of top two distinct cache/SPM candidates plus original cache controls at T128/256/4096')
        shortlist=[]
        for arch in ('cache','spm'):
            seen=set()
            for row in best(candidates,arch,len(candidates)):
                key=(row['desc']['id'],row['br'],row['bc'])
                if key not in seen:
                    shortlist.append(row);seen.add(key)
                if len(seen)==2: break
        shortlist += [r for r in candidates if r['desc']['id']=='original' and (r['br'],r['bc']) in ((64,256),(32,512))]
        shortlist=list({(r['desc']['id'],r['br'],r['bc']):r for r in shortlist}.values())
        finals=[]
        for t in (128,256,4096):
            for r in shortlist: reference(t,r['bc'])
            with cf.ThreadPoolExecutor(max_workers=4) as pool:
                futures={pool.submit(simulate,r['desc'],t,r['br'],r['bc'],'retune',references[(t,r['bc'])]):r for r in shortlist}
                for f in cf.as_completed(futures):
                    try: finals.append(f.result())
                    except Exception as e:
                        with lock: errors.append(dict(stage='retune',t=t,error=repr(e)));save()
        results={}
        for t in (128,256,1024,4096):
            eligible=candidates if t==1024 else [r for r in finals if r and r['t']==t]
            a=best(eligible,'cache')[0];b=best(eligible,'spm')[0]
            results[t]=dict(cache=a,spm=b,speedup=a['counters']['roi_us']/b['counters']['roi_us'])
        (out/'summary.json').write_text(json.dumps(dict(results=results,errors=errors,skipped=skipped,
            caveat='Bounded per-size shortlist retuning. Rich-input smoke comparison is against cache kernels, not independent numerical reference. Larger/pipelined QK only; PV remains 8-row. Fusion combines output scaling and accumulation, not direct softmax-to-packed-P.'),indent=2))
        event('COMPLETE: advanced summary.json written')
        return

    event('GROUPING: cache and K-only, groups 1/2/4/all at both tile-sweep winning layouts')
    candidates=[]
    for br,bc in ((64,256),(32,512)):
        candidates.append(simulate(original,1024,br,bc,'original_control',reference(1024,bc)))
    jobs=[]
    for placement in ('cache','k'):
        for group in (1,2,4,0):
            d=build(placement,group,int(placement=='k'))
            for br,bc in ((64,256),(32,512)): jobs.append((d,br,bc))
    candidates+=batch(jobs,'grouping')

    event('PLACEMENT/BUFFERING: K, V, K+V; single/double; top two SPM grouping/layout candidates')
    seeds=best(candidates,'spm',2); assert seeds,'No validated SPM grouping candidates'
    jobs=[]; seen=set()
    for seed in seeds:
        for p in ('k','v','kv'):
            for overlap in (0,1):
                key=(p,seed['desc']['group'],overlap,seed['br'],seed['bc'])
                if key in seen: continue
                seen.add(key)
                sets=(1+overlap)*((seed['bc']//128)*64*(p in ('k','kv'))+seed['bc']*(p in ('v','kv')))
                if sets>1024:
                    with lock: skipped.append(dict(stage='placement',key=key,reason='SPM set capacity',sets=sets));save()
                    continue
                jobs.append((build(p,seed['desc']['group'],overlap),seed['br'],seed['bc']))
    candidates+=batch(jobs,'placement')

    event('SCHEDULING: top cache/SPM candidates; K unroll 1/2/4; SPM-QK A-load-first alternative')
    jobs=[]
    for arch in ('cache','spm'):
        # Keep original baseline as a candidate; tune a grouped-driver candidate too.
        seeds=best([r for r in candidates if r['desc']['id']!='original'],arch,1)
        for seed in seeds:
            d=seed['desc']
            for u in (1,2,4):
                for order in ((0,1) if d['placement'] in ('k','kv') else (0,)):
                    jobs.append((build(d['placement'],d['group'],d['overlap'],u,order),seed['br'],seed['bc']))
    candidates+=batch(jobs,'scheduling')
    winners={arch:best(candidates,arch)[0] for arch in ('cache','spm')}
    (out/'winners_t1024.json').write_text(json.dumps(winners,indent=2))

    event('FINALISTS: fixed independently selected settings at T128/256/1024/4096')
    final=[]
    for t in (128,256,1024,4096):
        for winner in winners.values(): reference(t,winner['bc'])
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(simulate,w['desc'],t,w['br'],w['bc'],'finalist',references[(t,w['bc'])]) for w in winners.values()]
            for f in futures: final.append(f.result())
    summary=dict(winners=winners,finalists=final,errors=errors,skipped=skipped,
                 speedups={t:next(r['counters']['roi_us'] for r in final if r['t']==t and r['desc']['placement']=='cache')/
                              next(r['counters']['roi_us'] for r in final if r['t']==t and r['desc']['placement']!='cache') for t in (128,256,1024,4096)},
                 caveat='Finite matching reference hashes on existing periodic inputs, not independent numerical validation. Bounded staged search, not exhaustive Cartesian product.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));event('COMPLETE: summary.json written')

if __name__=='__main__': main()
