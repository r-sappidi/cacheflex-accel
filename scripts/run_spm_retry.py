"""Capacity-preserving SPM placement test; two simulations maximum.

Related ledger: FA-03/06/29, MOD-03/04. This is per-way physical set
rotation, not extra backing, K+V staging, or a guest-wide set offset.
"""
import concurrent.futures as cf
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import numpy as np
import run_finite_micro as finite
from build_access_timing import sha, save
from fa_headroom_profiles import leaves
from run_access_timing import stats
from test_micro_timing import check_global_guard
from export_safe_elf import export_hash
from reference import reference, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/cacheflex_flash_attention/spm_set_spread_20260927'
BUILD = BASE/os.environ.get('CACHEFLEX_RETRY_BUILD', 'build_retry_v3')
OUT = BASE/os.environ.get('CACHEFLEX_RETRY_EXPERIMENT', 'experiment_retry_v2')
MODEL = ROOT/'results/cacheflex_flash_attention/finite_model_20260926'
GATES = MODEL/'experiment_v5'
SEARCH = MODEL/'analytical_experiment_v5/experiment'


def audit(source, target, stride, retry):
    def fields(p):
        return dict(leaves(json.loads((p/'config.json').read_text().replace(str(p), '<OUT>'))))
    a, b = fields(source), fields(target)
    delta = {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    for k, (old, new) in delta.items():
        if k.endswith('.spm_silent_retry'):
            assert old in (None, False) and new == retry, (k, old, new)
            continue
        if k.endswith('.spm_silent_retry_latency'):
            assert old in (None, 8) and new == 8, (k, old, new)
            continue
        assert k.endswith('.spm_set_stride'), (k, old, new)
        expected = stride if re.fullmatch(r'system.ruby.l1_cntrl\d+.cache.spm_set_stride', k) else 0
        assert new == expected and old in (None, 0), (k, old, new)
    for i in range(8):
        assert b[f'system.ruby.l1_cntrl{i}.cache.spm_set_stride'] == stride
        assert b[f'system.ruby.l1_cntrl{i}.spm_silent_retry'] == retry
    return delta


def legacy_equivalent(source, target):
    """All old numeric nonhost stats exact; only NEW zero event counters allowed."""
    old, new = stats(source), stats(target)
    assert len(old) == len(new)
    for a, b in zip(old, new):
        for k in b.keys() - a.keys():
            assert any(e in k for e in ('Fwd_GETS_SilentRetry', 'SPM_SilentRetry', 'SPM_SilentReissue')), k
            assert b[k] == 0, (k,b[k])
        assert all(b.get(k) == v for k,v in a.items()), [(k,v,b.get(k)) for k,v in a.items() if b.get(k)!=v][:20]
    assert sha(source/'output.fp16') == sha(target/'output.fp16')


def main():
    OUT.mkdir(); (OUT/'runs').mkdir(); (OUT/'references').mkdir()
    state = dict(pid=os.getpid(), stage='waiting_for_build', active={}, completed=[])
    lock = threading.RLock(); stop = threading.Event()
    def status(stage=None):
        with lock:
            if stage: state['stage'] = stage
            state['updated'] = time.time(); save(OUT/'status.json', state)
    status()
    while True:
        if (BUILD/'status.json').exists():
            b = json.loads((BUILD/'status.json').read_text())
            assert b['stage'] != 'failed', b
            if b['stage'] == 'built_requires_validation': break
        time.sleep(5)
    assert sha(BUILD/'gem5.opt') == b['sha256']
    # Supplement the older builder's source list with the new parameter file.
    param = ROOT/'gem5/src/mem/ruby/structures/RubyCache.py'
    dst = BUILD/'sources/mem/ruby/structures/RubyCache.py'
    shutil.copy2(param, dst)
    frozen = json.loads((BUILD/'frozen.json').read_text())
    frozen[str(dst)] = sha(dst)
    save(BUILD/'frozen.json', frozen)
    # Commands retain the existing frozen finite configuration scripts verbatim.
    # Only the simulator and one CacheMemory parameter change.
    refs = {}
    def getref(t, bc):
        key = (t, bc)
        if key not in refs:
            name = f't{t}_bc{bc}_parts1.npy'
            for parent in (SEARCH, GATES):
                p = parent/'references'/name
                if p.exists() and sha(p) == json.loads(p.with_suffix('.json').read_text())['sha256']:
                    refs[key] = np.load(p); frozen[str(p)] = sha(p); break
            else:
                p = OUT/'references'/name
                refs[key] = reference(t, bc, 1); np.save(p, refs[key])
                save(p.with_suffix('.json'), dict(sha256=sha(p),key=[t,bc,1]))
                frozen[str(p)] = sha(p)
    def unchanged():
        for p, h in frozen.items(): assert sha(p) == h, ('changed input', p)
    def run(job):
        name, source, stride, exact, trace, retry, marker = job
        path = OUT/'runs'/name; path.mkdir(); proc = None; start = time.monotonic()
        try:
            assert not stop.is_set()
            cmd = json.loads((source/'command.json').read_text())
            assert Path(cmd[0]) in (MODEL/'build_v3/gem5.opt', BASE/'build/gem5.opt'), cmd[0]
            cmd[0] = str(BUILD/'gem5.opt'); cmd[1] = '--outdir='+str(path)
            for i in range(8): cmd += ['-P', f'system.ruby.l1_cntrl{i}.cache.spm_set_stride={stride}']
            for i in range(8): cmd += ['-P', f'system.ruby.l1_cntrl{i}.spm_silent_retry={retry}']
            if name.startswith('wide_'):
                cmd[cmd.index('--cmd')+1] = str(BASE/'wide_guest/guest')
                cmd[cmd.index('--options')+1] = ''
                cmd = ["system.cpu[0].workload[0].cmd=['spm-set-spread-wide']" if x.startswith('system.cpu[0].workload[0].cmd=') else x for x in cmd]
            if trace:
                old = stats(source)[0]
                begin = int(old['finalTick']-old['simTicks']+old['simTicks']*.25)
                end = begin+5_000_000
                cmd[2:2] = ['--debug-flags=CacheFlexResidency', '--debug-file=residency.gz',
                            f'--debug-start={begin}', f'--debug-end={end}']
                save(path/'trace_window.json', dict(begin=begin,end=end,selection='25% of control ROI, 5us window',
                    limitation='bounded time window, not whole-ROI eviction coverage'))
            save(path/'command.json', cmd)
            unchanged()
            with (path/'run.log').open('w') as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=finite.ENV, stdout=log, stderr=subprocess.STDOUT)
                with lock: state['active'][name] = dict(pid=proc.pid, source=str(source), stride=stride); status()
                while proc.poll() is None:
                    assert not stop.is_set(), 'peer failure'
                    assert time.monotonic()-start < 172800, 'timeout'
                    assert shutil.disk_usage(OUT).free > 50*2**30, 'disk guard'
                    save(path/'progress.json', dict(pid=proc.pid, host_seconds=time.monotonic()-start))
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired: pass
            log = (path/'run.log').read_text()
            assert proc.returncode == 0 and marker in log and 'Simulated exit code not 0!' not in log, log[-3000:]
            assert 'because exiting with last active thread context' in log
            # Wide checks reuse their previously validated wide configurations.
            delta = audit(source, path, stride, retry)
            ss = stats(path)
            for s in ss:
                check_global_guard(s)
                for i in range(8):
                    cpu = json.loads((path/'config.json').read_text())['system']['cpu'][i]
                    for domain in ('regular', 'spm'):
                        for kind, cap in [('Load',cpu['LQEntries']),('Store',cpu['SQEntries'])]:
                            assert s[f'system.cpu{i}.{domain}Lsq0.boundedMax{kind}Occupancy'] <= cap
            row = dict(name=name,source=str(source),stride=stride,config_delta=delta,host_seconds=time.monotonic()-start)
            if marker == 'KV_ROI_END':
                args = cmd[cmd.index('--options')+1].split()
                t,cores,br,bc,group,sets,kv,spm = map(int,args)
                assert cores == 8 and kv == 0
                export_hash(path/'output.fp16', log)
                row['arithmetic'] = validate(path/'output.fp16', refs[t,bc], t, dense=t==1024)
                assert sha(path/'output.fp16') == sha(source/'output.fp16'), 'mapping changed output bits'
                expected = 16*t*math.ceil((t//8)/(br*group)) if spm else 0
                for field in ('m_spmcpRequestsIssued','m_spmcpRequestsCompleted'):
                    assert sum(ss[0][f'system.ruby.l0_cntrl{i}.sequencer.{field}'] for i in range(8)) == expected
                assert ss[0].get('system.ruby.L1Cache_Controller.X.SPM_release::total',0) == (16*bc if spm else 0)
                row.update(roi_us=ss[0]['simTicks']/1e6,copies=expected,output_sha256=sha(path/'output.fp16'))
                if exact: legacy_equivalent(source,path)
                row['silent_retry_received'] = ss[0].get('system.ruby.L1Cache_Controller.SPM_SilentRetry::total',0)
                row['silent_retry_reissued'] = ss[0].get('system.ruby.L1Cache_Controller.SPM_SilentReissue::total',0)
                assert row['silent_retry_received'] == row['silent_retry_reissued']
                if name == 'spread_full_set':
                    assert row['silent_retry_received'] > 0, 'repair path not exercised'
            if trace:
                from analyze_spm_set_spread import analyze
                save(path/'residency_summary.json', analyze(path/'residency.gz'))
            unchanged(); save(path/'validated.json',row)
            with lock: state['completed'].append(row)
            return row
        except BaseException as e:
            stop.set();save(path/'failed.json',dict(error=repr(e)))
            with lock: state.setdefault('error',name+': '+repr(e))
            raise
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill();proc.wait()
            with lock: state['active'].pop(name,None);status()
    def job(name, source, stride=0, exact=False, trace=False, retry=True, marker='KV_ROI_END'):
        return (name,source,stride,exact,trace,retry,marker)
    defaults = [job('default_'+a,GATES/'runs'/f'finite_kv0_spm{s}',exact=True,retry=False) for s,a in enumerate(('cache','spm'))]
    bridges = [job('spread_'+n,GATES/'runs'/n,512,marker=m) for n,m in
               [('roundtrip','SPM_ROUNDTRIP PASS errors=0'),('roundtrip_raw','SPM_ROUNDTRIP PASS errors=0'),
                ('release_getx','SPM_RELEASE_GETX PASS errors=0')]]
    bridges += [job(f'wide_{s}', BASE/f'wide_validation_v2/stride{s}', s,
                    marker='SPM_SET_SPREAD PASS errors=0') for s in (0,512)]
    checks = [job('repaired_cache',GATES/'runs/finite_kv0_spm0'),
              job('repaired_spm',GATES/'runs/finite_kv0_spm1'),
              job('spread_small',GATES/'runs/finite_kv0_spm1',512),
              job('spread_full_set',GATES/'runs/large_kv0_spm1',512),
              job('stress_cache',GATES/'runs/stress_cache'),
              job('spread_stress',GATES/'runs/stress_spm',512),
              job('observer_equivalence',GATES/'runs/finite_kv0_spm1',trace=True)]
    targets = [job('t4096_cache',GATES/'runs/t4096_kv0_spm0'),
               job('t4096_spm_control',GATES/'runs/t4096_kv0_spm1'),
               job('t4096_spread',GATES/'runs/t4096_kv0_spm1',512),
               job('t16384_cache',SEARCH/'runs/t16384_q_cache_br16_bc512_r768',trace=True),
               job('t16384_spm_control',SEARCH/'runs/t16384_q_spm_br16_bc512_r768',trace=True),
               job('t16384_spm_spread',SEARCH/'runs/t16384_q_spm_br16_bc512_r768',512,trace=True)]
    all_jobs = defaults+bridges+checks+targets
    # Freeze all reference/guest/config inputs before any simulator starts.
    for j in all_jobs:
        source=j[1]; cmd=json.loads((source/'command.json').read_text())
        deps=[source/'command.json',source/'config.json',Path(cmd[cmd.index('--cmd')+1])]
        deps += list(Path(cmd[2]).parents[2].rglob('*.py'))
        if j[-1]=='KV_ROI_END':
            args=list(map(int,cmd[cmd.index('--options')+1].split()));getref(args[0],args[3])
            deps += [source/'output.fp16', source/'stats.txt']
        for p in deps: frozen[str(p)]=sha(p)
    for p in [Path(__file__),ROOT/'scripts/analyze_spm_set_spread.py',
              ROOT/'scripts/run_finite_micro.py',ROOT/'scripts/run_access_timing.py',
              ROOT/'benchmarks/cacheflex_flash_attention/kv_partition/reference.py',
              ROOT/'benchmarks/cacheflex_flash_attention/fa_independent_reference.py']:
        frozen[str(p)] = sha(p)
    save(OUT/'inputs.json',frozen)
    save(OUT/'plan.json',dict(ledger=['FA-31','FA-03','FA-06','FA-29','MOD-03','MOD-04'],
        hypothesis='Reduce set-local displacement with fixed occupied SPM bytes',
        primary_metric='matched full FA ROI microseconds; cache remains independently tuned comparator',
        storage='same SRAM and occupied slots; no separate backing',
        timing='unchanged model; per-way address rotation cost not synthesized/bank-calibrated',
        repair='Stale-owner NACK consumed into existing IX_D TBE; bounded timed retry, no added credits/data storage',
        jobs=[dict(name=j[0],source=str(j[1]),stride=j[2],exact=j[3],trace=j[4],retry=j[5]) for j in all_jobs]))
    def batch(jobs,stage):
        status(stage)
        with cf.ThreadPoolExecutor(max_workers=2) as pool: return list(pool.map(run,jobs))
    try:
        batch(defaults,'default_equivalence')
        batch(bridges,'bridge_regressions')
        # Run the original failure before the rest of the gates.
        batch([j for j in checks if j[0]=='spread_full_set'],'original_deadlock_regression')
        batch([j for j in checks if j[0]!='spread_full_set'],'numerical_stress_and_observer_gates')
        finite.equivalent(OUT/'runs/repaired_spm', OUT/'runs/observer_equivalence')
        save(OUT/'gates_passed.json',dict(default_equivalence=True,bridges=True,stress=True,observer_equivalence=True))
        batch(targets[:3],'t4096_matched_comparison')
        batch(targets[3:],'t16384_matched_residency_comparison')
        status('completed_requires_analysis')
    except BaseException as e:
        state.setdefault('error',repr(e));status('failed');raise

if __name__=='__main__': main()
