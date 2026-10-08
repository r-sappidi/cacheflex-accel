"""Gated, fixed-kernel oracle-staging experiment; previous queues untouched."""
import concurrent.futures as cf
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import numpy as np
from build_access_timing import ROOT, sha, save
from run_access_timing import stats
from test_micro_timing import check_global_guard
sys.path.insert(0, str(ROOT/'benchmarks/cacheflex_flash_attention'))
from single_head_reference import check

BASE = ROOT/'results/cacheflex_flash_attention'
OUT = BASE/'ideal_staging_20260923'
BUILD = OUT/os.environ.get('CACHEFLEX_IDEAL_BUILD', 'build_v2')
EX = OUT/os.environ.get('CACHEFLEX_IDEAL_EXPERIMENT', 'experiment_v3')
CAL = BASE/'timing_alignment_20260922'
REF = BASE/'micro_resources_20260921/experiment_v6/preserved'
ENV = dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
for key in list(ENV):
    if key.startswith('CACHEFLEX_AGGREGATE') or key == 'CACHEFLEX_TRACE_LINE': ENV.pop(key)

def prior(arch, t): return CAL/f'fa_v3/{arch}_t{t}'

def command(arch, t, variant, dest):
    cmd = json.loads((prior(arch, t)/'command.json').read_text())
    cmd = [f'--outdir={dest}' if x.startswith('--outdir=') else x for x in cmd]
    cmd[0] = str(BUILD/'gem5.opt')
    cmd[2] = str(EX/'configs/deprecated/example/ideal_staging_se.py')
    assert '--cacheflex-micro-timing' in cmd
    if variant != 'control': cmd.append('--cacheflex-ideal-staging')
    if variant == 'ideal_local':
        for i in range(8):
            cmd += ['-P', f'system.ruby.l1_cntrl{i}.spm_access_response_latency=1']
    return cmd

def audit_configs(old, new, variant):
    a = copy.deepcopy(old); b = copy.deepcopy(new)
    for i in range(8):
        seq = b['system']['ruby'][f'l0_cntrl{i}']['sequencer']
        for field, target in [('cacheflex_ideal_staging_cache', f'system.ruby.l1_cntrl{i}.cache'),
                              ('cacheflex_ideal_staging_controller', f'system.ruby.l1_cntrl{i}')]:
            value = seq.pop(field)
            assert value == (target if variant != 'control' else None), (field, value)
        if variant == 'ideal_local':
            l1 = b['system']['ruby'][f'l1_cntrl{i}']
            assert l1['spm_access_response_latency'] == 1
            l1['spm_access_response_latency'] = a['system']['ruby'][f'l1_cntrl{i}']['spm_access_response_latency']
    assert a == b, 'Unexpected configuration change beyond oracle wiring/local latency'

def counter(s, suffix):
    values = [v for k, v in s.items() if k.endswith('.'+suffix)]
    assert len(values) == 8, (suffix, len(values))
    return sum(values)

def compare_control(old, new):
    extra = {k: v for k, v in new.items() if k not in old}
    assert all('.m_idealStaging' in k and v == 0 for k, v in extra.items()), extra
    assert {k: v for k, v in new.items() if k in old} == old, 'Default changed timed counters'

def report(state):
    lines = ['# Ideal staging diagnostic', '',
        '8-core single-head T4096 VL16, fixed cache/SPM kernels; calibrated timing. '
        'No extra modeled data storage. Oracle copies use functional coherent reads into '
        'already claimed slots, with next-cycle acknowledgements. First claims, unsafe '
        'aliases and unresolved reads use the real protocol. Source cache residency is '
        'unchanged by oracle reads. CPU instructions, translation, LSQ ordering and original '
        'issue bandwidth remain. This is an optimistic diagnostic, not a proposed hardware design '
        'or a universal upper bound for other kernels.', '',
        '| T | Backend | Variant | ROI us | Oracle copies | Real-path copies | vs cache |',
        '|---|---|---|---:|---:|---:|---:|']
    rows = [x for x in state['completed'] if x.get('kind') == 'fa']
    for r in rows:
        c = next((x['roi_us'] for x in rows if x['T'] == r['T'] and x['arch'] == 'cache' and x['variant'] == 'control'), None)
        ratio = f'{c/r["roi_us"]:.6f}x' if c else 'pending'
        lines.append(f'| {r["T"]} | {r["arch"]} | {r["variant"]} | {r["roi_us"]:.4f} | '
                     f'{r["oracle_copies"]:.0f} | {r["real_copies"]:.0f} | {ratio} |')
    lines += ['', 'Only numerically/configuration-validated cases appear. '
              'No percentage of aggregate request latency is treated as a runtime fraction.']
    (EX/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    EX.mkdir(); (EX/'runs').mkdir()
    state = dict(pid=os.getpid(), stage='waiting_for_build', active={}, completed=[])
    lock = threading.RLock(); stop = threading.Event()
    def status(stage=None):
        with lock:
            if stage: state['stage'] = stage
            state['updated'] = time.time(); save(EX/'status.json', state); report(state)
    status()
    manifest = {}
    def frozen():
        for p, h in manifest.items(): assert sha(p) == h, ('Frozen input changed', p)
    def accept(p, row):
        frozen(); save(p/'validated.json', row)
        with lock: state['completed'].append(row); status()
    def execute(p, cmd):
        frozen(); save(p/'command.json', cmd); proc = None; start = time.monotonic()
        try:
            with (p/'run.log').open('w') as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
                with lock: state['active'][p.name] = dict(pid=proc.pid, path=str(p)); status()
                while proc.poll() is None:
                    assert not stop.is_set(), 'Peer failed'
                    assert time.monotonic()-start < 10800, '3-hour diagnostic limit'
                    assert shutil.disk_usage(EX).free > 50*2**30, 'Disk guard'
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired: pass
            assert proc.returncode == 0, (p.name, proc.returncode)
            with lock: state['active'].pop(p.name); status()
            return time.monotonic()-start
        except BaseException:
            stop.set()
            if proc and proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            with lock: state['active'].pop(p.name, None); status()
            raise
    def run(arch, t, variant):
        p = EX/'runs'/f'{arch}_t{t}_{variant}'; p.mkdir(); old = prior(arch,t)
        seconds = execute(p, command(arch,t,variant,p))
        a = json.loads((old/'config.json').read_text().replace(str(old), '<OUT>'))
        b = json.loads((p/'config.json').read_text().replace(str(p), '<OUT>'))
        audit_configs(a,b,variant)
        assert sha(p/'output.fp16') == sha(old/'output.fp16'), 'Full export differs'
        arithmetic = check(p/'output.fp16', np.load(REF/f't{t}_bc128.npy'))
        assert arithmetic['errors'] == arithmetic['nonfinite'] == 0, arithmetic
        h = 1469598103934665603
        for w in np.fromfile(p/'output.fp16', dtype=np.uint16): h = ((h^int(w))*1099511628211)&((1<<64)-1)
        printed = re.search(r'FLASH_RESULT.*?hash=([0-9a-f]+)', (p/'run.log').read_text())[1]
        assert f'{h:016x}' == printed, 'Guest/export mismatch'
        s = stats(p)[0]; check_global_guard(s)
        expected = (t//64)*t*2 if arch == 'spm' else 0
        for suffix in ('m_spmcpRequestsIssued', 'm_spmcpRequestsCompleted'):
            assert counter(s,suffix) == expected, (suffix,counter(s,suffix),expected)
        oracle = counter(s,'m_idealStagingCopies')
        if variant == 'control' or arch == 'cache':
            assert oracle == 0; compare_control(stats(old)[0],s)
        elif arch == 'spm':
            assert oracle > expected*0.5, ('Intervention did not exercise most copies', oracle, expected)
        fallbacks = {k:counter(s,'m_idealStaging'+k) for k in
                     ('FallbackSlot','FallbackAlias','FallbackRead','OrderingRetries')}
        accept(p, dict(kind='fa', arch=arch, T=t, variant=variant, passed=True,
                      roi_us=s['simTicks']/1e6, host_seconds=seconds, arithmetic=arithmetic,
                      output_hash=printed, oracle_copies=oracle, real_copies=expected-oracle,
                      fallbacks=fallbacks, config_audit=True, exact_control_stats=variant=='control' or arch=='cache'))
    def parallel(cases):
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            fs = [pool.submit(run,*case) for case in cases]
            try:
                for f in cf.as_completed(fs): f.result()
            except BaseException: stop.set(); raise
    try:
        while not (BUILD/'frozen.json').exists():
            p = BUILD/'status.json'
            if p.exists(): assert json.loads(p.read_text())['stage'] != 'failed', 'Build failed'
            time.sleep(5)
        shutil.copytree(BUILD/'configs', EX/'configs')
        shutil.copy2(__file__, EX/'runner_snapshot.py')
        manifest.update(json.loads((BUILD/'frozen.json').read_text()))
        paths = [Path(__file__).resolve(), ROOT/'scripts/ideal_staging_se.py',
                 ROOT/'scripts/run_access_timing.py', ROOT/'scripts/test_micro_timing.py',
                 ROOT/'scripts/build_access_timing.py',
                 ROOT/'benchmarks/cacheflex_flash_attention/single_head_reference.py']
        guests = [BASE/'micro_resources_20260921/roundtrip_guest/guest',
                  BASE/'micro_resources_20260921/roundtrip_raw_guest/guest',
                  ROOT/'results/ruby_getx_race_guest_20260916/guest']
        paths += guests + list((EX/'configs').rglob('*.py'))
        for t in (512,4096):
            paths.append(REF/f't{t}_bc128.npy')
            for a in ('cache','spm'):
                paths += [prior(a,t)/n for n in ('command.json','config.json','stats.txt','output.fp16','validated.json')]
                paths.append(CAL/'fa_export_guests'/a)
        manifest.update({str(p):sha(p) for p in paths}); save(EX/'inputs.json',manifest)
        status('default_equivalence_t512')
        parallel([(a,512,'control') for a in ('cache','spm')])
        status('oracle_ordering_regressions')
        for name,g,marker in zip(('roundtrip','roundtrip_raw','release_getx'),guests,
            ('SPM_ROUNDTRIP PASS errors=0','SPM_ROUNDTRIP PASS errors=0','SPM_RELEASE_GETX PASS errors=0')):
            p = EX/'runs'/name; p.mkdir(); cmd = command('spm',512,'ideal',p)
            cmd[cmd.index('--cmd')+1] = str(g)
            cmd = ["system.cpu[0].workload[0].cmd=['spm-smoke']" if x.startswith('system.cpu[0].workload[0].cmd=')
                   else x.replace('sve_vl_se=16','sve_vl_se=4') for x in cmd]
            seconds = execute(p,cmd)
            assert marker in (p/'run.log').read_text(), name
            exercised = sum(counter(s,'m_idealStagingCopies') for s in stats(p))
            if name.startswith('roundtrip'): assert exercised > 0, 'Oracle not exercised by ordering test'
            accept(p,dict(kind='ordering',name=name,passed=True,host_seconds=seconds,oracle_copies=exercised))
        status('oracle_numerical_t512')
        parallel([('spm',512,v) for v in ('ideal','ideal_local')])
        # Wire oracle into a cache-only run as a negative intervention control.
        run('cache',512,'ideal')
        neg = stats(EX/'runs/cache_t512_ideal')[0]
        compare_control(stats(prior('cache',512))[0],neg)
        save(EX/'gates_passed.json',dict(passed=True,checks=8))
        status('controls_t4096')
        parallel([(a,4096,'control') for a in ('cache','spm')])
        status('oracle_t4096')
        parallel([('spm',4096,v) for v in ('ideal','ideal_local')])
        status('completed_validated')
    except BaseException as exc:
        stop.set(); state['error'] = repr(exc); status('failed'); raise

if __name__ == '__main__': main()
