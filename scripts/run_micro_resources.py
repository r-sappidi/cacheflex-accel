"""MICRO resource translation: frozen correctness and latency comparisons."""
import concurrent.futures as cf
import configparser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks/cacheflex_flash_attention'))
from run_clean_controls import sha, save, OLD
from single_head_reference import check
from fa_export_patch import verify_dump
from validate_attribution import read
import numpy as np

OUT = ROOT / 'results/cacheflex_flash_attention/micro_resources_20260921'
BUILD = OUT / os.environ.get('CACHEFLEX_MICRO_BUILD', 'build')
EX = OUT / os.environ.get('CACHEFLEX_MICRO_EXPERIMENT', 'experiment')
PREVIOUS = ROOT / 'results/cacheflex_flash_attention/async_staging_20260921/experiment'
ENV = dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

def ini(path):
    c = configparser.ConfigParser(interpolation=None, strict=False); c.read(path)
    return {s: dict(c[s]) for s in c.sections()}

def profile(old, bounded, cap, small=False):
    params = {}
    for i in range(8):
        params[f'system.cpu[{i}].cacheFlexBoundedTracking'] = False
        params[f'system.ruby.l0_cntrl{i}.sequencer.max_outstanding_requests'] = cap
        if bounded:
            params[f'system.cpu[{i}].spmLoadPorts'] = 1
            params[f'system.cpu[{i}].spmStorePorts'] = 1
    return params

def param_assignment(key, value):
    """The -P argument is Python code, not a key/value configuration file."""
    return f'{key}={value!r}'

def audit(old, actual, params, path, guest, args):
    expected = {s: {k: v.replace(str(old), str(path)) for k, v in vals.items()}
                for s, vals in ini(old / 'config.ini').items()}
    for key, value in params.items():
        key = re.sub(r'cpu\[(\d+)\]', r'cpu\1', key)
        sec, k = key.rsplit('.', 1)
        expected[sec][k.lower()] = str(value).lower() if isinstance(value, bool) else str(value)
    expected['system.cpu0.workload']['executable'] = str(guest)
    expected['system.cpu0.workload']['cmd'] = ' '.join(args)
    enabled = any(k.endswith('.spmLoadPorts') for k in params)
    small = '_stress_' in path.name
    pool_caps = {}
    for i in range(8):
        for suffix,n in [('l0_cntrl%d.dataMissCredits'%i,8),
                         ('l0_cntrl%d.instMissCredits'%i,8),
                         ('l0_cntrl%d.writeCredits'%i,8),
                         ('l1_cntrl%d.missCredits'%i,20),
                         ('l1_cntrl%d.writeCredits'%i,8)]:
            pool_caps['system.ruby.'+suffix] = (2 if small else n) if enabled else 0
    pool_caps['system.ruby.microLlcMissCredits'] = (2 if small else 32) if enabled else 0
    pool_caps['system.ruby.microLlcWriteCredits'] = (2 if small else 16) if enabled else 0
    additions = set()
    for sec,n in pool_caps.items():
        assert actual[sec] == {'type':'MicroResourcePool', 'capacity':str(n), 'eventq_index':'0'}, (sec,actual[sec])
        additions.add(sec)
    for i in range(8):
        l0=f'system.ruby.l0_cntrl{i}'; l1=f'system.ruby.l1_cntrl{i}'
        sec=l0+'.microSpmQueue'
        assert actual[sec]['type']=='MessageBuffer' and actual[sec]['buffer_size']=='0'
        additions.add(sec)
        expected[l0].update(datamisscredits=l0+'.dataMissCredits', instmisscredits=l0+'.instMissCredits',
            writecredits=l0+'.writeCredits', downstreammisscredits=l1+'.missCredits',
            downstreamwritecredits=l1+'.writeCredits', microspmqueue=sec,
            microspmtol1=l1+'.spmBufferFromL0', replytol1=l0+'.bufferToL1')
        expected[l1].update(misscredits=l1+'.missCredits', writecredits=l1+'.writeCredits',
                            l0writecredits=l0+'.writeCredits')
        expected[l0+'.sequencer'].update(
            cacheflex_finite_ingress='false', cacheflex_max_line_targets='0',
            cacheflex_spm_ingress=sec if enabled else 'Null',
            cacheflex_micro_data_misses=l0+'.dataMissCredits' if enabled else 'Null',
            cacheflex_micro_inst_misses=l0+'.instMissCredits' if enabled else 'Null',
            cacheflex_micro_writebacks=l0+'.writeCredits' if enabled else 'Null',
            cacheflex_micro_targets='16' if enabled else '0')
    for sec in expected:
        if re.fullmatch(r'system.ruby.l2_cntrl\d+', sec):
            expected[sec].update(misscredits='system.ruby.microLlcMissCredits',
                                 writecredits='system.ruby.microLlcWriteCredits')
    # SimObject child lists may only gain the explicitly validated new objects.
    for sec in expected:
        if 'children' in expected[sec]:
            oldkids=set(expected[sec]['children'].split())
            newkids=set(actual[sec]['children'].split())
            allowed={p.rsplit('.',1)[1] for p in additions if p.rsplit('.',1)[0]==sec}
            assert newkids==oldkids|allowed,(sec,oldkids,newkids,allowed)
            expected[sec]['children']=actual[sec]['children']
    differences={}
    for sec in set(expected)|set(actual):
        if sec in additions: continue
        for k in set(expected.get(sec,{}))|set(actual.get(sec,{})):
            a,b=expected.get(sec,{}).get(k),actual.get(sec,{}).get(k)
            if a!=b: differences[sec+'.'+k]=[a,b]
    assert not differences,differences
    return dict(passed=True, explicit_overrides=params, pools=pool_caps)

def numeric_stats(path):
    epochs = []; current = {}
    for line in path.read_text().splitlines():
        if 'Begin Simulation Statistics' in line: current = {}
        elif 'End Simulation Statistics' in line: epochs.append(current)
        else:
            fields = line.split()
            if len(fields) < 2 or fields[0].startswith(('host', '#')): continue
            try: value = float(fields[1])
            except ValueError: continue
            if np.isfinite(value): current[fields[0]] = value
    return epochs

def main():
    resume = '--resume-wait' in sys.argv
    EX.mkdir(exist_ok=resume); (EX / 'runs').mkdir(exist_ok=resume); (EX / 'preserved').mkdir(exist_ok=resume)
    assert not list((EX / 'runs').iterdir()), 'Refuse to overwrite existing runs'
    state = dict(pid=os.getpid(), stage='waiting_for_build', completed=[], active={})
    lock = threading.RLock(); stop = threading.Event(); results = {}
    def status(stage=None):
        with lock:
            if stage: state['stage'] = stage
            state['updated'] = time.time(); save(EX / 'status.json', state)
    status()
    try:
        while not (BUILD / 'validated_build_ready.json').exists():
            assert not (BUILD / 'status.json').exists() or json.loads((BUILD / 'status.json').read_text())['stage'] != 'failed'
            time.sleep(5)
        status('bridge_roundtrip_and_release_regressions')
        subprocess.run([sys.executable, str(ROOT/'scripts/run_micro_resource_smoke.py'),
            '--build', BUILD.name, '--output', EX.name+'_smoke'], env=ENV, check=True)
        sim = BUILD / 'gem5.opt'; guests = {}; refs = {}
        for arch in ('cache', 'spm'):
            old = PREVIOUS / 'runs' / f'{arch}_t512'
            assert (old / 'accepted.json').exists()
            cmd = json.loads((old / 'command.json').read_text())
            guest = EX / 'preserved' / arch; shutil.copy2(cmd[cmd.index('--cmd')+1], guest)
            guests[arch] = guest
        for t in (512, 4096):
            src = OLD / 'references' / f't{t}_bc128.npy' if t == 512 else ROOT / 'results/cacheflex_flash_attention/contention_optimization_v2_20260920/references/t4096_bc128.npy'
            dst = EX / 'preserved' / src.name; shutil.copy2(src, dst); refs[t] = np.load(dst)
        files = [Path(__file__), sim, *guests.values(), *list((BUILD / 'configs').rglob('*.py')),
                 *list((EX / 'preserved').glob('*.npy'))]
        files += [Path(sys.modules[m].__file__) for m in ('run_clean_controls', 'single_head_reference', 'fa_export_patch', 'validate_attribution')]
        manifest = {str(p): sha(p) for p in files}; save(EX / 'inputs.json', manifest)
        def frozen():
            for p, digest in manifest.items(): assert sha(Path(p)) == digest, ('changed dependency', p)
        def run(arch, t, kind):
            frozen(); bounded = kind != 'default'; small = kind == 'stress'
            cap = 16 if not bounded else 4096 if kind == 'large_guard' else 1024
            old = PREVIOUS / 'runs' / f'{arch}_t512'
            name = f'{arch}_{kind}_t{t}'; p = EX / 'runs' / name; p.mkdir()
            params = profile(ini(old / 'config.ini'), bounded, cap, small)
            cmd = json.loads((old / 'command.json').read_text()); cmd[0] = str(sim)
            cmd[2] = str(BUILD / 'configs/deprecated/example/bounded_se.py')
            if bounded: cmd += ['--cacheflex-micro-resources']
            if small: cmd += ['--cacheflex-micro-small']
            cmd[cmd.index('--cmd')+1] = str(guests[arch])
            args = ['cacheflex-fa', str(t), '64', '32' if arch == 'cache' else '16', '128']
            cmd = [f'--outdir={p}' if a.startswith('--outdir=') else
                   f'system.cpu[0].workload[0].cmd={args!r}' if a.startswith('system.cpu[0].workload[0].cmd=') else a for a in cmd]
            for key, value in params.items(): cmd += ['-P', param_assignment(key, value)]
            save(p / 'command.json', cmd); start = time.monotonic(); proc = None
            try:
                with (p / 'run.log').open('w') as log:
                    proc = subprocess.Popen(cmd, env=ENV, stdout=log, stderr=subprocess.STDOUT)
                    with lock: state['active'][name] = dict(pid=proc.pid, path=str(p)); status()
                    while proc.poll() is None:
                        if stop.is_set(): raise RuntimeError('peer failed')
                        assert time.monotonic()-start < 6*3600, 'timeout'
                        assert shutil.disk_usage(EX).free > 50*1024**3, 'disk guard'
                        try: proc.wait(timeout=5)
                        except subprocess.TimeoutExpired: pass
                assert proc.returncode == 0, (name, proc.returncode)
                config_audit = audit(old, ini(p / 'config.ini'), params, p, guests[arch], args)
                save(p / 'config_audit.json', config_audit)
                output, counters, _ = read(p, 16)
                count = (t//64)*t*2 if arch == 'spm' else 0
                assert counters['m_spmcpRequestsIssued'] == counters['m_spmcpRequestsCompleted'] == count
                row = dict(path=str(p), arch=arch, T=t, kind=kind, output=output, counters=counters,
                           host_seconds=time.monotonic()-start, parameters=params)
                verify_dump(row); row['validation'] = check(p / 'output.fp16', refs[t])
                if not bounded:
                    prev = json.loads((old / 'accepted.json').read_text())
                    assert output == prev['output']
                    assert {k:v for k,v in counters.items() if k!='path'} == {k:v for k,v in prev['counters'].items() if k!='path'}, 'Default drift'
                stats = numeric_stats(p / 'stats.txt')
                assert stats
                if bounded:
                    for epoch in stats:
                        for resource, capacity in config_audit['pools'].items():
                            assert epoch.get(resource+'.maxLive', 0) <= capacity, (resource, capacity)
                        for key, value in epoch.items():
                            if key.endswith('boundedMaxLoadOccupancy'): assert value <= (8 if small else 32), (key, value)
                            if key.endswith('boundedMaxStoreOccupancy'): assert value <= (8 if small else 48), (key, value)
                            if key.endswith('m_globalAdmissionFullRejects'): assert value == 0, (key, value)
                    row['bounds_assertions'] = 'Explicit cache transaction pools enforce capacities; legacy MICRO core tracking retained'
                    row['resources'] = {k:v for k,v in stats[0].items()
                        if any(k.startswith(r+'.') for r in config_audit['pools'])}
                if kind == 'large_guard':
                    base = EX / 'runs' / f'{arch}_bounded_t{t}'
                    assert stats == numeric_stats(base / 'stats.txt'), 'Sequencer guard remains timing-relevant'
                    row['nonbinding_guard'] = 'All finite numeric non-host stats exactly equal at sequencer1024 and4096'
                frozen(); save(p / 'accepted.json', row)
                with lock:
                    results[t,arch,kind] = row
                    state['active'].pop(name)
                    state['completed'].append(dict(name=name, roi_us=counters['roi_us'], path=str(p)))
                    save(EX / 'results.json', list(results.values())); status()
                    lines = ['# MICRO resource translation validation', '', '| Case | ROI us |', '|---|---:|']
                    lines += [f"| {r['name']} | {r['roi_us']:.4f} |" for r in state['completed']]
                    (EX / 'REPORT.md').write_text('\n'.join(lines)+'\n')
            except BaseException as exc:
                stop.set(); save(p / 'failed.json', dict(error=repr(exc)))
                if proc and proc.poll() is None:
                    proc.terminate()
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                raise
        for t, kind in [(512,'default'), (512,'bounded'), (512,'large_guard'), (512,'stress'),
                        (4096,'bounded'), (4096,'large_guard')]:
            status(f't{t}_{kind}')
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run,a,t,kind) for a in ('cache','spm')]
                for future in cf.as_completed(futures): future.result()
        status('completed_requires_analysis')
    except BaseException as exc:
        stop.set(); state['error'] = repr(exc); status('failed'); raise

if __name__ == '__main__': main()
