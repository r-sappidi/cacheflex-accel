"""Bounded, independently checked reconstruction of MICRO's VL16 FA sweep.

Corrected GCC15 guests, unchanged placement/timers, exact upstream Classic
options. Original published winning artifacts are unavailable: not a claim of
byte-identical paper reproduction. Never edits existing experiments.
"""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import numpy as np
from run_micro_fa_ruby_pilot import ROOT, save, sha, gemm
from revalidate_gcc15_export import epochs

B = ROOT / 'results/cacheflex_flash_attention'
UP = B / 'micro_upstream_20260918'
OLD = B / 'micro_gcc15_export_revalidation_v2_20260919'
OUT = B / 'micro_paper_reproduction_v2_20260921'
ENV = dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
TILES = [(64, 384), (32, 384), (64, 256), (32, 256), (64, 128), (32, 128)]
FUS = {0: 2, 2: 2, 5: 2, 7: 2, 9: 2}
stop = threading.Event()
ref_lock = threading.Lock()


def upstream_options():
    script = (UP / 'experiments/end2end/scripts/sweep_vl16.sh').read_text()
    return shlex.split(re.search(r'GEM5_CONFIG="(.*?)"', script, re.S)[1].replace('\\\n', ' '))


def flatten(value, prefix=''):
    if isinstance(value, dict):
        return {k: v for key, item in value.items()
                for k, v in flatten(item, f'{prefix}.{key}' if prefix else key).items()}
    if isinstance(value, list):
        return {k: v for i, item in enumerate(value)
                for k, v in flatten(item, f'{prefix}[{i}]').items()}
    return {prefix: value}


def config_audit(actual, original, paper, argv, guest, outdir=None):
    expected = json.loads(json.dumps(original))
    cpu = expected['system']['cpu'][0]
    if paper:
        for index, count in FUS.items():
            cpu['fuPool']['FUList'][index]['count'] = count
    cpu['workload'][0]['cmd'] = argv
    cpu['workload'][0]['executable'] = str(guest)
    if outdir is not None:
        for redirect in expected['system']['redirect_paths']:
            for i, path in enumerate(redirect['host_paths']):
                host_path = Path(path)
                if host_path.parent.name == 'fs' and host_path.name in ('proc', 'sys', 'tmp'):
                    redirect['host_paths'][i] = str(outdir/'fs'/host_path.name)
    a, b = flatten(actual), flatten(expected)
    differences = {k: [b.get(k), a.get(k)] for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert not differences, ('unexpected full-config drift', differences)
    return dict(passed=True, fields=len(a), allowed_changes='Five FU counts only, plus executable path, explicit workload dimensions and output-local fs redirects')


def input_arrays(p, t, hq, hkv):
    log = (p / 'run.log').read_text(errors='replace')
    arrays, hashes = {}, {}
    for name, heads in [('Q', hq), ('K', hkv), ('V', hkv), ('O', hq)]:
        f = p / f'{name}.fp16'
        raw = np.fromfile(f, dtype=np.uint16)
        assert raw.size == heads * t * 64
        h = 1469598103934665603
        for x in raw:
            h = ((h ^ int(x)) * 1099511628211) & ((1 << 64) - 1)
        assert f'EXPORT {name}.fp16 bytes={raw.size*2} hash={h:016x}' in log
        arrays[name] = raw.view(np.float16).reshape(heads, t, 64)
        assert np.isfinite(arrays[name]).all()
        hashes[name] = sha(f)
    return arrays, hashes


def reference(arrays, bc, progress):
    q, k, v = (arrays[n] for n in ('Q', 'K', 'V'))
    hq, t, d = q.shape
    expected = np.empty_like(q)
    for h in range(hq):
        for i in range(0, t, 32):
            qq = q[h, i:i+32]; n = len(qq)
            m = np.full(n, -1e30, np.float32); l = np.zeros(n, np.float32)
            acc = np.zeros((n, d), np.float32)
            for j in range(0, t, bc):
                s = gemm(qq, k[h // (hq // len(k)), j:j+bc].T).astype(np.float32) * np.float32(.125)
                mn = np.maximum(m, s.max(axis=1))
                alpha = np.exp((m - mn).astype(np.float64)).astype(np.float32)
                z = np.exp((s - mn[:, None]).astype(np.float64)).astype(np.float32)
                l = (l * alpha + z.sum(axis=1, dtype=np.float32)).astype(np.float32)
                acc = (acc * alpha[:, None] + gemm(z.astype(np.float16), v[h // (hq // len(v)), j:j+bc]).astype(np.float32)).astype(np.float32)
                m = mn
            expected[h, i:i+32] = (acc / l[:, None]).astype(np.float16)
        save(progress, dict(heads_checked=h+1, heads_total=hq))
    return expected


def numerical(p, t, hq, hkv, bc):
    arrays, hashes = input_arrays(p, t, hq, hkv)
    effective = int(re.search(r'\bBc=(\d+)', (p/'run.log').read_text(errors='replace'))[1])
    assert effective == bc
    key_data = dict(T=t, Hq=hq, Hkv=hkv, Bc=bc, inputs={n: hashes[n] for n in ('Q', 'K', 'V')},
                    oracle=sha(Path(__file__)), gemm=sha(ROOT/'benchmarks/cacheflex_flash_attention/fa_independent_reference.py'))
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    refdir = OUT / 'references' / key
    # Serialize shared reference creation; all elements still checked per run.
    with ref_lock:
        if not (refdir / 'accepted.json').exists():
            refdir.mkdir(exist_ok=True)
            expected = reference(arrays, bc, p/'reference_progress.json')
            expected.tofile(refdir/'expected.fp16')
            save(refdir/'accepted.json', dict(**key_data, sha256=sha(refdir/'expected.fp16')))
        metadata = json.loads((refdir/'accepted.json').read_text())
        assert sha(refdir/'expected.fp16') == metadata['sha256']
        expected = np.fromfile(refdir/'expected.fp16', dtype=np.float16).reshape(hq, t, 64)
    error = np.abs(arrays['O'].astype(np.float64) - expected.astype(np.float64))
    result = dict(elements=expected.size, errors=int((error > 2e-6 + .01*np.abs(expected.astype(np.float64))).sum()),
                  nonfinite=int((~np.isfinite(arrays['O'])).sum()), max_abs=float(error.max()),
                  reference=str(refdir), hashes=hashes, tolerance='2e-6 + 1% abs(reference)')
    assert result['errors'] == result['nonfinite'] == 0, result
    if t < 4096:
        # Also check the intended dense attention, independently of tile recurrence.
        err2 = norm2 = 0.
        for h in range(hq):
            score = arrays['Q'][h].astype(np.float64) @ arrays['K'][h//(hq//hkv)].astype(np.float64).T * .125
            prob = np.exp(score-score.max(axis=1, keepdims=True))
            ideal = (prob/prob.sum(axis=1, keepdims=True)) @ arrays['V'][h//(hq//hkv)].astype(np.float64)
            err2 += float(np.square(arrays['O'][h].astype(np.float64)-ideal).sum())
            norm2 += float(np.square(ideal).sum())
        result['dense_math_relative_l2'] = (err2/max(norm2, 1e-30))**.5
        assert result['dense_math_relative_l2'] < .01, result
    save(p/'numerical.json', result)
    return result


def main():
    OUT.mkdir(); (OUT/'preserved').mkdir(); (OUT/'runs').mkdir(); (OUT/'references').mkdir()
    state = dict(pid=os.getpid(), completed=[], active={}, max_simulators=2)
    state_lock = threading.RLock()
    def status(stage=None):
        with state_lock:
            if stage: state['stage'] = stage
            state['updated'] = time.time(); save(OUT/'status.json', state)
    assert subprocess.check_output(['git', '-C', str(UP), 'rev-parse', 'HEAD'], text=True).strip() == '82ac0b9de1ed408ce5cd53067267c8a855b4eef3'
    status('preserving')
    originals = {}; templates = {}; sims = {}; guests = {}; configs = {}
    for model in ('classic', 'ruby'):
        for arch in ('cache', 'spm'):
            old = OLD/'runs'/f'{model}_{arch}_vl16_gate'
            originals[model, arch] = json.loads((old/'config.json').read_text())
            templates[model, arch] = json.loads((old/'command.json').read_text())
        cmd = templates[model, 'cache']
        sims[model] = OUT/'preserved'/f'{model}.opt'
        shutil.copy2(cmd[0], sims[model])
        configs[model] = OUT/'preserved'/f'{model}_configs'
        shutil.copytree((UP/'gem5' if model=='classic' else ROOT/'gem5')/'configs', configs[model], ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for arch in ('cache', 'spm'):
        guests[arch] = OUT/'preserved'/f'{arch}_vl16'
        shutil.copy2(OLD/'builds'/f'{arch}_vl16', guests[arch])
        shutil.copy2(OLD/'builds'/f'{arch}_vl16.audit.json', OUT/'preserved'/f'{arch}_export.audit.json')
    shutil.copy2(UP/'experiments/end2end/scripts/sweep_vl16.sh', OUT/'preserved/sweep_vl16.sh')
    shutil.copy2(Path(__file__), OUT/'preserved/runner.py')
    helpers = [ROOT/'scripts/run_micro_fa_ruby_pilot.py', ROOT/'scripts/revalidate_gcc15_export.py',
               ROOT/'scripts/run_micro_fa_ruby_target.py', ROOT/'benchmarks/cacheflex_flash_attention/run_clean_controls.py',
               ROOT/'benchmarks/cacheflex_flash_attention/fa_independent_reference.py']
    manifest = {str(p): sha(p) for p in list((OUT/'preserved').rglob('*'))+helpers+[Path(__file__)] if p.is_file()}
    save(OUT/'manifest.json', manifest)
    save(OUT/'plan.json', dict(T=4096, Hq=32, Hkv=8, d=64, cores=1, VL=16, tiles=TILES,
        target_speedup=1.34, max_simulators=2, original_revision='82ac0b9de1ed408ce5cd53067267c8a855b4eef3',
        precision='FP16 operands and GEMM accumulators; FP32 softmax/running output; FP16 stored output',
        stages=['4 exact-counter T320 common-core controls', '24 T832 Hq4/Hkv1 gates (six tiles, two backends, both models)',
                'Classic fixed Br64/Bc384 pair with upstream FU counts', 'Ruby same fixed pair',
                'Remaining five Classic tile pairs', 'Transfer independent Classic winners to Ruby (reuse if already measured)'],
        roi='First stats ROI; unchanged upstream phase timers and first-call convention',
        limitations=['Corrected dense FP16 conversions, not untouched paper ELF', 'Published winning manifests unavailable',
                     'Classic prefetch/interconnect intentionally retained; Ruby memory model remains distinct',
                     'SPM VL16 stages K and V but upstream PV still reads coherent V; unchanged here']))

    def run(model, arch, br, bc, gate=False, control=False):
        if stop.is_set(): raise RuntimeError('peer failed')
        t, hq, hkv = (320, 1, 1) if control else ((832, 4, 1) if gate else (4096, 32, 8))
        label = f'{model}_{arch}_'+('control' if control else 'gate' if gate else 'target')+f'_br{br}_bc{bc}'
        p = OUT/'runs'/label
        if (p/'accepted.json').exists(): return json.loads((p/'accepted.json').read_text())
        p.mkdir()
        args = [str(x) for x in (t, hq, hkv, 64, 1, br, bc)]
        argv = [originals[model, arch]['system']['cpu'][0]['workload'][0]['cmd'][0]] + args
        if model=='classic' and not control:
            cmd = [str(sims[model]), f'--outdir={p}', str(configs[model]/'deprecated/example/se.py'),
                   *upstream_options(), '--num-cpus=1', '--prog-interval=100kHz', '-c', str(guests[arch]), '-o', ' '.join(args),
                   '-P', f'system.cpu[0].workload[0].cmd={argv!r}']
        else:
            cmd = list(templates[model, arch]); cmd[0] = str(sims[model]); cmd[2] = str(configs[model]/'deprecated/example/se.py')
            cmd = [f'--outdir={p}' if x.startswith('--outdir=') else f'system.cpu[0].workload[0].cmd={argv!r}'
                   if x.startswith('system.cpu[0].workload[0].cmd=') else x for x in cmd]
            cmd[cmd.index('-c' if model=='classic' else '--cmd')+1] = str(guests[arch])
            if model=='classic': cmd[cmd.index('-o')+1] = ' '.join(args)
            if not control:
                for index, count in FUS.items(): cmd += ['-P', f'system.cpu[0].fuPool.FUList[{index}].count={count}']
        if gate or control: cmd += ['--abs-max-tick=200000000000']
        deps = [sims[model], guests[arch], Path(__file__), *configs[model].rglob('*.py'), *helpers]
        assert all(sha(f)==manifest[str(f)] for f in deps)
        save(p/'command.json', cmd)
        start = time.monotonic()
        try:
            with (p/'run.log').open('w') as log:
                child = subprocess.Popen(cmd, env=ENV, stdout=log, stderr=subprocess.STDOUT)
                with state_lock: state['active'][label] = dict(pid=child.pid, path=str(p)); status()
                audited = False
                try:
                    while child.poll() is None:
                        save(p/'progress.json', dict(stage='simulating', pid=child.pid, host_elapsed_s=time.monotonic()-start))
                        if stop.is_set(): raise RuntimeError('peer failed')
                        if shutil.disk_usage(OUT).free < 50*1024**3: raise RuntimeError('disk guard')
                        if time.monotonic()-start > (3600 if gate or control else 48*3600): raise RuntimeError('timeout')
                        if (p/'config.json').exists() and not audited:
                            try: actual = json.loads((p/'config.json').read_text())
                            except json.JSONDecodeError: pass
                            else:
                                save(p/'config_audit.json', config_audit(actual, originals[model, arch], not control, argv, guests[arch], p)); audited=True
                        try: child.wait(timeout=5)
                        except subprocess.TimeoutExpired: pass
                finally:
                    if child.poll() is None: child.terminate(); child.wait(timeout=30)
            save(p/'exit.json', dict(returncode=child.returncode, host_elapsed_s=time.monotonic()-start))
            assert child.returncode == 0, (label, child.returncode)
            save(p/'config_audit.json', config_audit(json.loads((p/'config.json').read_text()), originals[model, arch], not control, argv, guests[arch], p))
            assert all(sha(f)==manifest[str(f)] for f in deps)
            save(p/'progress.json', dict(stage='full_numerical_reference', host_elapsed_s=time.monotonic()-start))
            check = numerical(p, t, hq, hkv, bc)
            measured = epochs(p)[0]
            if control:
                old = OLD/'runs'/f'{model}_{arch}_vl16_gate'
                assert measured == epochs(old)[0], ('control counter drift', label)
                assert all(sha(p/f'{n}.fp16') == sha(old/f'{n}.fp16') for n in ('Q','K','V','O'))
                save(p/'equivalence.json', dict(passed=True, counters=len(measured), original=str(old)))
            row = dict(model=model, arch=arch, Br=br, Bc=bc, T=t, gate=gate, control=control,
                       roi_us=float(measured['simTicks'])/1e6, path=str(p), numerical=check,
                       host_elapsed_s=time.monotonic()-start)
            save(p/'accepted.json', row); save(p/'progress.json', dict(stage='accepted', host_elapsed_s=row['host_elapsed_s']))
            return row
        except BaseException as e:
            save(p/'failed.json', dict(error=repr(e))); stop.set(); raise
        finally:
            with state_lock: state['active'].pop(label, None); status()

    def report():
        targets = [r for r in state['completed'] if not r['gate'] and not r['control']]
        text = ['# MICRO VL16 reconstruction (corrected guests)', '',
                'Target: T4096/Hq32/Hkv8/d64, one core; upstream FU configuration.',
                'No published-result reproduction claim until timing and provenance are resolved.', '',
                '| Model | Kernel | Br | Bc | ROI ms |', '|---|---|---:|---:|---:|']
        text += [f"| {r['model']} | {r['arch']} | {r['Br']} | {r['Bc']} | {r['roi_us']/1000:.6f} |" for r in targets]
        pairs = {}
        for model in ('classic', 'ruby'):
            rows = {r['arch']: r for r in targets if r['model']==model and (r['Br'],r['Bc'])==(64,384)}
            if len(rows)==2:
                ratio=rows['cache']['roi_us']/rows['spm']['roi_us']
                old=json.loads((OLD/'summary_vl16.json').read_text())
                pairs[model]=dict(speedup=ratio, old_speedup=old['speedup'][model],
                                  latency_us={a:rows[a]['roi_us'] for a in rows})
                text += ['', f'{model} fixed 64/384: {ratio:.6f}x (old core {old["speedup"][model]:.6f}x).']
        save(OUT/'fixed_pair_effects.json', pairs)
        (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')

    def batch(stage, cases):
        status(stage)
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, *c) for c in cases]
            try:
                for f in cf.as_completed(futures):
                    row=f.result()
                    if not any(r['path']==row['path'] for r in state['completed']): state['completed'].append(row)
                    status(); report()
            except BaseException: stop.set(); raise

    try:
        batch('common_core_equivalence', [(m,a,32,128,False,True) for m in ('classic','ruby') for a in ('cache','spm')])
        batch('paper_core_all_tile_gates', [(m,a,br,bc,True,False) for br,bc in TILES for m in ('classic','ruby') for a in ('cache','spm')])
        save(OUT/'gates_passed.json', dict(common_controls=4, tile_gates=24, passed=True))
        batch('classic_fixed_pair', [('classic',a,64,384) for a in ('cache','spm')])
        batch('ruby_fixed_pair', [('ruby',a,64,384) for a in ('cache','spm')])
        batch('classic_independent_tile_search', [('classic',a,br,bc) for br,bc in TILES[1:] for a in ('cache','spm')])
        winners = {a:min((r for r in state['completed'] if r['model']=='classic' and r['arch']==a and r['T']==4096), key=lambda r:r['roi_us']) for a in ('cache','spm')}
        save(OUT/'classic_winners.json', dict(winners=winners, speedup=winners['cache']['roi_us']/winners['spm']['roi_us']))
        batch('ruby_classic_winner_transfer', [('ruby',a,r['Br'],r['Bc']) for a,r in winners.items()])
        for a,r in winners.items():
            ruby=OUT/'runs'/f'ruby_{a}_target_br{r["Br"]}_bc{r["Bc"]}'
            assert all(sha(Path(r['path'])/f'{n}.fp16')==sha(ruby/f'{n}.fp16') for n in ('Q','K','V','O'))
        assert all(sha(Path(f))==h for f,h in manifest.items())
        status('completed_requires_paper_provenance_and_gap_analysis'); report()
    except BaseException as e:
        state['error']=repr(e); status('failed'); raise


if __name__=='__main__':
    if '--launch' in sys.argv:
        assert not OUT.exists()
        with OUT.with_suffix('.host.log').open('x') as log:
            p=subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())],
                               stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               env=ENV, cwd=ROOT, start_new_session=True)
        print(p.pid)
    else:
        main()
