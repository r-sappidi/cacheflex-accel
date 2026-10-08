"""Host-only AI-13 tests; no simulation. Run before launch."""
import copy, json, shutil, sys, tempfile
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import plan as P
import run as RUN


def strip(cmd):
    return [x for x in cmd if not x.startswith('--outdir=')]


def test_plan_shape():
    names = []
    for wl in P.WORKLOADS:
        g, s = P.gate_cases(wl), P.screen_cases(wl)
        assert len(g) == 8 and all(c['credits'] == 16 and c['t'] == 24 for c in g)
        attrib = [c for c in s if c['stage'] == 'attrib']
        assert len(s) == 8 + len(attrib) and all(c['credits'] == 8 and c['arm'] == 'cache' for c in attrib)
        assert all(P.old_run(wl, 'cache', 1024, c['kc']) is None for c in attrib)
        names += [c['name'] for c in g + s]
    names += [c['name'] for c in P.replay_cases()]
    assert len(names) == len(set(names)), 'duplicate case names'
    print('plan shape ok:', {wl: len([c for c in P.screen_cases(wl) if c['stage'] == 'attrib']) for wl in P.WORKLOADS}, 'attribution cases')


def test_commands():
    tmp = Path(tempfile.mkdtemp())
    try:
        for c in P.replay_cases():
            old = P.old_run(c['wl'], c['arm'], c['t'], c['kc']); assert old, c
            cmd, _, _ = RUN.command(c, tmp)
            assert strip(cmd) == strip(json.loads((old / 'command.json').read_text())), ('replay command differs', c['name'])
        for wl in P.WORKLOADS:
            for c16 in P.gate_cases(wl) + P.screen_cases(wl):
                c8 = dict(c16, credits=8)
                a, _, _ = RUN.command(c8, tmp); b, _, _ = RUN.command(c16, tmp)
                if c16['credits'] == 8:
                    assert a == b
                else:
                    assert b[:len(a)] == a and b[len(a):] == sum((['-P', f'{k}=16'] for k in P.credit_parameters(16)), [])
    finally:
        shutil.rmtree(tmp)
    print('commands ok: 8-credit commands reproduce AI-11; 16-credit adds exactly 8 capacity overrides')


def test_validation_on_old_runs():
    RUN.EXPECTED = RUN.expected_hashes()
    for (wl, t) in [(w, t) for w in P.WORKLOADS for t in (24, *P.SIZES)]:
        assert (wl, t) in RUN.EXPECTED, ('no accepted reference bits', wl, t)
    for wl, arm, t, kc in [('out', 'cache', 1024, 512), ('qkv', 'spm', 1024, 1024), ('ffn', 'cache', 24, 512), ('ffn', 'spm', 24, 1024)]:
        old = P.old_run(wl, arm, t, kc) if t != 24 else next((p for p in sorted((P.AI10 / 'experiment_v1/runs').glob(f'qwen_shape_gates_{arm}_t24_mp8_kc{kc}_g0'))), None)
        assert old, (wl, arm, t, kc)
        c = P.case('screen', wl, arm, t, kc, credits=8)
        _, args, inp = RUN.command(c, old)
        row = RUN.validate(c, old, inp, args)
        acc = json.loads((old / 'accepted.json').read_text())
        assert (row['first_us'], row['repeat_us'], row['output_sha256']) == (acc['first_us'], acc['repeat_us'], acc['output_sha256'])
    print('full AI-13 validation accepts old AI-10/AI-11 runs and reproduces their timings/bits')


def test_config_negatives():
    old = P.old_run('out', 'cache', 1024, 512)
    c = P.case('screen', 'out', 'cache', 1024, 512, credits=8)
    for mutate, why in [(lambda s: s['ruby']['l1_cntrl0']['missCredits'].__setitem__('capacity', 40), 'L2 credits'),
                        (lambda s: s['cpu'][0].__setitem__('LQEntries', 64), 'LQ'),
                        (lambda s: s['ruby']['l0_cntrl3']['dataMissCredits'].__setitem__('capacity', 16), 'credit mismatch'),
                        (lambda s: s['ruby']['l1_cntrl0']['cache'].__setitem__('num_spm_ways', 6), 'SPM ways')]:
        tmp = Path(tempfile.mkdtemp())
        try:
            text = (old / 'config.json').read_text().replace(str(old), str(tmp)); cfg = json.loads(text)
            mutate(cfg['system']); (tmp / 'config.json').write_text(json.dumps(cfg))
            _, args, inp = RUN.command(c, tmp)
            try:
                RUN.config_audit(tmp, c, inp, args)
            except AssertionError:
                continue
            raise SystemExit(f'config audit accepted forbidden change: {why}')
        finally:
            shutil.rmtree(tmp)
    # Positive: credits 16 everywhere passes for a 16-credit case.
    tmp = Path(tempfile.mkdtemp())
    try:
        cfg = json.loads((old / 'config.json').read_text().replace(str(old), str(tmp)))
        for i in range(8): cfg['system']['ruby'][f'l0_cntrl{i}']['dataMissCredits']['capacity'] = 16
        (tmp / 'config.json').write_text(json.dumps(cfg)); c16 = dict(c, credits=16)
        _, args, inp = RUN.command(c16, tmp); RUN.config_audit(tmp, c16, inp, args)
    finally:
        shutil.rmtree(tmp)
    print('config audit rejects L2-credit/LQ/partial-credit/SPM-way changes, accepts uniform 16 credits')


def test_selection():
    rows = [dict(case=P.case('screen', 'ffn', 'cache', 1024, kc), first_us=f, repeat_us=r)
            for kc, f, r in [(128, 10, 9), (256, 9, 8.5), (512, 9.5, 8), (1024, 11, 10)]]
    assert P.select(rows, 'ffn', 'cache') == [256, 512]
    rows[2]['repeat_us'] = 9; assert P.select(rows, 'ffn', 'cache') == [256]
    print('selection ok')


if __name__ == '__main__':
    test_plan_shape(); test_commands(); test_config_negatives(); test_selection(); test_validation_on_old_runs()
    print('ALL HOST TESTS PASS')
