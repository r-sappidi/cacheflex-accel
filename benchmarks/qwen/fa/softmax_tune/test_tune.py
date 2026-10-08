"""Host-only FA-51 tests; no simulation."""
import json, shutil, sys, tempfile
from pathlib import Path
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
import tune as T


def test_work_matches_fa35():
    n = 0
    for x in T.read(T.MATRIX)['completed']:
        c = x['case']
        if c['mode'] != 1 or c['hq'] != 14 or c['br'] != 16: continue
        tasks, updates, copies = T.work(c['t'], c['bc'], c['r'])
        assert (tasks, updates, copies) == (x['work']['tasks'], x['work']['updates'], x['work']['copies']), (x['name'], tasks, x['work']['tasks'])
        n += 1
    assert n >= 8; print(f'work() reproduces {n} FA-35 per-core task/update/copy records')


def test_command_matches_fa50():
    old = Path('/data2/rsappidi/cacheflex_mc/results/cacheflex_flash_attention/softmax_opt_20261005')
    for spm, lane, f in ((False, 0, 0), (True, 3, 15)):
        ref = json.loads(next((old / f'lane{lane}/runs').glob(f"t1024_{'spm' if spm else 'cache'}_f{f}/command.json")).read_text())
        new = T.command(T.case(1024, spm, 256, 256, f, 8, 'x'), Path('/tmp/x'))
        diff = [(a, b) for a, b in zip(new, ref) if a != b and not a.startswith('--outdir=')]
        assert len(new) == len(ref) and all('softmax_tune_20261005' in a and a.replace('softmax_tune_20261005', 'softmax_opt_20261005') == b for a, b in diff), diff
    print('Bc256/R256 commands equal FA-50 commands except guest path and outdir')


def test_credit_command_and_config_negative():
    c16 = T.case(1024, True, 512, 128, 15, 16, 'x'); cmd = T.command(c16, Path('/tmp/x'))
    assert sum(1 for x in cmd if '.dataMissCredits.capacity=16' in x) == 8 and '--options' in cmd
    assert cmd[cmd.index('--options') + 1].split() == list(map(str, [1024, 8, 16, 512, 8, 1024, 0, 1, 1, 1, 14, 2, 1, 1, 0, 0, 15]))
    tmp = Path(tempfile.mkdtemp())
    try:
        cfg = json.loads((T.PARENT / 'config.json').read_text().replace(str(T.PARENT), str(tmp))); s = cfg['system']
        s['cpu'][0]['workload'][0]['executable'] = str(T.G / 'guest.export-safe'); s['cpu'][0]['workload'][0]['cmd'] = ['kv-fa', *map(str, T.argv(c16))]
        for i in range(8):
            s['ruby'][f'l1_cntrl{i}']['cache']['num_spm_ways'] = 6; s['ruby'][f'l0_cntrl{i}']['dataMissCredits']['capacity'] = 16
        (tmp / 'config.json').write_text(json.dumps(cfg)); T.config_check(tmp, c16, cmd)
        s['ruby']['l1_cntrl2']['missCredits']['capacity'] = 40; (tmp / 'config.json').write_text(json.dumps(cfg))
        try: T.config_check(tmp, c16, cmd)
        except AssertionError: pass
        else: raise SystemExit('config check accepted an unplanned change')
    finally: shutil.rmtree(tmp)
    print('16-credit command adds exactly 8 overrides; config check accepts plan, rejects L2-credit change')


def test_bits_and_guest():
    bits = T.matrix_bits()
    assert bits[(4096, 256)].startswith('27f14c53') and bits[(4096, 128)].startswith('a52e9614') and bits[(4096, 512)].startswith('994d3ce2')
    for p, h in json.loads((T.G / 'frozen.json').read_text()).items(): assert T.sha(p) == h, p
    print('FA-35 reference bits loaded for', sorted(bits), '; guest v2 frozen hashes ok')


if __name__ == '__main__':
    test_work_matches_fa35(); test_command_matches_fa50(); test_credit_command_and_config_negative(); test_bits_and_guest(); print('ALL HOST TESTS PASS')
