"""Run ACL first/warm MLP pairs on the frozen clean-MLP hardware.

Usage: python3 run_mlp_acl.py NEW_ABSOLUTE_OUTPUT BUILD_DIRECTORY
No simulation or reference artifact is overwritten. Numerical failures stop
the queue before larger shapes; no speedup is accepted from a failed case.
"""
from pathlib import Path
import concurrent.futures as cf
import configparser
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'cacheflex_flash_attention'))
from analyze_traffic import analyze
from validate_mlp_acl import validate

def config_audit(path, reference):
    a, b = (configparser.ConfigParser(interpolation=None, strict=False) for _ in range(2))
    a.read(reference / 'config.ini'); b.read(path / 'config.ini')
    assert a.sections() == b.sections()
    for sec in a.sections():
        assert set(a[sec]) == set(b[sec])
        for key, x in a[sec].items():
            y = b[sec][key]
            x = x.replace(str(reference), '<RUN>')
            y = y.replace(str(path), '<RUN>')
            if x == y:
                continue
            if sec == 'system.cpu0.workload' and key in ('cmd', 'executable'):
                continue
            if sec == 'system' and key == 'readfile':
                continue  # bytes separately SHA-256 checked against archived input
            raise AssertionError((sec, key, x, y))

def main():
    out, build = (Path(x).resolve() for x in sys.argv[1:3])
    out.mkdir()
    old = ROOT / 'results/ai_workloads/mlp_clean_20260909/runs'
    shutil.copy2(build / 'acl', out / 'acl')
    shutil.copy2(old / 'gem5.opt', out / 'gem5.opt')
    hashes = {v: hashlib.sha256((out/v).read_bytes()).hexdigest() for v in ('acl', 'gem5.opt')}
    rows = []

    def run(t, d, f, mode):
        name = f'acl_{mode}_t{t}_d{d}_f{f}'
        path = out / name
        env = dict(PATH=os.environ['PATH'], HOME=os.environ['HOME'],
                   CACHEFLEX_FLASH_VL='16', CACHEFLEX_FLASH_MATCHED_LRU='1',
                   CACHEFLEX_FLASH_SPM_WAYS='0',
                   CACHEFLEX_FLASH_OPTIONS=f'{t} {d} {f} {mode}',
                   CACHEFLEX_FLASH_READFILE=str(out/f'input_{t}_{d}_{f}.bin'),
                   CACHEFLEX_FLASH_BIN=str(out/'acl'), CACHEFLEX_FLASH_OUT=str(path),
                   CACHEFLEX_GEM5_BIN=str(out/'gem5.opt'))
        with (out / (name+'.host.log')).open('w') as log:
            subprocess.run(['bash', str(HERE.parent/'cacheflex_flash_attention/run_8core.sh'),
                            'cache_pretouch'], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        log = (path/'run.log').read_text()
        assert all(x in log for x in ('MLP_ROI begin', 'MLP_ROI end',
                                     'exiting with last active thread context',
                                     'sve=1 sve2=0 fp16=1 vl_bits=2048 threads=8'))
        record = next(l for l in log.splitlines() if l.startswith('MLP_RESULT'))
        result = dict(re.findall(r'(\w+)=([^\s]+)', record))
        assert not re.search(r'panic:|fatal:| FAIL', log), record
        assert result['nonfinite'] == '0', record
        roi = log.split('MLP_ROI begin',1)[1].split('MLP_ROI end',1)[0]
        assert '[INFO]' not in roi and '[CORE]' not in roi, 'ACL logging in ROI'
        numerical = validate(path, out/f'input_{t}_{d}_{f}.bin',t,d,f)
        config_audit(path, old/f'cache_t{t}_d{d}_f{f}_mp2')
        stats = analyze(path)
        assert stats['m_spmcpRequestsIssued'] == stats['m_spmcpRequestsCompleted'] == 0
        return dict(t=t, d=d, f=f, mode=mode, result=result, numerical=numerical,
                    counters=stats, path=str(path))

    for t, d, f in ((24,128,256), (64,512,1536), (128,512,1536)):
        inp = f'input_{t}_{d}_{f}.bin'
        shutil.copy2(old/inp, out/inp)
        hashes[inp] = hashlib.sha256((out/inp).read_bytes()).hexdigest()
        assert hashes[inp] == hashlib.sha256((old/inp).read_bytes()).hexdigest()
        (out/'hashes.json').write_text(json.dumps(hashes, indent=2))
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            pair = list(pool.map(lambda mode: run(t,d,f,mode), ('first','warm')))
        assert pair[0]['result']['hash'] == pair[1]['result']['hash']
        rows.extend(pair)
        (out/'progress.json').write_text(json.dumps(rows, indent=2))
        print('PASS', t, [(r['mode'], r['counters']['roi_us']) for r in pair], flush=True)
    (out/'summary.json').write_text(json.dumps(rows, indent=2))
    print('COMPLETE', flush=True)

if __name__ == '__main__':
    main()
