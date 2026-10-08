"""Stage-one native VL16 tile search, with frozen guests and incremental results.

Usage: python3 sweep_vl16_tiles.py NEW_ABSOLUTE_OUTPUT_DIRECTORY
Only the previously tested scalar K-only double-buffered guest is used.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from validate_attribution import read

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def main():
    out = Path(sys.argv[1])
    assert out.is_absolute()
    out.mkdir()  # Never overwrite an existing sweep.
    guests = out / 'guests'
    guests.mkdir()
    control = ROOT / 'results/cacheflex_flash_attention/clean_vl16_20260908/runs'
    for name in ('cache', 'spm'):
        shutil.copy2(control/'guests'/name, guests/name)
    shutil.copy2(control/'gem5.opt', out/'gem5.opt')
    shutil.copy2(HERE/'run_8core.sh', out/'runner.snapshot.sh')
    manifest = {}
    for path in [out/'gem5.opt', guests/'cache', guests/'spm', HERE/'run_8core.sh',
                 ROOT/'gem5/configs/ruby/MESI_Three_Level.py']:
        manifest[str(path)] = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))

    def run(br, bc, variant):
        # Four ways x two banks x (Bc/128)*d sets, within 1024 physical sets.
        assert br % 8 == 0 and bc % 128 == 0 and 2*(bc//128)*64 <= 1024
        name = f'{variant}_t1024_br{br}_bc{bc}'
        path = out/name
        env = dict(PATH=os.environ['PATH'], HOME=os.environ['HOME'],
                   CACHEFLEX_FLASH_VL='16', CACHEFLEX_FLASH_MATCHED_LRU='1',
                   CACHEFLEX_FLASH_SPM_WAYS='4' if variant == 'spm' else '0',
                   CACHEFLEX_FLASH_BIN=str(guests/variant),
                   CACHEFLEX_GEM5_BIN=str(out/'gem5.opt'),
                   CACHEFLEX_FLASH_OUT=str(path))
        mode = 'spm_konly_overlap' if variant == 'spm' else 'cache_pretouch'
        with (out/f'{name}.host.log').open('w') as log:
            subprocess.run(['bash', str(HERE/'run_8core.sh'), mode, '1024', str(br), str(bc)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        output, counters, cfg = read(path, expected_vl=16)
        assert output['Br'] == str(br) and output['Bc'] == str(bc)
        expected = 16384 if variant == 'spm' else 0
        assert counters['m_spmcpRequestsIssued'] == counters['m_spmcpRequestsCompleted'] == expected
        for core in range(8):
            assert cfg[f'system.cpu{core}.isa']['sve_vl_se'] == '16'
            assert cfg[f'system.ruby.l1_cntrl{core}.cache.replacement_policy']['type'] == 'LRURP'
        return dict(name=name, variant=variant, br=br, bc=bc, output=output, counters=counters)

    rows, errors = [], []
    jobs = [(br, bc, variant) for br in (16,32,64,128)
            for bc in (128,256,512) for variant in ('cache','spm')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run,*job):job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                row = future.result()
                rows.append(row)
                print(row['name'], row['counters']['roi_us'], row['output']['hash'], flush=True)
            except Exception as exc:
                errors.append(dict(job=futures[future], error=repr(exc)))
                print('FAILED', futures[future], repr(exc), flush=True)
            (out/'progress.json').write_text(json.dumps(dict(completed=rows,errors=errors),indent=2))

    pairs = []
    for br in (16,32,64,128):
        for bc in (128,256,512):
            pair = {r['variant']:r for r in rows if r['br']==br and r['bc']==bc}
            if len(pair) != 2:
                continue
            a,b = pair['cache'],pair['spm']
            assert a['output']['hash'] == b['output']['hash'], (br,bc,'output mismatch')
            _,_,ca=read(out/a['name'],expected_vl=16)
            _,_,cb=read(out/b['name'],expected_vl=16)
            assert set(ca.sections())==set(cb.sections())
            for section in ca.sections():
                assert set(ca[section])==set(cb[section])
                for key,x in ca[section].items():
                    y=cb[section][key]
                    if x==y or (section=='system.cpu0.workload' and key in ('cmd','executable')):
                        continue
                    assert key=='num_spm_ways' and x=='0' and y=='4',(section,key,x,y)
            pairs.append(dict(br=br,bc=bc,speedup=a['counters']['roi_us']/b['counters']['roi_us']))
    assert not errors and len(pairs)==12, errors
    winners={v:min((r for r in rows if r['variant']==v),key=lambda r:r['counters']['roi_us']) for v in ('cache','spm')}
    summary=dict(pairs=pairs,winners=winners,
                 best_to_best_speedup=winners['cache']['counters']['roi_us']/winners['spm']['counters']['roi_us'],
                 validation='Finite matching pair hashes only; not independent numerical reference validation.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE',summary['best_to_best_speedup'],flush=True)


if __name__ == '__main__':
    main()
