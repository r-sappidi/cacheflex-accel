"""Independent final checks/report for the timing-alignment FA experiment."""
import json
import re
import argparse
import time
import numpy as np
from build_access_timing import sha,save
from run_timing_fa import EX,OUT,OLD,REF,config_audit,check
from run_access_timing import stats
from test_micro_timing import check_global_guard

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    if args.wait:
        while True:
            state=json.loads((EX/'status.json').read_text())
            assert state['stage']!='failed',state
            if state['stage']=='completed_validated':break
            time.sleep(5)
    manifest=json.loads((EX/'inputs.json').read_text())
    for p,h in manifest.items():assert sha(p)==h,('Changed experiment input',p)
    calibration=json.loads((OUT/'calibration.json').read_text())
    assert calibration['passed']
    assert calibration['profile_sha256']==sha(EX/'configs/deprecated/example/micro_timing_se.py')
    rows=[]
    for t in (512,4096):
        for arch in ('cache','spm'):
            p=EX/f'{arch}_t{t}'
            if not (p/'validated.json').exists():continue
            row=json.loads((p/'validated.json').read_text());assert row['passed']
            config_audit(OLD/f'{arch}_t{t}_off',p)
            guards=check_global_guard(stats(p)[0])
            numeric=check(p/'output.fp16',np.load(REF/f't{t}_bc128.npy'))
            assert numeric['errors']==numeric['nonfinite']==0
            # In-guest checksum vs complete exported buffer, separately from
            # equality to previous accepted output and the arithmetic oracle.
            h=1469598103934665603
            for v in np.fromfile(p/'output.fp16',dtype=np.uint16):h=((h^int(v))*1099511628211)&((1<<64)-1)
            expected=re.search(r'FLASH_RESULT.*?hash=([0-9a-f]+)',(p/'run.log').read_text())[1]
            assert f'{h:016x}'==expected
            assert sha(p/'output.fp16')==sha(OLD/f'{arch}_t{t}_off/output.fp16')
            rows.append(dict(**row,global_guard_rejects=guards,guest_hash=expected))
    save(EX/'verification.json',dict(passed=True,complete=len(rows)==4,rows=rows,
        scope='Only completed accepted cases; pending runs excluded'))
    lines=['# Fixed-kernel FA after MICRO timing calibration','',
        '8 cores, VL16, single head, d64; unchanged64KiB privateL2/4MiB totalLLC scaled workload configuration. '
        'Cache Br32/Bc128/G2; SPM Br16/Bc128/G4, two SPM ways. Prefetch remains enabled. '
        'Same resource pools/retention policy/NoC and FP16 inputs/FP32 online state. '
        'This is not the paper\'s original workload or a retuned best-to-best comparison.',
        '', '| T | Backend | Old ROI us | Calibrated ROI us |', '|---|---|---:|---:|']
    for r in rows:lines.append(f'| {r["T"]} | {r["arch"]} | {r["old_roi_us"]:.4f} | {r["roi_us"]:.4f} |')
    for t in (512,4096):
        pair={r['arch']:r for r in rows if r['T']==t}
        if len(pair)==2:
            lines.append(f'\nT{t}: cache/SPM = {pair["cache"]["roi_us"]/pair["spm"]["roi_us"]:.6f}x '
                f'(previous {pair["cache"]["old_roi_us"]/pair["spm"]["old_roi_us"]:.6f}x).')
    lines += ['', 'All listed outputs pass the unchanged independent numerical reference, exact prior-output hash, '
        'and in-guest/export hash agreement. Full configuration audits allow only timing fields and the documented '
        'post-ROI exporter patch. All8 global sequencer guard counters are present and zero.',
        '', 'Local ordering, no-barrier WB RAW, and cross-core release/GETX regressions passed. '
        'Default-timing exporter controls reproduce every finite non-host ROI counter exactly.',
        '', 'Pending cases are not results. See status.json and ../REPORT.md for calibration scope/residuals.']
    (EX/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=len(rows)==4,rows=[dict(T=r['T'],arch=r['arch'],roi_us=r['roi_us']) for r in rows]),indent=2))

if __name__=='__main__':main()
