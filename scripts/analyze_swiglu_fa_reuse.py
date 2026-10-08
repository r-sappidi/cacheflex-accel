"""Read-only extraction of completed ROI blocks; never accepts partial runs."""
import configparser
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
MLP = ROOT / 'results/ai_workloads/swiglu_model_pilot_v4_20260921'
FA = ROOT / 'results/cacheflex_flash_attention/output_resident_v10_20260924'


def extract(path, roi_limit=None):
    raw = (path / 'stats.txt').read_bytes()
    rows = []
    for block in raw.decode().split('Begin Simulation Statistics')[1:]:
        if 'End Simulation Statistics' not in block:
            continue
        stats = {}
        for line in block.split('End Simulation Statistics')[0].splitlines():
            w = line.split()
            if len(w) > 1:
                try:
                    stats[w[0]] = float(w[1])
                except ValueError:
                    pass
        def total(pattern):
            return sum(v for k, v in stats.items() if re.search(pattern, k))
        row = dict(ms=stats['simTicks']/1e9, instructions=stats['simInsts'])
        for name, pattern in {
            'l1d_misses': r'l0_cntrl\d+\.Dcache\.m_demand_misses$',
            'private_l2_misses': r'l1_cntrl\d+\.cache\.m_demand_misses$',
            'dram_read_bytes': r'\.dram\.bytesRead::total$',
            'packets': r'^system\.ruby\.network\.packets_injected::total$',
            'copies': r'\.m_spmcpRequestsIssued$',
            'copies_completed': r'\.m_spmcpRequestsCompleted$',
            'copy_busy_cycles_sum': r'\.m_spmcpBusyCycles$',
            'copy_outstanding_area': r'\.m_spmcpOutstandingCycleArea$',
            'spm_reschedules': r'\.spmLsq0\.rescheduledLoads$',
            'spm_load_responses': r'\.spmLsq0\.spmLoadResponses$',
            'spm_load_service_cycles_sum': r'\.spmLsq0\.spmLoadServiceCycles$',
        }.items():
            row[name] = total(pattern)
        if row['copies']:
            row['copy_mean_outstanding_cycles'] = row['copy_outstanding_area']/row['copies']
            # Occupancy measure only; NOT a fraction of runtime caused by copies.
            row['copy_busy_mean_core_ms'] = row['copy_busy_cycles_sum']/8/2.5e6
        if row['spm_load_responses']:
            row['spm_load_mean_service_cycles'] = row['spm_load_service_cycles_sum']/row['spm_load_responses']
        rows.append(row)
        if roi_limit is not None and len(rows) >= roi_limit:
            break  # FA's later exit dump includes post-ROI checking/export.
    cfg = configparser.ConfigParser(interpolation=None, strict=False)
    cfg.read(path/'config.ini')
    fields = {}
    for sec, keys in {
        'system.cpu0': ['spmLoadPorts', 'cacheLoadPorts', 'LQEntries', 'SQEntries'],
        'system.ruby.l0_cntrl0.Dcache': ['size'],
        'system.ruby.l1_cntrl0.cache': ['size', 'num_spm_ways'],
        'system.ruby.l2_cntrl0.L2cache': ['size'],
        'system.ruby.l0_cntrl0.sequencer': ['max_outstanding_requests'],
    }.items():
        for key in keys:
            fields[sec+'.'+key] = cfg.get(sec, key, fallback=None)
    return dict(path=str(path), stats_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
                completed_roi_blocks=rows, config=fields,
                guest_checks=[l for l in (path/'run.log').read_text().splitlines()
                              if l.startswith('MLP_RESULT')])


def main():
    data = {'scope': 'Provisional SwiGLU completed iterations; FA validated runs. Different frozen models, not a workload-only controlled comparison.'}
    for arch in ('cache', 'spm'):
        data['swiglu_'+arch] = extract(MLP/'runs'/f'pilot_{arch}_t2048_d2048_f8192_mp2_kc128_g0')
    for row in json.loads((FA/'status.json').read_text())['completed']:
        if row.get('T') == 4096 and row['kind'] in ('recheck_cache', 'recheck_spm'):
            data['fa_'+row['kind'].removeprefix('recheck_')] = extract(Path(row['path']), roi_limit=1)
    data['comparisons'] = {}
    for workload in ('swiglu', 'fa'):
        comparisons = []
        for c, s in zip(data[workload+'_cache']['completed_roi_blocks'], data[workload+'_spm']['completed_roi_blocks']):
            comparisons.append(dict(speedup=c['ms']/s['ms'], **{
                k+'_reduction_pct': 100*(1-s[k]/c[k])
                for k in ('ms','instructions','l1d_misses','private_l2_misses','dram_read_bytes','packets') if c[k]
            }))
        data['comparisons'][workload] = comparisons
    rendered = json.dumps(data, indent=2)
    if len(sys.argv) == 3 and sys.argv[1] == '--save':
        dest = Path(sys.argv[2])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rendered+'\n')
        print(dest)
    else:
        print(rendered)


if __name__ == '__main__':
    main()
