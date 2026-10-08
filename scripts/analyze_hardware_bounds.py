"""Report validated finite-resource bounds without inferring causal cycles."""
import json
from pathlib import Path
from run_hardware_bounds import EX, numeric_stats

def main():
    rows = []
    for accepted in sorted((EX / 'runs').glob('*/accepted.json')):
        row = json.loads(accepted.read_text())
        epochs = numeric_stats(accepted.parent / 'stats.txt')
        def maximum(suffix):
            return max((v for e in epochs for k,v in e.items() if k.endswith(suffix)), default=0)
        rows.append(dict(case=accepted.parent.name, T=row['T'], arch=row['arch'],
            kind=row['kind'], roi_us=row['counters']['roi_us'],
            arithmetic=row['validation'],
            maximum_observed_LQ_occupancy=maximum('boundedMaxLoadOccupancy'),
            maximum_observed_SQ_occupancy=maximum('boundedMaxStoreOccupancy'),
            maximum_global_rejects_any_core_epoch=maximum('m_globalAdmissionFullRejects'),
            guard_equivalence=row.get('nonbinding_guard'), path=row['path']))
    report = EX / 'resource_audit'; report.mkdir(exist_ok=True)
    (report / 'results.json').write_text(json.dumps(rows, indent=2)+'\n')
    lines = ['# Finite-resource validation audit', '',
        'Only independently accepted runs are included. Capacities are a candidate hardware contract, not silicon calibration.', '',
        '| Case | ROI (us) | Max LQ | Max SQ | Global guard rejects* |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['case']} | {row['roi_us']:.4f} | {row['maximum_observed_LQ_occupancy']:.0f} | {row['maximum_observed_SQ_occupancy']:.0f} | {row['maximum_global_rejects_any_core_epoch']:.0f} |")
    lines += ['', '*Maximum per-core per-epoch reject count. Zero means no global guard rejection in any saved epoch. Default-off occupancy instrumentation is disabled, so its zero maxima are not physical occupancy measurements.', '']
    for t in sorted({r['T'] for r in rows}):
        found = {r['arch']:r for r in rows if r['T']==t and r['kind']=='bounded'}
        if len(found)==2:
            speedup=found['cache']['roi_us']/found['spm']['roi_us']
            lines += [f'T{t}: matched finite-profile cache/SPM speedup **{speedup:.6f}x**. One deterministic timing per configuration; not a tuned or paper-ready result.']
    for row in rows:
        if row['guard_equivalence']:
            lines += [f"- {row['case']}: {row['guard_equivalence']}."]
    lines += ['', 'Where guard-equivalence is explicitly reported above, the global sequencer cap is nonbinding for that tested pair. Missing pairs remain pending. These tests do not establish area/timing calibration, exhaustive deadlock freedom, or an additive causal latency breakdown.']
    (report / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__ == '__main__': main()
