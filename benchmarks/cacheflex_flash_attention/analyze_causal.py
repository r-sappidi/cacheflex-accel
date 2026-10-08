"""Factorial latency-gap attribution, explicitly conditional on interventions.

Shapley values allocate interactions without pretending that overlapping
service-time counters are additive wall-clock penalties.
"""
import json
import math
from pathlib import Path
import statistics
import sys

FACTORS = ('copy_service', 'spm_load_service', 'coherence_transport', 'capacity')

def decompose(values, factors=FACTORS):
    n = len(factors)
    assert set(values) == set(range(1 << n)), 'incomplete factorial'
    phi = {}
    for i, name in enumerate(factors):
        bit = 1 << i
        phi[name] = sum(
            math.factorial(mask.bit_count()) * math.factorial(n-mask.bit_count()-1)
            / math.factorial(n) * (values[mask] - values[mask | bit])
            for mask in range(1 << n) if not mask & bit)
    original, residual = values[0], values[(1 << n)-1]
    assert math.isclose(sum(phi.values()) + residual, original, abs_tol=1e-8)
    return dict(original=original, residual=residual, contributions=phi,
                standalone={name: values[0]-values[1 << i] for i,name in enumerate(factors)},
                # Positive means the combined improvement exceeds the sum
                # of the two individual improvements at default settings.
                pair_synergy={f'{factors[i]}+{factors[j]}':
                    values[1 << i] + values[1 << j] - values[0] - values[(1 << i)|(1 << j)]
                    for i in range(n) for j in range(i+1,n)})

def analyze(rows, tokens):
    by = {}
    for row in rows:
        if row['stage'] == 'factorial' and row['t'] == tokens:
            by.setdefault((row['variant'],row['mask']), []).append(row['counters']['roi_us'])
    assert set(by) == {(v,m) for v in ('cache','spm') for m in range(16)}
    latencies = {v: {m: statistics.median(by[v,m]) for m in range(16)} for v in ('cache','spm')}
    gap = {m:latencies['spm'][m]-latencies['cache'][m] for m in range(16)}
    parts = decompose(gap)
    parts['units'] = 'microseconds of SPM minus cache wall-time gap'
    parts['latencies'] = latencies
    parts['spm_only'] = decompose(latencies['spm'])
    parts['baseline_only'] = decompose(latencies['cache'])
    parts['fraction_of_original_gap'] = ({k:v/gap[0] for k,v in parts['contributions'].items()}
                                        if gap[0] > 0 else None)
    parts['warning'] = ('Intervention-dependent attribution, not an intrinsic instruction-cost '
                        'breakdown. Residual retains compute, software scheduling, frontend costs, '
                        'unremoved memory/coherence delays and interactions with them. Negative '
                        'contributions are retained. Endpoint repetitions are not factorial CIs.')
    endpoints = {}
    for v in ('cache','spm'):
        for m in (0,15):
            times=[r['counters']['roi_us'] for r in rows if r['t']==tokens and
                   r['variant']==v and r['mask']==m and
                   r['stage'] in ('factorial','endpoint_repeats')]
            median=statistics.median(times)
            endpoints[f'{v}_m{m}']=dict(samples_us=times,median_us=median,
                range_us=max(times)-min(times),relative_range=(max(times)-min(times))/median)
    parts['endpoint_repeatability']=endpoints
    return parts

def report(out, rows, tokens=1024):
    result = analyze(rows, tokens)
    (out/'attribution.json').write_text(json.dumps(result, indent=2))
    lines = ['# FA matched causal ablations', '', result['warning'], '',
             f'T={tokens}, native VL16, eight cores, unchanged FA guests and math.',
             'All configurations require matching finite output hashes, normal exits, copy counts and configuration audits.',
             'Hash agreement is regression testing, not an independent numerical reference.', '',
             '| Intervention | Standalone gap reduction (us) | Interaction-allocated reduction (us) |',
             '|---|---:|---:|']
    for name in FACTORS:
        lines.append(f"| {name} | {result['standalone'][name]:.4f} | {result['contributions'][name]:.4f} |")
    lines += ['', f"Original SPM-minus-cache gap: {result['original']:.4f} us.",
              f"Residual gap with all four interventions: {result['residual']:.4f} us.",
              'The interaction-allocated contributions plus residual sum to the original gap.', '',
              'See experiment.json for exact interventions and summary.json for all measurements.']
    lines+=['', '| Endpoint | Median (us) | Range (us) | Samples |', '|---|---:|---:|---:|']
    for name,row in result['endpoint_repeatability'].items():
        lines.append(f"| {name} | {row['median_us']:.4f} | {row['range_us']:.4f} | {len(row['samples_us'])} |")
    lines+=['', 'Do not treat effects comparable to repeatability ranges as established causal differences.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result

if __name__ == '__main__':
    out = Path(sys.argv[1])
    report(out, json.loads((out/'summary.json').read_text()))
