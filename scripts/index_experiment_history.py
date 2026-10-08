"""Index preserved experiment evidence without rerunning or altering experiments.

This is discovery, NOT a numerical validator. Marker names and first stats dumps
are recorded literally; the curated ledger owns scientific interpretation.
"""
import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/experiments'
SKIP = {'upstream', 'source', 'sources', 'configs', 'fs', '.git', '__pycache__',
        'micro_upstream_20260918', 'micro_rebuttal_audit_20260921', 'references',
        'toolchain', 'tests', 'site-packages', 'node_modules'}


def skip(name):
    return name in SKIP or name.startswith(('build', 'preserved', 'snapshot', 'source_snapshot'))


def rel(p):
    return str(p.relative_to(ROOT))


def load(p):
    try:
        if p.stat().st_size > 30_000_000:
            return {'index_note': 'Large JSON; consult source directly'}
        return json.loads(p.read_text())
    except (OSError, ValueError) as e:
        return {'index_error': str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--query', help='Search existing index, without rescanning')
    args = ap.parse_args()
    if args.query:
        data = load(OUT / 'inventory.json')
        needle = args.query.lower()
        ledger = (OUT / 'LEDGER.md').read_text(errors='replace') if (OUT / 'LEDGER.md').exists() else ''
        if needle in ledger.lower():
            print('--- curated ledger matches ---')
            for i, line in enumerate(ledger.splitlines(), 1):
                if needle in line.lower():
                    print(f'LEDGER.md:{i}:{line}')
        for row in data['runs'] + data['evidence']:
            if needle in json.dumps(row).lower():
                print(json.dumps(row))
        return
    runs, evidence, roots = [], [], []
    for area in sorted((ROOT / 'results').iterdir()):
        if not area.is_dir():
            continue
        if area.name in {'cacheflex_flash_attention', 'ai_workloads', 'cacheflex_spm_gemm', 'acl_sve_gemm'}:
            roots.extend(rel(p) for p in sorted(area.iterdir()) if p.is_dir())
        else:
            roots.append(rel(area))
        for folder, dirs, files in os.walk(area):
            dirs[:] = sorted(d for d in dirs if not skip(d))
            p = Path(folder)
            # Source checkouts at the walk root are excluded too.
            if any(skip(x) for x in p.relative_to(area).parts):
                dirs[:] = []
                continue
            markers = [f for f in files if f in {'accepted.json', 'validated.json',
                       'timing_only_complete.json', 'validation.json', 'result.json'}]
            if 'stats.txt' in files or markers:
                row = {'path': rel(p), 'markers': markers}
                for f in markers:
                    obj = load(p / f)
                    if isinstance(obj, dict):
                        row[f] = {k: v for k, v in obj.items() if k in {
                            'stage', 'status', 'case', 'roi_us', 'roi_ms', 'copies',
                            'arithmetic', 'numerical', 'errors', 'passed', 'validation',
                            'arithmetic_validation', 'returncode', 'exit_code', 'epochs'}}
                if 'stats.txt' in files:
                    # Only the small header: never read huge traces/stat histograms.
                    with (p / 'stats.txt').open(errors='replace') as stream:
                        for i, line in enumerate(stream):
                            if i > 150:
                                break
                            m = re.match(r'(simTicks|simSeconds|simInsts)\s+(\S+)', line)
                            if m:
                                row['first_dump_' + m[1]] = m[2]
                row['evidence_level'] = ('acceptance_marker_present_not_revalidated'
                    if any(x in markers for x in ('accepted.json', 'validated.json'))
                    else 'timing_only_marker' if 'timing_only_complete.json' in markers
                    else 'raw_or_diagnostic_only')
                runs.append(row)
            for f in sorted(files):
                if f.lower().endswith('.md') or f in {
                    'summary.json', 'results.json', 'status.json', 'progress.json',
                    'effects.json', 'measurements.json', 'winners_t1024.json',
                    'STOPPED.json', 'STOPPED_BY_USER.json', 'STOPPED_FOR_ALIGNMENT.json'}:
                    row = {'path': rel(p / f), 'kind': 'report' if f.endswith('.md') else f}
                    if f == 'status.json':
                        obj = load(p / f)
                        if isinstance(obj, dict):
                            row['status'] = {k: v for k, v in obj.items()
                                             if k in {'stage','status','error','reason'}}
                    evidence.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    data = dict(scope='All results/ areas; source/build/config/reference trees excluded; '
                'first_dump values are NOT automatically ROI; markers NOT revalidated. '
                'Historical arithmetic/model caveats in LEDGER.md override marker names.',
                study_roots=roots, runs=runs, evidence=evidence)
    (OUT / 'inventory.json').write_text(json.dumps(data, indent=2) + '\n')
    with (OUT / 'runs.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path','evidence_level','first_dump_simTicks','markers'])
        for r in runs:
            writer.writerow([r['path'],r['evidence_level'],r.get('first_dump_simTicks',''),';'.join(r['markers'])])
    # Preserve every old context entry as searchable evidence, not current truth.
    agents = (ROOT / 'AGENTS.md').read_text()
    sections = re.split(r'(?m)^## ', agents)
    history = [{'title': x.split('\n',1)[0], 'text': x.split('\n',1)[1] if '\n' in x else ''}
               for x in sections[1:] if not x.startswith('Experiment ledger')]
    (OUT / 'historical_context.json').write_text(json.dumps(history, indent=2) + '\n')
    md = ['# Evidence inventory', '', data['scope'], '',
          f'{len(roots)} study roots; {len(runs)} run/diagnostic directories; {len(evidence)} report/status/summary files.', '',
          'See [LEDGER.md](LEDGER.md) for interpreted results, [runs.csv](runs.csv) for individual cases, '
          'and [inventory.json](inventory.json) for source paths and recorded status.', '',
          '## Reports and summary files', '']
    for r in evidence:
        if r['kind'] in {'report','summary.json','results.json','effects.json','measurements.json'}:
            md.append(f'- [{r["path"]}](../../{r["path"]})')
    (OUT / 'INVENTORY.md').write_text('\n'.join(md) + '\n')
    print(json.dumps({'study_roots':len(roots),'runs':len(runs),'evidence_files':len(evidence),
                      'levels':dict(Counter(x['evidence_level'] for x in runs))}, indent=2))


if __name__ == '__main__':
    main()
