"""Follow SPMLD through lane assembly to its first committed FMA.

This is a sampled dependency explanation, not an additive runtime attribution.
Only splice is traversed transparently; an accumulator FMA ends the traversal.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from analyze_spm_consumers import (parse_lines, describe, producer_kind,
                                  anchors, distribution, CYCLE)
from build_access_timing import save


def complete(n):
    return all(k in n for k in ('dispatch', 'done', 'commit', 'ready')) and len(n['issues']) == 1


def mnemonic(n):
    return n.get('asm', '').strip().split()[0] if n.get('asm', '').strip() else ''


def root_producer(n, nodes):
    """Traverse the latest actual producer of a splice, stopping at memory/FMA.

    Missing/boundary provenance remains unknown, never inferred from dispatch.
    """
    chain = [describe(n)]
    for _ in range(6):
        if mnemonic(n) != 'splice':
            return producer_kind(n), chain
        sources = [nodes.get((n['core'], s.get('producer'))) for s in n['sources']]
        if not sources or any(p is None or 'done' not in p for p in sources):
            return 'unknown_boundary', chain
        n = max(sources, key=lambda p: p['done'])
        chain.append(describe(n))
    return 'depth_limit', chain


def fma_rows(nodes, anchor=None):
    users = defaultdict(list)
    for n in nodes.values():
        if not complete(n):
            continue
        for sn in {s.get('producer') for s in n['sources']}:
            if sn is not None:
                users[(n['core'], sn)].append(n)
    rows = []; excluded = Counter()
    for load in nodes.values():
        if not (load.get('spm') and load.get('load') and 'commit' in load):
            continue
        if anchor and load['sn'] < anchor[load['core']]['sn']:
            continue
        if not complete(load):
            excluded['incomplete_or_reissued_load'] += 1
            continue
        frontier = [(load, [load])]; found = []; visited = set()
        while frontier:
            parent, path = frontier.pop()
            if parent['sn'] in visited or len(path) > 4:
                continue
            visited.add(parent['sn'])
            for child in users[(load['core'], parent['sn'])]:
                if mnemonic(child).startswith(('fmla', 'fmls', 'fmadd', 'fmsub')):
                    found.append((child, parent, path))
                elif mnemonic(child) == 'splice':
                    frontier.append((child, path + [child]))
        if not found:
            excluded['no_complete_FMA_via_splice'] += 1
            continue
        c, operand, path = min(found, key=lambda x: (x[0]['issues'][0], x[0]['sn']))
        others = []; unknown = False
        for s in c['sources']:
            if s.get('producer') == operand['sn']:
                continue
            p = nodes.get((c['core'], s.get('producer')))
            if p and 'done' in p:
                others.append(p)
            elif s.get('ready') and c['dispatch'] <= operand['done']:
                pass  # This boundary-ready source is provably not later.
            else:
                unknown = True
        later = [p for p in others if p['done'] > operand['done']]
        blocker_chain = []
        if unknown:
            reason = 'unknown_source'
        elif later:
            blocker = max(later, key=lambda p: p['done'])
            kind, blocker_chain = root_producer(blocker, nodes)
            reason = 'other_operand:' + kind
        elif c['dispatch'] > operand['done']:
            reason = 'FMA_dispatched_after_SPM_operand_ready'
        else:
            reason = 'SPM_operand_last_or_tied'
        waits = Counter(why for t, why in c['waits'] if c['ready'] <= t < c['issues'][0])
        rows.append(dict(load=describe(load), operand_path=[describe(p) for p in path],
                         consumer=describe(c), reason=reason, blocker_chain=blocker_chain,
                         load_latency_cycles=(load['done'] - load['issues'][0])/CYCLE,
                         operand_to_ready_cycles=max(0, c['ready']-operand['done'])/CYCLE,
                         ready_to_issue_cycles=(c['issues'][0]-c['ready'])/CYCLE,
                         load_to_FMA_cycles=(c['issues'][0]-load['done'])/CYCLE,
                         ready_queue_wait_cycles=dict(waits)))
    return rows, dict(excluded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('experiment', type=Path); ap.add_argument('label')
    args = ap.parse_args(); ex = args.experiment; label = args.label
    assert json.loads((ex/f'{label}_observer_equivalence.json').read_text())['passed']
    summaries = {}
    for v in ('normal', 'fast'):
        path = ex/(label+'_trace')/v
        with gzip.open(path/'trace.gz', 'rt') as f:
            nodes, rob, conflicts = parse_lines(f)
        rows, excluded = fma_rows(nodes, anchors(ex/(label+'_scan')/v))
        assert len(rows) > 64, (v, len(rows))
        reasons = Counter(r['reason'] for r in rows); waits = Counter(); roots = Counter()
        for r in rows:
            waits.update(r['ready_queue_wait_cycles'])
            if r['blocker_chain']:
                root = r['blocker_chain'][-1]
                roots[(root['pc'], root['asm'], root['kind'])] += 1
        cadence = {}
        for core in (0, 7):
            loads = sorted((n for n in nodes.values() if n['core']==core and
                           n.get('spm') and n.get('load') and complete(n)), key=lambda n:n['sn'])
            adjacent = [(a,b) for a,b in zip(loads,loads[1:]) if b['va']==a['va']+64]
            cadence[core] = {k: distribution([(b[k]-a[k])/CYCLE for a,b in adjacent])
                             for k in ('dispatch','done')}
            cadence[core]['issue'] = distribution([(b['issues'][0]-a['issues'][0])/CYCLE
                                                   for a,b in adjacent])
        blocked = Counter(nodes.get((core,sn),{}).get('pc')
                          for core, events in rob.items() for tick,sn in events)
        s = dict(samples=len(rows), excluded=excluded, reasons=dict(reasons),
                 cores=sorted({r['load']['core'] for r in rows}),
                 consecutive_slot_cadence=cadence,
                 blocked_ROB_head_samples=[dict(pc=hex(pc) if pc else None,count=n)
                                          for pc,n in blocked.most_common(12)],
                 metrics={k: distribution([r[k] for r in rows]) for k in
                          ('load_latency_cycles', 'operand_to_ready_cycles',
                           'ready_to_issue_cycles', 'load_to_FMA_cycles')},
                 waits=dict(waits), wakeup_provenance_overrides=conflicts,
                 root_pcs=[dict(pc=k[0], asm=k[1], kind=k[2], count=n)
                           for k,n in roots.most_common()], examples=[])
        for reason in reasons:
            group = sorted((r for r in rows if r['reason']==reason),
                           key=lambda r:r['load_to_FMA_cycles'])
            s['examples'].append(group[len(group)//2])
        save(path/'fma_consumer_rows.json', rows); summaries[v] = s
    save(ex/f'{label}_fma_analysis.json', dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        summaries=summaries))
    lines = [f'# SPMLD to first FMA: {label}', '',
             'Bounded cores 0/7 samples; trace-on/off exact non-host-stat equivalence passed. '
             'Follow actual register dependencies through splice lane assembly, stopping at first FMA. '
             'Counts are sampled load/FMA pairs, NOT percentages of FA runtime.', '',
             '| Metric (mean CPU cycles) | Normal | Fast |', '|---|---:|---:|']
    for k in summaries['normal']['metrics']:
        lines.append(f'| {k} | {summaries["normal"]["metrics"][k]["mean"]:.3f} | '
                     f'{summaries["fast"]["metrics"][k]["mean"]:.3f} |')
    lines += ['', '| Readiness classification | Normal samples | Fast samples |', '|---|---:|---:|']
    for k in sorted(set(summaries['normal']['reasons'])|set(summaries['fast']['reasons'])):
        lines.append(f'| {k} | {summaries["normal"]["reasons"].get(k,0)} | '
                     f'{summaries["fast"]["reasons"].get(k,0)} |')
    for v,s in summaries.items():
        lines += ['', f'## {v}', '', f'Samples: {s["samples"]}; exclusions: {s["excluded"]}.',
                  f'Accumulated ready-list wait events (not runtime): {s["waits"]}.', '', 'Later operand root PCs:', '']
        for p in s['root_pcs'][:10]:
            lines.append(f'- {p["pc"]}: `{p["asm"]}`, {p["kind"]}: {p["count"]} pairs.')
        lines += ['', 'Median successive-slot SPMLD dispatch/issue/completion intervals (cycles):', '']
        for core,c in s['consecutive_slot_cadence'].items():
            lines.append(f'- core {core}: {c["dispatch"]["median"]}, '
                         f'{c["issue"]["median"]}, {c["done"]["median"]}.')
    lines += ['', 'A prior FMA root identifies an accumulator dependency, not the ultimate '
              'source of that chain’s delay. A coherent-load root establishes a later data operand; '
              'its latency components and whole-run causal impact require separate evidence. '
              'Different variants are distributions, not assumed identical dynamic instruction sequences.']
    (ex/f'{label}_FMA_REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
