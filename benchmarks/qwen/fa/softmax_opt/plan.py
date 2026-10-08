"""FA-50 plan: correctness gates for every flag, then T1024 timing for combinations.

Same shape as FA-39 (Hq14/Hkv2/d64, Br16/Bc256/R256, causal, 8 cores), same
retained simulator and configuration; only the guest ELF and its flag word differ.
"""
EXACT = (0, 1, 2, 4, 8, 15)          # must reproduce FA-39 output bits exactly
PROBE = (16, 31)                     # Q pre-scale: oracle required, bit equality recorded
FLAGS = EXACT + PROBE
TIMING = (0, 15, 31)                 # T1024 matched cache/SPM timing set
# FA-39 accepted control output bits (same math, same shapes).
REFERENCE_SHA256 = {256: '554df33e099a389cc525322c6a524d7f5cdb14bcc1e5668bfe7044aa6ffb3782',
                    1024: '5f479413ee88e849e40d057776011fc09a483bf2606a117ae9be3c811bb51a79'}
LANE_OFFSETS = (12, 14, 16, 18)      # budget cpu_set offsets (AI-13 uses 4-11, FA-41 0-1)


def name(c): return f"t{c['t']}_{'spm' if c['spm'] else 'cache'}_f{c['flags']}"


def cases():
    rows = [dict(t=256, spm=spm, flags=f, stage='gate') for f in FLAGS for spm in (False, True)]
    rows += [dict(t=1024, spm=spm, flags=f, stage='timing') for f in TIMING for spm in (False, True)]
    for r in rows: r['name'] = name(r)
    return rows


def parameters(spm):
    return {f'system.ruby.l1_cntrl{i}.cache.num_spm_ways': 6 if spm else 0 for i in range(8)}


def argv(c):
    spm = int(c['spm'])
    return [c['t'], 8, 16, 256, 16, 1024, 0, spm, 1, 1, 14, 2, 1, spm, 0, 0, c['flags']]


def work(t, hq=14):
    tasks = [0] * 8; updates = [0] * 8; copies = 0; groups = t // 256
    for h in range(hq):
        for group in range(groups):
            core = (h * groups + (group - h) % groups) % 8; end = (group + 1) * 256
            tasks[core] += 1; updates[core] += 256 * (end // 256); copies += 2 * end
    return tasks, updates, copies


def lanes():
    """Gates first (all flags), then T1024; longest work spread across four lanes."""
    rows = cases(); out = [[] for _ in LANE_OFFSETS]
    for i, c in enumerate([r for r in rows if r['stage'] == 'gate']): out[i % 4].append(c)
    for i, c in enumerate([r for r in rows if r['stage'] == 'timing']): out[i % 4].append(c)
    return out
