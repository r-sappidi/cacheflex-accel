"""AI-13 plan: Qwen FFN/projection cache-vs-SPM at 16 L1D miss credits.

Only hardware change from AI-10/AI-11: l0_cntrl{0..7}.dataMissCredits.capacity.
Guests, inputs, simulator, configuration script and every other parameter are
the frozen AI-10/AI-11 ones. Candidate tilings are a bounded KC axis at MP8
(analytical rationale in README.md), not a Cartesian sweep.
"""
from pathlib import Path

ROOT = Path('/data2/rsappidi/cacheflex_mc')
AI10 = ROOT / 'results/ai_workloads/qwen_swiglu_20260928'
AI11 = ROOT / 'results/ai_workloads/qwen_projection_20260928'
OUT = ROOT / 'results/ai_workloads/qwen_mshr16_20261005'
WORKLOADS = ('qkv', 'out', 'ffn')      # cheapest first
ARMS = ('cache', 'spm')
KCS = (128, 256, 512, 1024)            # B panel 32/64/128/256 KiB per 128-column tile
MP = 8
CREDITS = 16
DEFAULT_CREDITS = 8
SIZES = (1024, 2048, 4096)
GATE_T = 24
CPU_SLOTS = (4, 6, 8, 10)              # budget cpu_set offsets; FA-41 owns 0-1
COORDINATOR_SLOT = 2
COLUMNS = dict(qkv=1152, out=896, ffn=896)


def credit_parameters(credits):
    return {f'system.ruby.l0_cntrl{i}.dataMissCredits.capacity': credits for i in range(8)}


def name(c):
    return f"{c['stage']}_{c['wl']}_{c['arm']}_t{c['t']}_mp{c['mp']}_kc{c['kc']}_cr{c['credits']}"


def case(stage, wl, arm, t, kc, credits=CREDITS, mp=MP):
    c = dict(stage=stage, wl=wl, arm=arm, t=t, mp=mp, kc=kc, credits=credits)
    if wl == 'ffn':
        c.update(d=896, f=4864, g=0)
    else:
        c.update(op=wl)
    c['name'] = name(c)
    return c


def input_path(wl, t):
    if wl == 'ffn':
        return AI10 / 'inputs' / f't{t}_d896_f4864.bin'
    return AI11 / 'inputs' / f'v2_{wl}_t{t}.bin'


def old_run(wl, arm, t, kc, mp=MP):
    """Accepted AI-10/AI-11 run with identical tiling at 8 credits, if any."""
    if wl == 'ffn':
        hits = sorted((AI10 / 'experiment_v1/runs').glob(f'*_{arm}_t{t}_mp{mp}_kc{kc}_g0'))
    else:
        hits = sorted((AI11 / 'experiment_v2/runs').glob(f'*_{wl}_{arm}_t{t}_mp{mp}_kc{kc}'))
    hits = [h for h in hits if (h / 'accepted.json').exists() and not h.name.startswith(('shape_gates', 'tail_gates', 'qwen_shape', 'compat'))]
    return hits[0] if hits else None


def replay_cases():
    """8-credit exact replays of accepted AI-11 runs: command and results must match."""
    return [case('replay', 'out', 'cache', 1024, 512, DEFAULT_CREDITS),
            case('replay', 'out', 'spm', 1024, 1024, DEFAULT_CREDITS)]


def gate_cases(wl):
    return [case('gate', wl, arm, GATE_T, kc) for arm in ARMS for kc in KCS]


def screen_cases(wl):
    rows = [case('screen', wl, arm, 1024, kc) for arm in ARMS for kc in KCS]
    # Tiling attribution at the default 8 credits: cache tilings never run in AI-10/11.
    rows += [case('attrib', wl, 'cache', 1024, kc, DEFAULT_CREDITS) for kc in KCS
             if old_run(wl, 'cache', 1024, kc) is None]
    return rows


def select(rows, wl, arm):
    """Per-backend winners at T1024/16 credits by first-call and by repeat-call ROI."""
    pool = [r for r in rows if r['case']['stage'] == 'screen' and r['case']['wl'] == wl and r['case']['arm'] == arm]
    assert len(pool) == len(KCS), (wl, arm, len(pool))
    first = min(pool, key=lambda r: (r['first_us'], r['case']['kc']))['case']['kc']
    repeat = min(pool, key=lambda r: (r['repeat_us'], r['case']['kc']))['case']['kc']
    return sorted({first, repeat})


def target_cases(wl, t, selected):
    return [case(f'target{t}', wl, arm, t, kc) for arm in ARMS for kc in selected[arm]]
