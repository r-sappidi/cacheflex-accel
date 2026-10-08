"""Non-intrusive ROB-head attribution and Garnet route accounting.

Phase bins are static DWARF/source attribution, not causal stall-cycle savings.
Shared packing/libc helpers are deliberately not assigned to an invented caller.
Run on the FIRST stats epoch and an independently byte-verified debug side ELF.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess

from analyze_traffic import analyze as traffic


def symbols(elf, pcs):
    output = subprocess.check_output(
        ['aarch64-linux-gnu-addr2line', '-a', '-f', '-C', '-i', '-e', str(elf)],
        input=''.join(hex(p)+'\n' for p in pcs), text=True)
    result = {}
    for block in re.split(r'(?m)^0x', output)[1:]:
        lines = block.strip().splitlines()
        result[int(lines[0], 16)] = list(zip(lines[1::2], lines[2::2]))
    return result


def instructions(elf):
    output = subprocess.check_output(['aarch64-linux-gnu-objdump', '-d', str(elf)], text=True)
    result = {}
    for line in output.splitlines():
        match = re.match(r'\s*([0-9a-f]+):\s+([0-9a-f]{8})\s+(\S+)(.*)', line)
        if not match:
            continue
        pc, word, op, args = match.groups()
        word = int(word, 16)
        if word & 0xfff0e000 == 0xa5f08000:
            op = 'SPMLD'
        elif word >> 24 == 0xff and word & 0xffff != 0x0110:
            op = ['SPMLDR', 'SPMSTR', 'SPMCP', 'SPMWB'][(word >> 22) & 3]
        if op in ('SPMLD', 'SPMLDR'):
            kind = 'spm_load'
        elif op.startswith(('ld', 'prf')):
            kind = 'cache_load_or_prefetch'
        elif op.startswith(('st', 'SPM')):
            kind = 'store_or_spm_command'
        elif op.startswith('f') or (re.search(r'\bz\d+\.', args) and op not in ('.inst', '.word')):
            kind = 'fp_vector_compute_or_control'
        else:
            kind = 'integer_control_other'
        result[int(pc, 16)] = dict(instruction=op+args, instruction_class=kind)
    return result


def phase(frames, cache_line_shift=0):
    joined = ' '.join(f for f, _ in frames)
    if 'clock_gettime' in joined or 'system_clock::now' in joined:
        return 'legacy_timer_overhead'
    if 'pack_A_fp16' in joined:
        return 'packing_Q_or_P'
    if 'sve_spm_mla' in joined or 'sve_mla_8x3VL' in joined:
        return 'QK'
    if 'sve_cache_pv' in joined or 'sve_spm_pv' in joined:
        return 'PV'
    if 'expf' in joined or 'with_errnof' in joined:
        return 'softmax_rescale'
    if 'pthread_barrier' in joined or 'futex' in joined:
        return 'synchronization'
    # Innermost explicit source frame first; it retains lambda phase identity.
    for _, location in frames:
        match = re.search(r'([^/]+):(\d+)', location)
        if not match:
            continue
        file, line = match[1], int(match[2])
        if file == 'cache_flash_attn.cpp' and line > 568:
            line -= cache_line_shift
        if file == 'spm_multicore.cpp':
            if 64 <= line <= 163 or 241 <= line <= 250 or 262 <= line <= 264:
                return 'stage_K'
            if 270 <= line <= 284:
                return 'QK'
            if 285 <= line <= 326:
                return 'softmax_rescale'
            if 327 <= line <= 337:
                return 'PV_pack_accumulate'
            if line == 359:
                return 'final_normalize'
            if 235 <= line <= 240:
                return 'query_pack_initialize'
        if file == 'cache_flash_attn.cpp':
            if 596 <= line <= 620:
                return 'QK'
            if 623 <= line <= 658:
                return 'softmax_rescale'
            if 660 <= line <= 676:
                return 'PV_pack_accumulate'
            if 679 <= line <= 698:
                return 'final_normalize'
            if 578 <= line <= 585:
                return 'query_pack_initialize'
        if file == 'pv_kernel_8x64.hpp':
            return 'PV'
    if 'memset' in joined or 'memcpy' in joined or 'memmove' in joined:
        return 'memory_helpers_shared'
    if 'scale_f32' in joined:
        return 'softmax_rescale'
    if 'acc_f16' in joined:
        return 'PV_pack_accumulate'
    if 'transpose_K' in joined or 'prepack_B' in joined:
        return 'prepare_K'
    return 'other_control_or_unresolved'


def analyze(path, elf, cache_line_shift=0):
    txt = (path/'stats.txt').read_text().split('End Simulation Statistics')[0]
    scalars = {}
    for line in txt.splitlines():
        fields = line.split()
        if len(fields) > 1:
            try:
                scalars[fields[0]] = float(fields[1])
            except ValueError:
                pass
    hist = defaultdict(lambda: defaultdict(int))
    for key, value in scalars.items():
        match = re.fullmatch(r'system.cpu(\d+).commit.(pcHeadCycles|pcSleepHeadCycles|pcNoRetireCycles|pcNotExecutedCycles|pcRetiredMicroops)::(\d+)', key)
        if match:
            hist[int(match[3])][match[2]] += int(value)
    assert hist, 'No PC histogram samples'
    frames = symbols(elf, sorted(hist))
    disasm = instructions(elf)
    phases = defaultdict(lambda: defaultdict(int))
    phase_classes = defaultdict(lambda: defaultdict(int))
    pc_rows = []
    for pc, counts in hist.items():
        row = dict(pc=hex(pc), phase=phase(frames[pc], cache_line_shift), frames=frames[pc],
                   **disasm.get(pc, dict(instruction='unknown', instruction_class='unknown')), **counts)
        row['attributed_cycles'] = counts.get('pcHeadCycles', 0) + counts.get('pcSleepHeadCycles', 0)
        pc_rows.append(row)
        for metric, count in counts.items():
            phases[row['phase']][metric] += count
            phase_classes[row['phase']+' / '+row['instruction_class']][metric] += count
    observed = sum(v for k, v in scalars.items() if k.endswith('.commit.pcObservedCycles'))
    empty = sum(v for k, v in scalars.items() if k.endswith('.commit.pcEmptyRobCycles'))
    head = sum(x.get('pcHeadCycles', 0) for x in hist.values())
    assert head + empty == observed, (head, empty, observed)
    for counts in hist.values():
        assert counts.get('pcNotExecutedCycles', 0) <= counts.get('pcNoRetireCycles', 0) <= counts.get('pcHeadCycles', 0)
    for row in phases.values():
        row['attributed_cycles'] = row.get('pcHeadCycles', 0) + row.get('pcSleepHeadCycles', 0)
        row['observed_cycle_percent'] = 100 * row.get('pcHeadCycles', 0) / observed
    routes = defaultdict(dict)
    metrics = ['packets', 'flits', 'bytes', 'received', 'network_ticks', 'queue_ticks']
    for key, val in scalars.items():
        match = re.fullmatch(r'system.ruby.network.message_(\w+)::(.+)', key)
        if match and match[2] != 'total':
            routes[match[2]][match[1]] = val
    tr = traffic(path)
    # Fixed paper core: 400 ticks/cycle. Include inactive/unobserved core time
    # explicitly rather than silently renormalizing hot PCs to 100% of ROI.
    roi_cycles = 8 * scalars['simTicks'] / 400
    sleeping = sum(v for k, v in scalars.items() if k.endswith('.commit.pcSleepCycles'))
    suspended = sum(v for k, v in scalars.items() if k.endswith('.commit.pcSuspendedCycles'))
    sleep_empty = sum(v for k, v in scalars.items() if k.endswith('.commit.pcSleepEmptyCycles'))
    sleep_head = sum(x.get('pcSleepHeadCycles', 0) for x in hist.values())
    assert sleep_head + sleep_empty + suspended == sleeping
    if sleeping:
        # Tick-event and wakeup accounting can differ from elapsed clock time
        # by a few cycles. Keep the signed residual visible; reject a material
        # gap rather than silently renormalizing attribution.
        assert abs(roi_cycles - observed - sleeping) <= max(64, roi_cycles*1e-5), (roi_cycles, observed, sleeping)
    for row in phases.values():
        row['roi_core_cycle_percent'] = 100 * row['attributed_cycles'] / roi_cycles
    assert sum(r.get('packets', 0) for r in routes.values()) == tr['packets_injected::total']
    assert sum(r.get('flits', 0) for r in routes.values()) == tr['flits_injected::total']
    assert sum(r.get('received', 0) for r in routes.values()) == scalars['system.ruby.network.packets_received::total']
    for metric, original in [('network_ticks', 'packet_network_latency'),
                             ('queue_ticks', 'packet_queueing_latency')]:
        match = re.search(r'^system.ruby.network\.'+original+r'\s+\|(.+)$', txt, re.M)
        assert match, original
        total = sum(float(x.split()[0]) for x in match[1].split('|'))
        assert sum(r.get(metric, 0) for r in routes.values()) == total, metric
    assert not any('unknown' in name and row.get('packets', 0) for name, row in routes.items())
    summary = defaultdict(lambda: defaultdict(float))
    for route, row in routes.items():
        # This matched protocol/config uses 8-byte control and 72-byte data
        # messages. Expose data-bearing writebacks without assuming every PUTX
        # necessarily carries data.
        packets = row.get('packets', 0)
        data_packets = (row.get('bytes', 0) - 8*packets) / 64
        assert 0 <= data_packets <= packets and data_packets.is_integer()
        row['data_packets'] = data_packets
        row['control_packets'] = packets-data_packets
        src, dst, vnet, msg = route.split('__')
        key = re.sub(r'\d+$', '', src)+' -> '+re.sub(r'\d+$', '', dst)+' '+msg
        for metric in metrics + ['data_packets', 'control_packets']:
            summary[key][metric] += row.get(metric, 0)
    return dict(path=str(path), traffic=tr, observed_core_cycles=observed,
                roi_core_cycles=roi_cycles, inactive_or_unobserved_cycles=roi_cycles-observed-sleeping,
                sleep_head_cycles=sleep_head, suspended_cycles=suspended, sleep_empty_cycles=sleep_empty,
                empty_rob_cycles=empty, phases=dict(phases), phase_classes=dict(phase_classes),
                messages=dict(summary), routes=dict(routes),
                pcs=sorted(pc_rows, key=lambda r: r['attributed_cycles'], reverse=True))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('result', type=Path)
    ap.add_argument('debug_elf', type=Path)
    ap.add_argument('--cache-line-shift', type=int, default=0)
    args = ap.parse_args()
    print(json.dumps(analyze(args.result, args.debug_elf, args.cache_line_shift), indent=2))
