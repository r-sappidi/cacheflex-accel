"""Read-only saved-run attribution, with instruction-byte verification per PC."""
from pathlib import Path
import json
import re
import subprocess
import analyze_attribution as a

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/cacheflex_flash_attention'


def words(path):
    text = subprocess.check_output(['aarch64-linux-gnu-objdump', '-d', str(path)], text=True)
    return {int(pc, 16): word for pc, word in re.findall(r'^\s*([0-9a-f]+):\s+([0-9a-f]{8})\s', text, re.M)}


def analyze(run):
    cmd = json.loads((run/'command.json').read_text())
    guest = Path(cmd[cmd.index('--cmd')+1])
    debug = guest.with_name('guest.debug')
    source = guest.with_name('driver.cpp').read_text().splitlines()
    markers = [(i, re.search(r'CF_PHASE\((\w+)\)', line)[1])
               for i, line in enumerate(source, 1) if 'CF_PHASE(' in line and '#define' not in line]
    def phase(frames, shift=0):
        names = ' '.join(f for f, _ in frames)
        for key, label in [('cf_wide_qk', 'QK'), ('cache_qk', 'QK'),
                           ('cf_pv16', 'PV_GEMM'), ('sve_cache_pv', 'PV_GEMM'),
                           ('sve_spm_pv', 'PV_GEMM'), ('exp', 'softmax_exp'),
                           ('pack_A', 'Q_or_P_packing'), ('scale_f32', 'output_rescale'),
                           ('acc_f16', 'PV_accumulate'), ('barrier', 'synchronization'),
                           ('futex', 'synchronization'), ('stage_kv', 'KV_staging')]:
            if key in names:
                return label
        for _, loc in frames:
            match = re.search(r'/driver.cpp:(\d+)', loc)
            if match:
                line = int(match[1])
                before = [(n, label) for n, label in markers if n <= line]
                if before:
                    return before[-1][1]
        return 'unresolved_or_shared'
    a.phase = phase
    result = a.analyze(run, debug)
    actual, expected = words(guest), words(debug)
    for row in result['pcs']:
        pc = int(row['pc'], 16)
        assert pc in actual and actual[pc] == expected[pc], (guest, hex(pc))
    result['verification'] = 'All profiled PC instruction words match executed guest and debug ELF; source markers from its generated driver.'
    return result


def main():
    out = BASE/'q_resident_analysis_20260917'
    out.mkdir()
    cases = dict(cache=BASE/'single_head_20260915/runs/cache_t16384_br32_bc128_g2_p1',
                 v_spm=BASE/'single_head_20260915/runs/spm_t16384_br16_bc128_g4_p1',
                 q_spm=BASE/'q_getx_fix_20260916_v3/t16384')
    for name, run in cases.items():
        result = analyze(run)
        (out/(name+'.json')).write_text(json.dumps(result, indent=2))
        print(name, json.dumps({k: dict(mean_core_us=v['attributed_cycles']*.0004/8,
                                      uops=v.get('pcRetiredMicroops', 0))
                               for k,v in result['phases'].items()}), flush=True)


if __name__ == '__main__':
    main()
