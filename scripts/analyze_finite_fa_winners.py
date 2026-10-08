"""Read-only ROI/PC audit of the completed finite-profile FA search."""
import json
import re
import subprocess
from pathlib import Path

from run_access_timing import stats

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/cacheflex_flash_attention/finite_model_20260926'
SEARCH = BASE / 'analytical_experiment_v5/experiment/runs'
PILOTS = BASE / 'experiment_v5/runs'
OUT = BASE / 'winner_mechanism_audit_20260927'
ELF = ROOT / 'results/cacheflex_flash_attention/model_tuning_20260926/build/guest.export-safe'
CASES = {
    'cache_R768': SEARCH / 't16384_q_cache_br16_bc512_r768',
    'spm_R768': SEARCH / 't16384_q_spm_br16_bc512_r768',
    'spm_R1024': SEARCH / 't16384_q_spm_br16_bc512_r1024',
    'spm_R256_pilot': PILOTS / 't16384_kv0_spm1',
    'persistent_V_pilot': PILOTS / 't16384_kv1_spm1',
}
PCS = {
    0x401980: 'QK packed-K vector load (loop 1)',
    0x4019e0: 'QK packed-K vector load (loop 2)',
    0x40124c: 'Probability packer stack reload, NOT probability operand load',
    0x4019f0: 'QK packed-Q replicated load',
    0x401d08: 'FP32 output accumulator load before softmax rescale',
    0x402680: 'SPM staging slot-address integer division, NOT SPMCP',
    0x402690: 'SPMCP first half of V row',
    0x40269c: 'SPMCP second half of V row',
    0x402300: 'Cache PV V vector load (cache branch only)',
    0x401f04: 'SPM PV V vector load (SPM branch only)',
}


def total(s, pattern):
    return sum(v for k, v in s.items() if re.fullmatch(pattern, k))


def main():
    OUT.mkdir(exist_ok=True)
    records = {}
    for name, path in CASES.items():
        validation = next((path / f for f in ('validated.json', 'accepted.json')
                           if (path / f).exists()), None)
        assert validation, path
        s = stats(path)[0]  # First dump is ROI; never include verification/export.
        responses = total(s, r'system.cpu\d+.spmLsq0.spmLoadResponses')
        cycles = total(s, r'system.cpu\d+.spmLsq0.spmLoadServiceCycles')
        pc_counts = {hex(pc): total(s, rf'system.cpu\d+.commit.pcNotExecutedCycles::{pc}')
                     for pc in PCS}
        records[name] = dict(
            source=str(path), validation=str(validation),
            roi_ms=s['simTicks'] / 1e9,
            copies=total(s, r'.*sequencer.m_spmcpRequestsCompleted'),
            packets=s['system.ruby.network.packets_injected::total'],
            dram_read_mib=total(s, r'system.mem_ctrls\d+.dram.bytesRead::total') / 2**20,
            l1d_misses=total(s, r'system.ruby.l0_cntrl\d+.Dcache.m_demand_misses'),
            private_l2_misses=total(s, r'system.ruby.l1_cntrl\d+.cache.m_demand_misses'),
            mean_core_copy_busy_ms=total(s, r'.*sequencer.m_spmcpBusyCycles') / 8 / 2.5e6,
            mean_spm_load_service_cycles=cycles / responses if responses else None,
            spm_raw_blocks=total(s, r'system.cpu\d+.spmLsq0.spmRawBlockEvents'),
            global_admission_rejects=total(s, r'.*sequencer.m_globalAdmissionFullRejects'),
            not_executed_pc_samples=total(s, r'system.cpu\d+.commit.pcNotExecutedCycles::\d+'),
            pc_counts=pc_counts,
        )
    cache, spm = records['cache_R768'], records['spm_R768']
    result = dict(records=records, pc_meanings={hex(k): v for k, v in PCS.items()},
                  speedup=cache['roi_ms']/spm['roi_ms'],
                  runtime_reduction_pct=100*(1-spm['roi_ms']/cache['roi_ms']))
    (OUT / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    disasm = subprocess.check_output(['aarch64-linux-gnu-objdump', '-d', str(ELF)], text=True)
    lines = disasm.splitlines()
    extracts = []
    for pc, meaning in PCS.items():
        i = next(i for i, line in enumerate(lines) if line.lstrip().startswith(f'{pc:x}:'))
        extracts.extend([f'\n# {pc:#x}: {meaning}', *lines[max(0,i-2):i+3]])
    (OUT / 'instruction_evidence.txt').write_text('\n'.join(extracts) + '\n')
    report = [
        '# Finite-profile FA: SPM-specific mechanism directions', '',
        '8 cores, single head, T16384/d64/VL16; 512 KiB private L2/core, 4 MiB LLC. '
        'Only the corrected finite-resource profile is compared. First ROI includes packing/staging/reduction. '
        'All selected runs have saved numerical/configuration acceptance. Search is bounded, not an exhaustive optimum.', '',
        f'Matched measured winners: cache {cache["roi_ms"]:.7f} ms, SPM {spm["roi_ms"]:.7f} ms; '
        f'{result["speedup"]:.6f}x, {result["runtime_reduction_pct"]:.3f}% less runtime. '
        'Both use query partition, Br16/Bc512/R768 and the same ELF.', '',
        '| Case | ROI ms | SPMCP count | Network packets | DRAM read MiB |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, r in records.items():
        report.append(f'| {name} | {r["roi_ms"]:.7f} | {r["copies"]:.0f} | {r["packets"]:.0f} | {r["dram_read_mib"]:.3f} |')
    report += ['', '## Exact instruction attribution', '',
               'Counts below sum unexecuted ROB-head samples over eight cores. They overlap across cores; '
               'they are NOT an additive runtime breakdown, miss counts, or predictions of removable time.', '',
               '| Instruction | Cache samples | SPM samples |', '|---|---:|---:|']
    for pc, meaning in PCS.items():
        report.append(f'| `{pc:#x}` {meaning} | {cache["pc_counts"][hex(pc)]:.0f} | {spm["pc_counts"][hex(pc)]:.0f} |')
    report += ['', '## Interpretation and next interventions', '',
        '1. **Protect the remaining packed-K operand, not just V.** The current driver always calls '
        '`cf_wide_qk_cache`; only PV uses SPM. Two QK K-load PCs alone contribute 39.56 million '
        'SPM unexecuted-head samples (33.3% of this sample population). V-load samples fall from '
        '17.63 million on the cache branch to 1,154 on the SPM branch, while K-load samples remain '
        'essentially unchanged. This supports a remaining QK operand-delivery target, not proof of a '
        'particular cache-miss cause or a 33% removable runtime fraction.', '',
        'First compare V-only, K-only, and K+V at fixed Br/Bc/R with numerical checks. An existing '
        '`cf_wide_qk_spm` kernel is present but unused by this driver. Each Bc512 K or V panel occupies '
        '64 KiB; both together need 128 KiB actual slots. Respect wide-load way layout, slot disjointness '
        'and six-way eligibility; claiming extra slots reduces coherent capacity. Then retune R for each '
        'placement. This is a software residency change, not yet a novel hardware contribution. If '
        'K residency helps but its extra fills limit the gain, that supplies a concrete target for '
        'bounded early tile filling/reuse-aware staging rather than another generic latency reduction.', '',
        '2. **Consider direct producer-to-SPM probability layout only with a matched fused cache control.** '
        'Softmax currently writes row-major probabilities, then a separate packing pass prepares PV. '
        'A major packer PC is a stack reload (14.23 million samples), so these are NOT all probability '
        'cache misses. Produce the PV layout directly for both backends first; evaluate SPM-native '
        'stores/consumption separately. A tile-layout store mechanism is a hardware hypothesis only '
        'if ordinary software fusion cannot capture its benefit.', '',
        '3. **Output-state residency is a secondary hypothesis.** The exact FP32 O-rescale load has '
        '5.75 million samples in SPM versus 1.91 million in cache. Protect a bounded active O tile, '
        'rather than pinning the entire query group: R768 O alone is 192 KiB/core. A targeted latency '
        'intervention on these addresses could test runtime sensitivity before designing support. '
        'The PC difference alone does not establish evictions or prove that pinning O helps.', '',
        '## Lower priority and limitations', '',
        'The winning SPMLD service averages exactly seven cycles and reports zero SPM RAW blocks. '
        'Its direct PV load almost never appears as an unexecuted ROB head. That weakens the case '
        'for another pure-SPMLD latency/retry optimization in this configuration.', '',
        'R1024 performs fewer copies than R768 but runs slower and generates more packets/DRAM reads. '
        'Persistent KV-partition V has 24x fewer copies than the winner yet is slower; mapping changes '
        'also change computation, query processing and reduction. Neither comparison isolates copying '
        'causally. The mean-core copy-busy interval is only about 0.383 ms in the winner, but that is '
        'neither a causal runtime fraction nor a bound on all staging-induced interference.', '',
        'No additional simulations, hardware edits, or running-job changes were made for this audit. '
        'Use fixed-workload matched interventions and end-to-end ROI deltas to validate each direction. '
        'Do not claim a new mechanism, paper-ready optimum, or fundamental FA limit from these counters.', '',
        'Reproduce: `python3 scripts/analyze_finite_fa_winners.py`. '
        '`metrics.json` lists exact result paths; `instruction_evidence.txt` records disassembly.',
    ]
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n')
    print(OUT / 'REPORT.md')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
