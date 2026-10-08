#!/usr/bin/env python3
"""Create an evidence-only mentor handoff; never launches simulations."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/cacheflex_flash_attention/staging_trace_20260917'
OUT = ROOT / 'mentor_evidence'
OUT.mkdir(exist_ok=False)

def copy(src, name):
    dst = OUT / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

for name in ['ANALYSIS.md', 'admission_analysis.json', 'manifest.json',
             'observer_equivalence.json', 'parser_repair.json', 'plan.json',
             'window.json', 'status.json', 'target_context.log']:
    copy(ROOT / name, name)
for part in ['trace', 'scan']:
    for name in ['command.json', 'config.ini', 'stats.txt', 'run.log',
                 'exit.json', 'diagnostic.trace']:
        src = ROOT / part / name
        if src.exists():
            copy(src, f'{part}/{name}')
for name in ['analyze_fa_staging_trace.py', 'run_fa_staging_trace.py',
             'test_fa_staging_trace.py', 'package_fa_staging_evidence.py']:
    copy(REPO / 'scripts' / name, f'scripts/{name}')
prior = Path(json.loads((ROOT / 'plan.json').read_text())['prior'])
for name in ['accepted.json', 'command.json', 'config.ini', 'run.log']:
    copy(prior / name, f'parent_completed_run/{name}')
guest = REPO / 'results/cacheflex_flash_attention/long_context_pv_20260914/spm.guest'
copy(guest, 'guest/spm.guest')
asm = subprocess.check_output(['aarch64-linux-gnu-objdump', '-d',
                              '--start-address=0x401db0', '--stop-address=0x401e94',
                              str(guest)], text=True)
(OUT / 'staging_disassembly.txt').write_text(asm)
# Explicitly label present-day source rather than implying a historical snapshot.
for name in ['gem5/src/mem/ruby/system/Sequencer.cc',
             'gem5/src/cpu/o3/lsq_unit.cc', 'gem5/src/cpu/o3/lsq.cc']:
    copy(REPO / name, 'current_source_reference/' + name)
data = json.loads((ROOT / 'admission_analysis.json').read_text())
lines = ['# Raw evidence excerpts (original trace line numbers)', '']
intervals = data['attempts'][0]['intervals']
addresses = {x['line'] for x in intervals}
with (ROOT / 'trace/diagnostic.trace').open('rb') as stream:
    for n, raw in enumerate(stream, 1):
        text = raw.decode('utf-8', errors='backslashreplace').rstrip()
        target = '[sn:158524822]' in text or 'CFBOUND pc=0x401e14 sn=158524822' in text
        relevant_seq = ('Seq' in text and '   0 ' in text and
                        (any(a in text for a in addresses) or '0xa79c8' in text))
        if target or relevant_seq:
            lines.append(f'{n}: {text}')
(OUT / 'evidence_excerpt.txt').write_text('\n'.join(lines)+'\n')
table = ['| SPMCP source line | Accepted tick | Completion tick |',
         '|---|---:|---:|']
for item in sorted(intervals, key=lambda x: x['start']):
    table.append(f"| {item['line']} | {item['start']} | {item['end']} |")
readme = '''# CacheFlex FA staging: evidence handoff

## Claim supported by this trace

In one selected V-staging episode, SPMCP fetches fill all 16 shared Ruby
sequencer request entries and prevent an otherwise-ready ordinary stack load
from being admitted for 111 CPU cycles. After admission, Ruby completes in
1 cycle and the CPU sees completion in 2 cycles. This is direct evidence of
SPMCP-to-coherent admission interference, not a long-latency stack DRAM fetch.

It does NOT establish the fraction of total FA runtime caused by this mechanism,
that it is the largest whole-kernel bottleneck, or the speedup from fixing it.

## Run identity and configuration

- Experiment: staging_trace_20260917, frozen retained-SPM multi-head FA.
- T=16384; Hq=8, Hkv=1 (eight query heads share K/V); d=64.
- Q: 8 x 16384 x 64; K and V: each 16384 x 64. FP16 operands.
- Br=16, Bc=512, query group=16 blocks (256 query rows); full-width PV.
- V-only staging, single buffer, SPM offset256; scalar SPMCP, no DMA,
  multicast or batch copies. Four eligible private-L2 ways.
- 8 ARM out-of-order cores, 2.5GHz, SVE VL16 (2048 bits).
- Per core: 64KiB L1I + 64KiB L1D, 4-way; 512KiB private L2, 8-way.
- Shared LLC: 4MiB total across 8 banks, 16-way. 64-byte lines.
- Private L2 forced LRU; prefetch enabled; LLC retention enabled.
- Ruby naming differs from architectural naming: l0=L1, l1=private L2,
  l2=shared LLC. This is NOT the 64KiB-private-L2 single-head configuration.
- ROB128, IQ80, LQ32, SQ48. Fetch/decode/rename/commit width5;
  dispatch/issue/writeback width8. Ordinary and SPM ports each 2 load/1 store.
- Shared Ruby sequencer maximum outstanding requests: 16 per core.
- Garnet Mesh_XY, 2 rows, routing algorithm1, router/link latency1,
  128-bit links, 4 VCs/vnet; system clock1.5GHz, memory2GiB.
- Full resolved hardware/DRAM parameters are in trace/config.ini.
  Actual guest argv is ['cacheflex-fa','16384','64','16','512']; the later
  workload.cmd override supersedes the legacy --options argument.

## How the episode was selected

Two deterministic prefix replays stop at tick30045715600 (ROI start26045715600
plus4ms). Boundary-only scan selects the largest gap between successive
retirements of PC0x401e14 within staging, excluding compute between panels.
The detailed trace spans ticks27178567200..27179012800 (445.6ns).
It is deliberately worst-gap selection in that prefix, NOT random sampling.
43,546 non-host numeric counters match exactly between observer passes.
The partial traces do not perform a new full-output correctness check; the
completed parent run's acceptance record is supplied separately.

## Exact load timeline

Core0, PC0x401e14: `ldr x3,[sp,#488]`; dynamic SN158524822.
Physical address0xa79c8, cache line0xa79c0. One cycle=400 simulator ticks.

| Event | Tick | Cycles after first send attempt |
|---|---:|---:|
| Operand ready, no memory dependency, issued | 27178766400 | -1 |
| First send rejected | 27178766800 | 0 |
| First observed unready head-of-ROB stall | 27178768400 | 4 |
| Send finally accepted | 27178811200 | 111 |
| Ruby Done | 27178811600 | 112 |
| CPU completion / dependents wake | 27178812000 | 113 |
| Retirement | 27178812800 | 115 |

Seven failed send attempts occurred at ticks27178766800, 27178776000,
27178781600, 27178785200, 27178792400, 27178796000, 27178805600.
Each has 16 paired SPMCP intervals in flight at end of tick. At successful
admission the observed set is 15 SPMCP plus the ordinary load.

## Sixteen copies occupying the queue at the first rejection

Every interval below starts before27178766800 and ends afterward. Distinct
source lines remove same-line alias ambiguity for this saturation proof.

''' + '\n'.join(table) + '''

Matched Begin/Done intervals are generally a lower bound (pre-window requests
and aliases can be missing), but sixteen observed distinct requests suffice
to establish saturation of a sixteen-entry queue at the first rejection.
End-of-tick snapshots alone do not resolve same-tick event ordering.

Prefix statistics further show core0 m_spmcpMaxOutstanding=16,
m_spmcpBufferFullRejects=31899, m_spmcpBlockedLineRejects=0.
Rejected-attempt counts are NOT unique requests or stall-cycle counts.

## Files and reproduction

- ANALYSIS.md: interpretation, limitations and proposed controlled tests.
- evidence_excerpt.txt: line-numbered raw instruction/protocol evidence.
- trace/diagnostic.trace: entire detailed trace, unmodified (contains binary bytes).
- admission_analysis.json: each failed/accepted attempt and matched intervals.
- trace/config.ini, command.json, stats.txt, run.log: full resolved run metadata.
- scan/: first-pass data; observer_equivalence.json: non-host counter gate.
- staging_disassembly.txt and guest/spm.guest: actual instruction bytes.
- parent_completed_run/accepted.json: parent output/config/counter acceptance.
- manifest.json: original simulator/guest/config hashes, absolute source paths.
- parser_repair.json: recovery audit for invalid-UTF8 postprocessing; trace unchanged.
- current_source_reference/: PRESENT-DAY explanatory source snapshots, not a
  certified source tree of the frozen simulator. Binary identity comes from the
  original manifest. Frozen simulator/build dependencies are not bundled.
- SHA256SUMS: checksums of every supplied evidence file.

From the extracted bundle:

```
sha256sum -c SHA256SUMS
python3 scripts/analyze_fa_staging_trace.py .
```

The second command reproduces interval reconstruction from the raw trace and
rewrites admission_analysis.json. The simulator is not needed for analysis.
To rerun simulations, the original repository, frozen simulator and configuration
dependencies referenced by command.json are required; this is an evidence bundle,
not a standalone simulator distribution.

## Recommended wording to share

“Our detailed gem5 trace demonstrates an admission bottleneck during V staging:
16 outstanding SPMCP fetches fill the shared per-core Ruby sequencer, delaying
an ordinary stack load by111 cycles before admission even though it completes
within2 CPU cycles afterward. This stalls retirement at the ROB head. We have
identified this mechanism in a selected multi-head FA episode; its aggregate
runtime contribution and the benefit of reserved coherent request capacity
still require controlled experiments.”
'''
(OUT / 'README.md').write_text(readme)
checks = []
for path in sorted(OUT.rglob('*')):
    if path.is_file():
        checks.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(OUT)}')
(OUT / 'SHA256SUMS').write_text('\n'.join(checks)+'\n')
archive = ROOT / 'mentor_evidence.tar.gz'
with tarfile.open(archive, 'x:gz') as tar:
    tar.add(OUT, arcname='mentor_evidence')
print(f'{archive}: {archive.stat().st_size} bytes, {len(checks)} checksummed files')
