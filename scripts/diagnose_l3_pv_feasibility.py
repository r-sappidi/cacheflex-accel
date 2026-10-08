"""MOD-11: evidence-anchored bounds and finite synthetic PV schedules, not gem5 ROI."""
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import resource
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/cacheflex_l3_spm/pv_feasibility_20260930'
TRACE = ROOT / 'results/cacheflex_l3_spm/diagnostics_20260929/experiment_v1/runs/t4096_direct_trace'
NEIGHBORS = ROOT / 'results/cacheflex_l3_spm/qwen_neighbors_20260929/experiment_v1/status.json'


def schedule(rows, lanes, prefetch, share, latency=35, fma_latency=4, bc=256, d=64):
    """One finite row group, P buffered, V streamed; no cross-key reassociation.

    Common bank port initiates a 64B transfer every 1/share cycles. V buffer
    slots stay occupied until their last FMA issue. This model grants pipelined
    SRAM reads and separate operand registers; it does NOT grant extra SRAM ports.
    P and O read upfront; O write after all dependent arithmetic completes.
    Includes FP16->FP32 conversion and nonfused recurrence at the old 16-lane rate.
    No CPU/NoC/translation/claim/descriptor/drain timing is inferred here.
    """
    step = 1 / share
    p_lines = math.ceil(rows * bc * 2 / 64)
    o_lines = math.ceil(rows * d * 4 / 64)
    port = (p_lines + o_lines) * step
    start = port + latency
    port = start
    issue_width = math.ceil(rows * d / lanes)
    key_ii = max(issue_width, fma_latency)
    done, issued, occupancy = [], [], []
    for k in range(bc):
        release = done[k-prefetch] if k >= prefetch else start
        load_start = max(port, release)
        port = load_start + math.ceil(d * 2 / 64) * step
        ready = port + latency
        compute_start = max(ready, issued[-1] + key_ii if issued else start)
        issued.append(compute_start)
        done.append(compute_start + issue_width)
        occupancy.append((load_start, done[-1]))
    compute_end = done[-1] + fma_latency
    recurrence = rows * (6 + 9)  # existing 64-element conversion/update, 16 lanes
    finish = max(port, compute_end + recurrence) + o_lines * step + latency
    peak = max(sum(a <= x < b for a, b in occupancy) for x, _ in occupancy)
    assert peak <= prefetch
    return dict(tile_cycles=finish, max_prefetch_occupancy=peak,
                internal_register_bytes=rows*bc*2 + rows*d*2 + rows*d*4 + prefetch*d*2,
                bank_read_write_bytes=(p_lines+2*o_lines+bc*math.ceil(d*2/64))*64,
                key_issue_cycles=key_ii)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    # A short single-process analysis, <=512MiB AS, sharing the existing CPU set.
    resource.setrlimit(resource.RLIMIT_AS, (512*1024**2,512*1024**2))
    os.sched_setaffinity(0, {44})
    OUT.mkdir(parents=True, exist_ok=True)
    resources = json.loads((ROOT/'results/parallel_memory_recovery_20260929/resources.json').read_text())
    assert time.time()-resources['updated'] < 120, 'Resource snapshot stale'
    assert resources['reserved_bytes']+512*1024**2 < resources['ram_limit_bytes']
    assert resources['cpu_pct_of_host']+100/os.cpu_count() < 20
    assert resources['server_headroom_after_full_user_budget'] > 64*1024**3
    source_paths = [TRACE/'accepted.json', TRACE/'stats.txt', TRACE/'config.json', NEIGHBORS,
                    Path(__file__), ROOT/'docs/experiments/L3_PV_EXTENSION_FEASIBILITY.md']
    accepted = json.loads((TRACE/'accepted.json').read_text())
    assert accepted['arithmetic']['errors']==0 and accepted['exact_prior_stats']
    cfg = json.loads((TRACE/'config.json').read_text())
    cycle_ns = cfg['system']['ruby']['clk_domain']['clock'][0]/1000
    assert cycle_ns == .5 and cfg['system']['ruby']['l2_cntrl0']['l3_spm_latency']==35
    # First stats block is the ROI, consistent with existing acceptance parser.
    stats = {}
    in_block = False
    for line in (TRACE/'stats.txt').read_text().splitlines():
        if 'Begin Simulation Statistics' in line:
            in_block = True
            continue
        if 'End Simulation Statistics' in line and in_block:
            break
        words = line.split()
        if in_block and len(words)>1:
            try: stats[words[0]]=float(words[1])
            except ValueError: pass
    commands = [int(stats[f'system.ruby.l2_cntrl{i}.L2cache.l3ReductionStarts']) for i in range(8)]
    assert sum(commands)==accepted['commands'] and max(commands)==64000
    state=json.loads(NEIGHBORS.read_text())
    assert state['stage']=='completed_validated'
    best={}
    for rec in state['completed']:
        if rec['case']['t'] != 4096: continue
        path=Path(rec['path'])/'accepted.json'
        assert json.loads(path.read_text())['roi_us']==rec['roi_us']
        arm=rec['case']['arm']
        if arm not in best or rec['roi_us']<best[arm]['roi_us']:best[arm]=rec
    for rec in best.values(): source_paths.append(Path(rec['path'])/'accepted.json')
    reference=best['private']['roi_us']
    target=reference*.90  # predeclared 10% latency-reduction screen, not user requirement
    rows=[]
    for lanes,g,prefetch,share,latency in itertools.product((16,32,64,128,256),(1,4,16),(1,4,16),(1,.5,.25),(35,9)):
        # 9-cycle placement is an explicit hypothetical local-L2 sensitivity,
        # not measured PV timing or a claim that its port is physically identical.
        s=schedule(g,lanes,prefetch,share,latency)
        jobs=max(commands)//g
        assert all(n%g==0 for n in commands)
        fmas=max(commands)*256*64
        compute_floor=fmas/lanes*cycle_ns/1000
        memory_floor=jobs*s['bank_read_write_bytes']/64/share*cycle_ns/1000
        service=jobs*s['tile_cycles']*cycle_ns/1000
        # Compulsory operand placement traffic. V is refilled once per R256 panel,
        # then reused across 256/g groups; P is written once per group.
        p_bytes=max(commands)*256*2
        v_bytes=max(commands)//256*256*64*2
        stage_bytes=p_bytes+v_bytes
        # This models destination SRAM writes only. Source/network/claim overhead
        # remains unknown and must be added in implementation.
        stage_floor=stage_bytes/64/share*cycle_ns/1000
        bound=max(compute_floor,memory_floor+stage_floor)
        # Deliberately serialized destination writes + synthetic engine schedule.
        # Not an upper/lower bound on actual complete FA, since CPU timing omitted.
        service_plus_staging=service+stage_floor
        rows.append(dict(lanes=lanes,reuse_rows=g,prefetch_keys=prefetch,bank_share=share,
            placement='L3' if latency==35 else 'L2_latency_sensitivity',read_latency=latency,
            compute_floor_us=compute_floor,port_floor_including_install_us=memory_floor+stage_floor,
            engine_lower_bound_us=bound,synthetic_service_us=service_plus_staging,
            residual_serial_budget_for_10pct_us=target-service_plus_staging,
            hard_reject_10pct=bound>=target,**s))
    (OUT/'candidates.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (OUT/'candidates.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    manifest={str(p):sha(p) for p in source_paths}
    (OUT/'inputs.json').write_text(json.dumps(dict(evidence_sha256=manifest,resources_before=resources,
        commands_per_bank=commands,cycle_ns=cycle_ns,private_roi_us=reference,
        direct_roi_us=accepted['roi_us'],screen_target_us=target,
        screen_definition='10% less latency than tuned private; hypothetical selection threshold',
        assumptions='Native FP16 FMA II1 latency4, sequential key order, finite registers/prefetch, pipelined shared64B port; no CPU schedule inferred.'),indent=2)+'\n')
    lines=['# MOD-11: PV-plus-recurrence feasibility diagnostics','',
        f'{len(rows)} analytical cases completed. These are synthetic service schedules, NOT measured FA speedups.',
        f'Accepted T4096 private {reference:.4f} us; direct reduction {accepted["roi_us"]:.4f} us.',
        f'Design screen: 10% less latency than private = {target:.4f} us. This is a selection criterion, not a prediction.',
        '', '## Representative L3 cases (16-key prefetch, 35-cycle reads)', '',
        '| Lanes/bank | Rows reusing V | Bank share | Engine+destination staging ms | Registers/bank KiB | Remaining serial budget ms |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['placement']=='L3' and r['prefetch_keys']==16 and r['lanes'] in (64,128,256) and r['reuse_rows'] in (1,16) and r['bank_share'] in (1,.5):
            lines.append(f"|{r['lanes']}|{r['reuse_rows']}|{r['bank_share']}|{r['synthetic_service_us']/1000:.3f}|{r['internal_register_bytes']/1024:.2f}|{r['residual_serial_budget_for_10pct_us']/1000:.3f}|")
    lines += ['', '## Interpretation and limits', '',
        '* MOD-10 already showed near-null full-FA sensitivity to recurrence lane count, read/reply latency and pipelining. L3-02/03/05 already implement retry isolation, compact contributions and direct PV epilogues; these are not new proposals.',
        '* This adds a finite operand schedule to the earlier MOD-10 PV arithmetic bound. Row reuse, prefetch depth and shared-port contention are new analytical dimensions. No accepted near-cache PV implementation exists.',
        '* Hard rejection uses only mandatory compute/port demand. Synthetic service includes finite prefetch, dependent-key FMA order, P/O reads, V streams, recurrence, O writes, and destination P/V installation. Positive residual budget does not establish that CPU QK/softmax, source/network transfers, translation, coherence, commands and drain fit.',
        '* Whole P group buffering plus FP16 PV and FP32 O registers are explicit NEW engine state. Persistent O and staged P/V use existing SPM capacity. No extra data SRAM or extra bank port is assumed. Register access/broadcast area, clock and energy are not calibrated.',
        '* V is assumed installed once per R256 panel and reused across row groups. For 16 rows, P is8KiB vs2KiB old contribution; V panel32KiB. Existing256B capture cannot carry these as a free command. Real staging/descriptor integration is required.',
        '* L2_latency_sensitivity changes only read delay35->9. It does not validate private-L2 port availability, capacity, ownership, clock or routing; do not rank physical placements from this alone.',
        '* Memory allocation bounds do not validate numerical implementation. Preserve sequential FP16 FMA and nonfused FP32 recurrence in any simulator prototype; no reassociation or precision substitution.',
        '* PC-head occupancy is NOT used as removable runtime. Current command trace covers startup only. CPU production cadence and steady-state/drain causality remain unmeasured.',
        '', '## Decision gates', '',
        '1. Reject configurations whose mandatory demand exceeds the target. Select a small Pareto set in lanes/registers/bandwidth; do not blindly widen the current recurrence engine.',
        '2. For candidates with margin, implement bounded PV command/staging and same-hardware engine-disabled controls. Verify full arithmetic, exact same-tiling bits, hazards, finite occupancy and coherent progress before full FA.',
        '3. Measure complete T4096 ROI versus tuned private/cache and current direct reduction, including all staging and drain; require >=10% lower latency than private for this study’s marked-improvement criterion. Retune affected tile parameters equally. Repeat at an accepted long-context size.',
        '4. A definitive architecture recommendation additionally needs clock/area feasibility. Analytical survival alone is not approval of a performance or physical-cost claim.',
        '', 'CSV/JSON retain every null/regressive candidate; inputs.json records source hashes and resource admission. Existing simulator queues and frozen inputs were untouched.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'status.json').write_text(json.dumps(dict(stage='completed_analytical_only',cases=len(rows),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,simulators_launched=0),indent=2)+'\n')
    print('\n'.join(lines[:25]))

if __name__=='__main__':main()
