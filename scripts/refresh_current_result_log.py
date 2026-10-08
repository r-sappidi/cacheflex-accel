"""Refresh discovery log of validator-accepted results; no simulations or grading."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'FA-53': 'results/cacheflex_flash_attention/winner_profile_20261007',
    'AI-13': 'results/ai_workloads/qwen_mshr16_20261005',
    'FA-45-cache': 'results/cacheflex_flash_attention/software_confirm_20261005/cache',
    'FA-45-cache_stage': 'results/cacheflex_flash_attention/software_confirm_20261005/cache_stage',
    'FA-45-burst': 'results/cacheflex_flash_attention/software_confirm_20261005/burst',
    'FA-45-stage': 'results/cacheflex_flash_attention/software_confirm_20261005/stage',
    'FA-45-paced': 'results/cacheflex_flash_attention/software_confirm_20261005/paced',
    'FA-45-combined': 'results/cacheflex_flash_attention/software_confirm_20261005/combined',

    'FA-44-cache': 'results/cacheflex_flash_attention/paced_staging_20261004/cache',
    'FA-44-burst': 'results/cacheflex_flash_attention/paced_staging_20261004/burst',
    'FA-44-paced': 'results/cacheflex_flash_attention/paced_staging_20261004/paced',
    'FA-44-overlap': 'results/cacheflex_flash_attention/paced_staging_20261004/overlap',
    'FA-44-dbburst': 'results/cacheflex_flash_attention/paced_staging_20261004/dbburst',
    'FA-43-cache_buf8': 'results/cacheflex_flash_attention/network_relief_20261004/cache_buf8',
    'FA-43-spm_buf8': 'results/cacheflex_flash_attention/network_relief_20261004/spm_buf8',
    'FA-43-cache_buf16': 'results/cacheflex_flash_attention/network_relief_20261004/cache_buf16',
    'FA-43-spm_buf16': 'results/cacheflex_flash_attention/network_relief_20261004/spm_buf16',
    'FA-42': 'results/cacheflex_flash_attention/packet_diag_20261004/parallel_v1',
    'FA-41': 'results/ordered_fa_followup_20261005/llama',
    'FA-40': 'results/ordered_fa_followup_20261005/fa40',
    'FA-39-4K': 'results/cacheflex_flash_attention/bottleneck_20261004/targets_continuation_v1',
    'FA-39-1K': 'results/cacheflex_flash_attention/bottleneck_20261004/screening_t1024_v1',
    'FA-39': 'results/cacheflex_flash_attention/bottleneck_20261004/experiment_v1',
    'FA-38': 'results/cacheflex_flash_attention/ring_async_20261003/experiment_v1',
    'FA-37': 'results/cacheflex_flash_attention/dsb_diag_20261003/experiment_v1',
    'FA-36': 'results/cacheflex_flash_attention/v_only_ring_20261003/experiment_v1',
    'FA-35': 'results/retained_protocol_restart_20261002/fa_matrix_spm_only',
    'L3-10': 'results/retained_protocol_restart_20261002/long_context_spm_only',
    'FA-34': 'results/cacheflex_flash_attention/optimistic_ring_20261001/experiment_retained_v1',
}

def main():
    # FA-46 arm queues are created by the bounded overnight scheduler.
    for arm in ('cache','spm','drained','kpack','ppack','pv_oracle','pv_traffic','pv64','pv128','pv256','pv_l3_budget'):
        for suffix in ('','_target'):
            relative='results/cacheflex_flash_attention/ncc_probe_20261005_v5/'+arm+suffix
            if (ROOT/relative/'status.json').exists(): SOURCES['FA-46-'+arm+suffix]=relative
    for arm in ('cache','spm','block128','async128','block256','async256'):
        relative='results/cacheflex_flash_attention/pv_async_20261005/'+arm
        if (ROOT/relative/'status.json').exists(): SOURCES['FA-47-'+arm]=relative
    for arm in ('spm_off','async_off','spm','async','no_probe','zero_service','no_probe_zero','less_fence','no_probe_less_fence','inert','spm_inert'):
        relative='results/cacheflex_flash_attention/pv_diag_20261005/'+arm
        if (ROOT/relative/'status.json').exists(): SOURCES['FA-48-'+arm]=relative
    for arm in ('cache_off','spm_off','cache','spm'):
        relative='results/cacheflex_flash_attention/phase_profile_20261005/'+arm
        if (ROOT/relative/'status.json').exists(): SOURCES['FA-49-'+arm]=relative
    if not (ROOT/SOURCES['AI-13']/'status.json').exists(): SOURCES.pop('AI-13')
    for lane in range(4):
        relative='results/cacheflex_flash_attention/softmax_opt_20261005/lane'+str(lane)
        if (ROOT/relative/'status.json').exists(): SOURCES['FA-50-lane'+str(lane)]=relative
    if (ROOT/'results/ai_workloads/qwen_mshr16_neighbors_20261005/status.json').exists(): SOURCES['AI-13b']='results/ai_workloads/qwen_mshr16_neighbors_20261005'
    if (ROOT/'results/cacheflex_flash_attention/softmax_tune_20261005/status.json').exists(): SOURCES['FA-51']='results/cacheflex_flash_attention/softmax_tune_20261005'
    if (ROOT/'results/cacheflex_flash_attention/softmax_tune_20261005/followup/status.json').exists(): SOURCES['FA-51b']='results/cacheflex_flash_attention/softmax_tune_20261005/followup'
    for m in ('qwen','llama'):
        rel='results/cacheflex_flash_attention/softmax_tune_20261005/fa51c_'+m
        if (ROOT/rel/'status.json').exists(): SOURCES['FA-51c-'+m]=rel
    if (ROOT/'results/cacheflex_flash_attention/q_only_20261006/status.json').exists(): SOURCES['FA-52']='results/cacheflex_flash_attention/q_only_20261006'
    rows, queues, issues = [], [], []
    for ledger, relative in SOURCES.items():
        root = ROOT / relative
        state = json.loads((root / 'status.json').read_text())
        queues.append(dict(ledger=ledger, path=relative, stage=state['stage'],
                           updated=state.get('updated'), active=state.get('active', {}),
                           pending=state.get('pending', [])))
        for row in state.get('completed', []):
            path = Path(row.get('path', str(root / 'runs' / row['name'])))
            marker = path / 'accepted.json'
            if not marker.exists():
                issues.append(f'{ledger} {row["name"]}: missing acceptance marker {marker}')
                continue
            data = marker.read_bytes()
            accepted = json.loads(data)
            if accepted.get('roi_us') != row.get('roi_us'):
                issues.append(f'{ledger} {row["name"]}: status/marker timing mismatch')
                continue
            rows.append(dict(ledger=ledger, name=row['name'], case=row.get('case'),
                             roi_us=row.get('roi_us'), evidence=str(marker.relative_to(ROOT)),
                             acceptance_sha256=hashlib.sha256(data).hexdigest()))
    result = dict(updated=datetime.now(timezone.utc).isoformat(), queues=queues,
                  accepted_records=rows, issues=issues)
    out = ROOT / 'docs/experiments'
    (out / 'current_results.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['Current validator-accepted result log', result['updated'], '',
             'Read LEDGER.md first: its caveats and invalidations take precedence.',
             'This supplements historical inventory; it does not revalidate raw arithmetic.',
             'Cases include correctness gates and targets. Timings are not all comparable.',
             'No global optimum, completed tuning, or paper-readiness claim is implied.', '']
    for q in queues:
        lines.append(f"{q['ledger']}: {q['stage']} | {q['path']}/status.json")
    for r in rows:
        lines += ['', f"{r['ledger']} {r['name']} | ROI us: {r['roi_us']}",
                  f"  Configuration: {json.dumps(r['case'], sort_keys=True)}",
                  f"  Evidence: {r['evidence']}"]
    lines += ['', 'Record discrepancies:'] + (issues or ['None.'])
    (out / 'CURRENT_RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print(f'{len(rows)} accepted records; {len(issues)} discrepancies; {len(queues)} queues')

if __name__ == '__main__':
    main()
