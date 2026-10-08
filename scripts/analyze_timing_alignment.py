"""Accept only complete same-ELF timing probes; retain explicit NoC residual."""
import json
from build_access_timing import ROOT,sha,save
from run_timing_fa import config_audit

OUT=ROOT/'results/cacheflex_flash_attention/timing_alignment_20260922'

def main():
    cases=[('cache',4,64),('cache',256,64),('cache',2048,64),
           ('spm',4,64),('spm',4,128),('spm',4,256),
           ('cache_stream',4,256),('spm_stream',4,256)]
    rows=[];files=[]
    for mode,kib,width in cases:
        stream=mode.endswith('_stream')
        paths={m:OUT/('streams_v3' if stream else ('reference_v3' if m=='classic' else 'translated_v3'))/
            f'{m}_{mode}_{kib}k_b{width}' for m in ('classic','ruby')}
        vals={}
        for m,p in paths.items():
            r=json.loads((p/'result.json').read_text());assert r['passed'] and len(r['epochs'])==3
            # Final two warm epochs, no first-epoch instruction cold-start.
            vals[m]=sum(x['cycles_per_load'] for x in r['epochs'][1:])/2
            files += [p/'command.json',p/'config.json',p/'stats.txt',p/'result.json']
            c=json.loads((p/'command.json').read_text())
            assert str(OUT/'guest_v4/guest') in c
        delta=vals['ruby']-vals['classic'];llc=(mode=='cache' and kib==2048)
        assert abs(delta)<(2.0 if llc else 0.03),(mode,kib,width,vals)
        if not stream:
            old=OUT/'old_same_guest'/f'ruby_{mode}_{kib}k_b{width}'
            config_audit(old,paths['ruby'])
            prior=json.loads((old/'result.json').read_text())
            vals['old_ruby']=sum(x['cycles_per_load'] for x in prior['epochs'][1:])/2
            files += [old/'result.json',paths['ruby']/'config_changes.json']
        rows.append(dict(mode=mode,KiB=kib,bytes=width,cycles_per_load=vals,
                         ruby_minus_classic=delta,network_residual=llc))
    profile=OUT/'configs/deprecated/example/micro_timing_se.py'
    files += [profile,OUT/'guest_v4/guest',OUT/'classic_build_v2/manifest.json',
              ROOT/'scripts/analyze_timing_alignment.py']
    record=dict(passed=True,scope='Isolated dependent hit latency and eight-chain throughput, VL16; NOT full multicore/NoC equivalence',
        profile_sha256=sha(profile),guest_sha256=sha(OUT/'guest_v4/guest'),rows=rows,
        tolerances='0.03 CPU cycles/load for local paths/throughput; LLC residual reported, sanity bound2 cycles',
        files={str(p):sha(p) for p in files})
    save(OUT/'calibration.json',record)
    lines=['# MICRO-derived Ruby timing verification','',
        'Same guest ELF, CPU and cache geometry; prefetch disabled only for isolated probes. '
        'CPU clock 2.5GHz. Final two of three warmed epochs are averaged. '
        'Dependent probe cycles/load include pointer extraction/loop overhead, not pure SRAM latency.',
        '', '| Probe | MICRO Classic | Old Ruby | Corrected Ruby |',
        '|---|---:|---:|---:|']
    for r in rows:
        v=r['cycles_per_load'];old=f'{v["old_ruby"]:.4f}' if 'old_ruby' in v else '—'
        lines.append(f'| {r["mode"]}, {r["KiB"]}KiB, {r["bytes"]}B | {v["classic"]:.4f} | {old} | {v["ruby"]:.4f} |')
    lines += ['',
        'MICRO source revision: 82ac0b9de1ed408ce5cd53067267c8a855b4eef3. '
        'Pristine simulator rebuilt locally; compiler/config/binary hashes retained under classic_build_v2.',
        '', '## Scope and residual differences','',
        '- L1, private-L2 and direct-SPM CPU-visible dependent latency and two eight-chain throughput probes match within the stated tolerance.',
        '- LLC service is35 CPU cycles. Ruby retains its fixed8-router mesh; Classic uses crossbars. The measured LLC transport residual is not fitted away.',
        '- Same storage, credits, NoC bandwidth/topology and guest kernels. Cache controllers now use the CPU clock; transport clocks remain unchanged.',
        '- Ordered local FIFOs clamp later short-latency arrivals behind earlier messages; no reordering or added queue storage.',
        '- Narrow-predicate SPM probes initialize all4 ways because MICRO sends a full-VL SPM packet. These are latency checks, not evidence of equal masked-access traffic.',
        '- SPMCP forwarding/install delays are source-derived, but not independently calibrated for every residency/coherence state. WB-credit lifetime and bank arbitration are unchanged.',
        '- Full FA validation/results: fa_v3/status.json. This is timing calibration, not reproduction of the published FA speedup.',
        '- fa_v1 post-ROI export failed on a stale functional-read filename. The established filename-refresh repair is restricted to post-ROI code; default controls require exact old ROI counters and output hashes.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
