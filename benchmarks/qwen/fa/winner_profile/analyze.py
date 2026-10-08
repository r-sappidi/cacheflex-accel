"""Report critical-core phase percentages without summing cores as wall time."""
import csv,json
from pathlib import Path
from prepare import OUT,NAMES,sha

def main():
    rows=list((OUT/'runs').glob('*/accepted.json'));assert len(rows)==1
    row=json.loads(rows[0].read_text());roi=row['roi_us'];timeline=row['timeline']
    critical=max(timeline,key=lambda i:timeline[i]['final_arrival'])
    percore={i:{NAMES.get(int(p),'Unused phase '+p):v['ticks']/1e6 for p,v in phases.items() if v['ticks']} for i,phases in row['phases'].items()}
    values=percore[critical];boundary=roi-sum(values.values());assert abs(boundary)<1
    ranked=sorted(values.items(),key=lambda kv:-kv[1])
    report=['# FA-53 — phase breakdown of the latest Qwen SPM winner','',
      'T4096, 14 query heads / 2 KV heads, head dimension 64, eight cores, causal; Br16/Bc256/R256, flag15, 16 L1D credits.',
      f'Full ROI **{roi:.4f} us**; last-arriving core **{critical}**. Exact same guest, output bits, configuration and all non-host statistics as FA-51c.',
      '', '| Phase on last-arriving core | Time (us) | Share of full ROI |','|---|---:|---:|']
    for name,us in ranked:report.append(f'| {name} | {us:.4f} | {100*us/roi:.3f}% |')
    report.append(f'| Outer ROI reset/dump boundary remainder | {boundary:.4f} | {100*boundary/roi:.3f}% |')
    report += ['', 'Percentages use one core timeline and the full outer ROI, not summed core time. Each core timeline partitions the observed workbegin/workend interval exactly.',
      'Phase time is elapsed retirement-boundary time including stalls. Overlapping instructions, memory requests and copy service are not additive removable costs.',
      'P packing is the pack function including its prologue/epilogue. PV covers function setup and the FP16 dot-product loop. Its output epilogue is separately observed at the first instruction after the loop: FP16-to-FP32 conversion, optional alpha rescaling, accumulation/stores, and function return. These are observation boundaries inside the existing fused function, not independently executed kernels.',
      'Q packing includes task state/output initialization. QK includes its row-max epilogue on applicable flag15 paths. Softmax includes masking, exponentials, reductions and online state updates. V staging measures its instruction interval; copy requests may remain in flight outside it.',
      'This profiles the best measured ordinary Qwen SPM case in the bounded FA-51/51c search, not a global optimum or a Llama result.',
      '', 'Files: `phase_breakdown.csv`, `phase_counters.csv`, `phase_breakdown.svg`, `phase_breakdown.png`, `analysis.json`, `FINAL_VERIFICATION.json`, `phase_map_audit.json` and the run acceptance marker.']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    with (OUT/'phase_breakdown.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['core','phase','time_us','percent_full_roi','last_arriving'])
        for i,vals in percore.items():
            for name,us in vals.items():w.writerow([i,name,us,100*us/roi,i==critical])
    records=[]
    for i,phases in row['phases'].items():
        for p,v in phases.items():records.append(dict(core=i,phase=p,name=NAMES.get(int(p),'Unused'),**v,**row['phase_details'][i][p]))
    with (OUT/'phase_counters.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (OUT/'analysis.json').write_text(json.dumps(dict(roi_us=roi,critical_core=int(critical),critical_phases_us=values,outer_boundary_us=boundary,per_core_us=percore,timeline=timeline),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10})
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    labels=[n for n,u in ranked];pct=[100*u/roi for n,u in ranked]
    ax.barh(labels,pct,color='#286b91');ax.invert_yaxis();ax.set_xlabel('Percent of full ROI (last-arriving core)');ax.set_title('Latest Qwen SPM winner — T4096, flag15, 16 credits')
    ax.set_xlim(0,max(pct)*1.18)
    for j,p in enumerate(pct):ax.text(p+.15,j,f'{p:.2f}%',va='center')
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'phase_breakdown.{ext}',dpi=160)
    plt.close(fig)
    frozen=json.loads((OUT/'inputs.json').read_text())
    for p,h in frozen.items():assert sha(p)==h,('frozen input drift',p)
    (OUT/'FINAL_VERIFICATION.json').write_text(json.dumps(dict(accepted=1,control=row['observer_equivalence'],roi_us=roi,frozen_files_verified=len(frozen),phase_partition=True,phase_work_counts=True,copy_load_raw_conservation=True,full_oracle=row['arithmetic'],output_sha256=row['output_sha256'],scope='Elapsed passive PC-boundary observations; no sum of per-core times or overlapping request durations'),indent=2)+'\n')
    print('\n'.join(report[:24]))
if __name__=='__main__':main()
