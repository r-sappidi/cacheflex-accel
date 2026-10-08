"""Accepted-only FA-33 interpretation; records null/regressive results as well."""
import argparse,json,time,os
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc');BASE=ROOT/'results/cacheflex_flash_attention/qwen_qo_ring_20260930';OUT=BASE/'experiment_v3'
def run():
 s=json.loads((OUT/'status.json').read_text());rows=s['completed'];accepted=[]
 for r in rows:
  assert json.loads((Path(r['path'])/'accepted.json').read_text())==r
  assert r['arithmetic']['errors']==r['arithmetic']['nonfinite']==r['commands']==0
  accepted.append(r)
 effects=[];lines=['# FA-33: combined residency effects','',f"Status: {s['stage']}",'','Only accepted full-output/config/resource checks. T256 is correctness only.','', '| T | Placement | Full ROI us | Speedup vs same-ELF legacy V | Speedup vs strongest existing V |','|---:|---|---:|---:|---:|']
 for t in (1024,4096):
  arms={r['case']['mode']:r for r in accepted if r['case']['t']==t}
  if 1 not in arms:continue
  old={1024:818.6768,4096:9473.3184}[t]
  for m,r in sorted(arms.items()):
   row=dict(t=t,mode=m,roi_us=r['roi_us'],same_elf_legacy_v_speedup=arms[1]['roi_us']/r['roi_us'],strongest_prior_v_speedup=old/r['roi_us']);effects.append(row)
   lines.append(f"| {t} | {r['name']} | {r['roi_us']:.4f} | {row['same_elf_legacy_v_speedup']:.6f} | {row['strongest_prior_v_speedup']:.6f} |")
 lines+=['','Capacity fit is not a performance guarantee. Q has the existing 64B-load/TBL broadcast adapter; its cost must not be misattributed to the ring. O initialization uses real coherent-to-SPM copies.','FA-34 optimistic/finite ring remains a dependent implementation stage, not an accepted result or currently running simulator.']
 (OUT/'EFFECTS.md').write_text('\n'.join(lines)+'\n');(OUT/'accepted_effects.json').write_text(json.dumps(effects,indent=2))
 if s['stage'] in ('completed_validated','failed') and not (OUT/'ledger_recorded.json').exists():
  p=ROOT/'docs/experiments/LEDGER.md';text=p.read_text();marker='**FA-33 experiment_v3 accepted-only outcome:**'
  if marker not in text:
   text+='\n'+marker+f" stage={s['stage']}, accepted={len(accepted)}. "
   text+='All accepted subsets retained; failed/cancelled cases excluded. '
   text+='; '.join(f"T{e['t']} mode{e['mode']} {e['roi_us']:.4f}us ({e['strongest_prior_v_speedup']:.6f}x priorV)" for e in effects)
   text+='\n[Effects](../../results/cacheflex_flash_attention/qwen_qo_ring_20260930/experiment_v3/EFFECTS.md). No ring performance claim.\n';p.write_text(text)
  (OUT/'ledger_recorded.json').write_text(json.dumps(dict(time=time.time(),stage=s['stage'],accepted=len(accepted))))
 return s['stage']
if __name__=='__main__':
 os.sched_setaffinity(0,{44});p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 while True:
  stage=run()
  if not a.watch or stage in ('failed','completed_validated'):break
  time.sleep(30)
