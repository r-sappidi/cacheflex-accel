"""Require unchanged ROI counters between old and repaired export guests."""
from pathlib import Path
import json
from queue_correct_best import key
from validate_attribution import read
from fa_export_patch import verify_dump

def main():
    base=Path(__file__).resolve().parents[2]/'results/cacheflex_flash_attention'
    repair=base/'optuna_export_repair_20260913';old=base/'optuna_20260912'
    rows=json.loads((repair/'summary.json').read_text())['completed'];comparisons=[]
    for r in rows:
        verify_dump(r)
        ident=key(dict(config=r['config'],br=r['br'],bc=r['bc'],t=1024,rich=True,rep=0))
        p=old/'runs'/ident
        if not (p/'stats.txt').exists():continue
        output,counters,_=read(p,16)
        assert output==r['output'],(r['case'],'guest output changed')
        delta={k:(counters.get(k),r['counters'].get(k)) for k in (set(counters)|set(r['counters']))-{'path'} if counters.get(k)!=r['counters'].get(k)}
        assert not delta,(r['case'],delta)
        comparisons.append(dict(case=r['case'],roi_us=r['roi_us'],output_identical=True,counters_identical=True))
    assert {r['case'] for r in comparisons}>={'failed_cache','cache_control','spm_control'}
    (repair/'roi_equivalence.json').write_text(json.dumps(comparisons,indent=2));print('ROI equivalence PASS',comparisons)

if __name__=='__main__':main()
