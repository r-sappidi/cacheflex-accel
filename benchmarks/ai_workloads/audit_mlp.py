"""Read-only audit of historical full-FFN and fused-projection measurements."""
import configparser
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
NAMES=['ffn_cache_t64_d512_f1536','ffn_spm_t64_d512_f1536',
       'ffn_spm_matched_swiglu_t64_d512_f1536']+[
       f'real_operators/diffusion_ffn_m{m}_{v}' for m in (128,512) for v in ('baseline','spm')]
rows=[]
for name in NAMES:
    path=ROOT/'results/ai_workloads'/name
    cfg=configparser.ConfigParser(interpolation=None,strict=False);cfg.read(path/'config.ini')
    log=(path/'run.log').read_text();stats=(path/'stats.txt').read_text()
    cpus=[s for s in cfg.sections() if re.fullmatch(r'system.cpu\d*',s)]
    policy=[cfg[s]['type'] for s in cfg.sections() if re.fullmatch(r'system.ruby.l1_cntrl\d+.cache.replacement_policy',s)]
    rows.append(dict(name=name,cores=len(cpus),spm_load_ports=[cfg[c]['spmloadports'] for c in cpus],
                     private_l2_policy=policy,stats_dumps=stats.count('Begin Simulation Statistics'),
                     sim_ticks=re.findall(r'^simTicks\s+(\d+)',stats,re.M),
                     pass_records=[l for l in log.splitlines() if ' PASS ' in l],
                     roi_markers=[l for l in log.splitlines() if 'GEMM_ROI' in l],
                     commands=[cfg[s]['cmd'] for s in cfg.sections() if s.endswith('workload') and 'cmd' in cfg[s]],
                     fu_counts={s:cfg[s]['count'] for s in cfg.sections() if s.startswith(cpus[0]+'.fuPool.FUList') and 'count' in cfg[s]}))
print(json.dumps(rows,indent=2))
