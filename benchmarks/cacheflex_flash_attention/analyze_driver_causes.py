"""Function-level corroboration for driver ablations; not causal phase sums."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import analyze_attribution as a

def function_bin(frames,shift=0):
    names=' '.join(f for f,_ in frames)
    for token,label in [('cf_vector_pack_A','explicit_SVE_Q_or_P_pack'),
                        ('pack_A_fp16','legacy_Q_or_P_pack'),
                        ('cache_qk','QK'),('cacheflex_qk','QK'),
                        ('spm_kernel_1vl','QK'),('cache_pv','PV_kernel'),
                        ('stage_kv','K_staging'),('acc_f16','PV_accumulate'),
                        ('scale_f32','output_rescale'),('exp','softmax_exp'),
                        ('barrier','barrier'),('futex','barrier'),
                        ('memset','memset'),('memcpy','memcpy')]:
        if token in names:return label
    return 'driver_or_other'

root=Path(sys.argv[1]).resolve()
rows=json.loads((root/'progress.json').read_text())
summary=[]
for r in rows:
    if r['t']!=1024 or r['observer'] or r['original'] or r['mask'] not in [0,1,2,3,7]:continue
    b=root/f"build_{r['variant']}_g{r['group']}_m{r['mask']}"
    with tempfile.TemporaryDirectory(prefix='cf-text-check-') as scratch:
        tmp=Path(scratch)
        for name in ['guest','guest.debug']:
            subprocess.run(['aarch64-linux-gnu-objcopy','--dump-section',f'.text={tmp/name}.text',str(b/name),str(tmp/name)],check=True)
        assert (tmp/'guest.text').read_bytes()==(tmp/'guest.debug.text').read_bytes()
    a.phase=function_bin
    result=a.analyze(Path(r['path']),b/'guest.debug')
    target=Path(r['path'])/'function_attribution.json';target.write_text(json.dumps(result,indent=2))
    summary.append(dict(variant=r['variant'],mask=r['mask'],roi_us=r['counters']['roi_us'],
        bins={k:dict(attributed_core_us_div8=v['attributed_cycles']*.0004/8,
                    retired_microops=v.get('pcRetiredMicroops',0)) for k,v in result['phases'].items()}))
(root/'function_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
