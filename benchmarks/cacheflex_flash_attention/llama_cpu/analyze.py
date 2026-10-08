"""Read only accepted measurements; classify dynamically retired SVE arithmetic."""
import json,re,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[3]/'results/cacheflex_flash_attention/llama_cpu_20260925'
ex=BASE/'experiment_v2'
for p in sorted((ex/'runs').glob('*/validated.json')):
    row=json.loads(p.read_text());mode=row['mode']
    function='';instructions={}
    for line in (BASE/mode/'guest.dis').read_text().splitlines():
        m=re.match(r'[0-9a-f]+ <(.*)>:',line)
        if m:function=m[1]
        m=re.match(r'\s*([0-9a-f]+):\s+[0-9a-f]+\s+(.*)',line)
        if m:instructions[int(m[1],16)]=(function,m[2])
    fp32_fma=0;tile_sve=0
    for event in json.loads((p.parent/'sve_retired.json').read_text()):
        pc=int(event['counter'].rsplit('::',1)[1]);func,ins=instructions[pc]
        if 'flash_attn_ext_tiled' in func:
            tile_sve+=event['count']
            if re.search(r'\bfmla\s+z\d+\.s',ins):fp32_fma+=event['count']
    print(json.dumps(dict(mode=mode,t=row['t'],roi_us=row['roi_us'],
        tiled_sve_retired=tile_sve,tiled_fp32_sve_fma_retired=fp32_fma,
        arithmetic=row['arithmetic']),indent=2))
