"""Build/run tiny wide-SPM-store bit-pattern roundtrip on the frozen model."""
import json
import os
from pathlib import Path
import subprocess
from build_access_timing import ROOT,save
from run_fa_output_resident import BUILD,ENV,GUESTS

OUT=GUESTS/'roundtrip'
if __name__=='__main__':
    OUT.mkdir()
    here=ROOT/'benchmarks/cacheflex_flash_attention'
    script=f'''set -e
source "{ROOT}/setup_spm_env.sh" >/dev/null
"$CROSS_CXX" -std=c++17 -O3 -g -march=armv8.2-a+sve+fp16 -DCACHEFLEX_OUTPUT_SPM=1 -DCACHEFLEX_FLASH_RICH_INPUT=1 -I"$M5_INCLUDE" -include "{here}/correct_fa_conversions.hpp" -S "{here}/tests/test_output_resident.cpp" -o "{OUT}/guest.s"
python3 "$SPM_COMPILER" "{OUT}/guest.s" "{OUT}/guest.enc.s"
"$CROSS_CXX" -static "{OUT}/guest.enc.s" "$M5OP_OBJ" -lm -o "{OUT}/guest"
'''
    (OUT/'build.sh').write_text(script);subprocess.run(['bash',str(OUT/'build.sh')],check=True)
    old=ROOT/'results/cacheflex_flash_attention/output_resident_v2_20260924/runs/output_spm_t512'
    cmd=json.loads((old/'command.json').read_text())
    cmd=[f'--outdir={OUT}' if a.startswith('--outdir=') else a for a in cmd if not a.startswith('--debug-')]
    cmd[cmd.index('--cmd')+1]=str(OUT/'guest')
    for i,a in enumerate(cmd):
        if a.startswith('system.cpu[0].workload[0].cmd='):cmd[i]="system.cpu[0].workload[0].cmd=['roundtrip']"
    save(OUT/'command.json',cmd)
    with (OUT/'run.log').open('w') as f:
        p=subprocess.run(cmd,env=ENV,stdout=f,stderr=subprocess.STDOUT)
    save(OUT/'result.json',dict(exitcode=p.returncode))
