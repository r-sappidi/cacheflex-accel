"""Freeze/build AI-12 without changing any existing experiment input."""
import os,sys,json,shutil,subprocess
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc')
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'benchmarks/ai_workloads')]
from build_access_timing import save,sha
from repair_screen_export import patch
BASE=ROOT/'results/ai_workloads/qwen_remainder_20260929'

def main():
    budget=json.loads((ROOT/'results/parallel_budget_20260929/budget.json').read_text())
    os.sched_setaffinity(0,budget['cpu_set'])
    out=BASE/'guest_v1';out.mkdir(parents=True)
    shutil.copy2(ROOT/'benchmarks/ai_workloads/qwen_remainder.cpp',out/'driver.cpp')
    build='''set -euo pipefail
source /data2/rsappidi/cacheflex_mc/setup_spm_env.sh >/dev/null
out="$1"
"$CROSS_CXX" -std=c++17 -O3 -ffp-contract=off -march=armv8.2-a+sve+fp16 -mbranch-protection=none -DGEM5 -DGEM5_SE -I"$M5_INCLUDE" -static -pthread "$out/driver.cpp" "$M5OP_OBJ" -lm -o "$out/guest"
'''
    (out/'build.sh').write_text(build)
    with (out/'build.log').open('w') as log:subprocess.run(['bash',str(out/'build.sh'),str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
    patch(out/'guest')
    dis=subprocess.check_output(['aarch64-linux-gnu-objdump','-d',str(out/'guest')],text=True)
    (out/'disassembly.txt').write_text(dis)
    assert all(x in dis for x in ('ld1h','st1h','fcvt','faddv')),'expected SVE conversion/reduction missing'
    save(out/'frozen.json',{str(p):sha(p) for p in (out/'guest',out/'driver.cpp',out/'build.sh')})
    save(out/'status.json',dict(stage='built_runtime_gates_pending',same_elf_both_arms=True,sha256=sha(out/'guest')))
if __name__=='__main__':main()
