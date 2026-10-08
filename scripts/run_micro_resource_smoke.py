"""Run bridge/direct-access and cross-core snapshot regressions, no tuning."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks/cacheflex_flash_attention'))
from run_clean_controls import sha, save

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cacheflex_flash_attention/micro_resources_20260921'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--build',default='build_v2')
    parser.add_argument('--output',default='smoke_v2')
    args=parser.parse_args()
    build=OUT/args.build
    output=OUT/args.output
    output.mkdir()
    template=OUT/'experiment_v2/runs/spm_bounded_t512/command.json'
    base=json.loads(template.read_text())
    for label, guest, marker in [
        ('roundtrip',OUT/'roundtrip_guest/guest','SPM_ROUNDTRIP PASS errors=0'),
        ('release_getx',ROOT/'results/ruby_getx_race_guest_20260916/guest','SPM_RELEASE_GETX PASS errors=0')]:
        run=output/label;run.mkdir()
        cmd=[str(build/'gem5.opt')]+base[1:]
        cmd[2]=str(build/'configs/deprecated/example/bounded_se.py')
        cmd[cmd.index('--cmd')+1]=str(guest)
        cmd=[f'--outdir={run}' if x.startswith('--outdir=') else
             "system.cpu[0].workload[0].cmd=['spm-smoke']" if x.startswith('system.cpu[0].workload[0].cmd=') else
             x.replace('sve_vl_se=16','sve_vl_se=4') for x in cmd]
        save(run/'command.json',cmd)
        save(run/'inputs.json',{str(p):sha(p) for p in (build/'gem5.opt',guest,template,Path(__file__))})
        with (run/'run.log').open('w') as log:
            p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180,
                env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
        text=(run/'run.log').read_text()
        row=dict(returncode=p.returncode,passed=p.returncode==0 and marker in text)
        save(run/'result.json',row)
        print(label,row,flush=True)
        assert row['passed'],run

if __name__=='__main__': main()
