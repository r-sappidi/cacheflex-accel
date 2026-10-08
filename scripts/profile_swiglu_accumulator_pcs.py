"""Frozen-ELF accumulator-PC selection and observed ROB-head attribution."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/ai_workloads/swiglu_model_pilot_v4_20260921'
OUT=ROOT/'results/ai_workloads/swiglu_accumulator_oracle_20260925'


def select(guest):
    asm=subprocess.check_output(['aarch64-linux-gnu-objdump','-d','-C',str(guest)],text=True)
    selected=[]; stores=[]; active=False
    for line in asm.splitlines():
        if re.match(r'^[0-9a-f]+ <',line):
            active='(anonymous namespace)::gemm(' in line
        if active:
            m=re.match(r'\s*([0-9a-f]+):\s+[0-9a-f]+\s+(ld1w|st1w)\s+(.*)',line)
            if m:
                (selected if m[2]=='ld1w' else stores).append(dict(pc=int(m[1],16),hex='0x'+m[1],instruction=m[2]+' '+m[3]))
    assert len(selected)==len(stores)==16,(guest,len(selected),len(stores))
    return selected,stores


def main():
    OUT.mkdir(exist_ok=True)
    data={}
    for arch in ('cache','spm'):
        guest=OLD/'builds/custom_g0'/arch
        loads,stores=select(guest)
        path=OLD/'runs'/f'pilot_{arch}_t2048_d2048_f8192_mp2_kc128_g0'
        blocks=(path/'stats.txt').read_text().split('Begin Simulation Statistics')[1:]
        rows=[]
        for block in blocks:
            if 'End Simulation Statistics' not in block:continue
            counts={key:dict(total=0,accumulator_load=0,accumulator_store=0) for key in ('pcNotExecutedCycles','pcNoRetireCycles','pcHeadCycles')}
            lp={x['pc'] for x in loads};sp={x['pc'] for x in stores}
            for line in block.split('End Simulation Statistics')[0].splitlines():
                w=line.split()
                if len(w)<2:continue
                m=re.fullmatch(r'system.cpu\d+.commit.(pcNotExecutedCycles|pcNoRetireCycles|pcHeadCycles)::(\d+)',w[0])
                if m:
                    c=counts[m[1]];value=int(w[1]);pc=int(m[2]);c['total']+=value
                    if pc in lp:c['accumulator_load']+=value
                    if pc in sp:c['accumulator_store']+=value
            rows.append(counts)
        data[arch]=dict(guest=str(guest),sha256=hashlib.sha256(guest.read_bytes()).hexdigest(),
            loads=loads,stores=stores,epochs=rows,
            warning='Sum of observed per-core ROB-head categories, NOT causal end-to-end percentages; downstream dependent arithmetic can hide load attribution.')
    (OUT/'pc_profile.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({a:v['epochs'] for a,v in data.items()},indent=2))


if __name__=='__main__':main()
