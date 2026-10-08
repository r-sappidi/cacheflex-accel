"""Run the repaired frozen T16K case past its exact original failure tick.

Diagnostic only: reaching the tick limit is not full numerical acceptance.
"""
from pathlib import Path
import json,os,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/cacheflex_flash_attention'
old=BASE/'q_resident_20260915_v3'
new=BASE/'q_resident_20260915_v4'
out=BASE/'q_bounds_regression_20260915'
out.mkdir()
with tempfile.TemporaryDirectory() as tmp:
    blobs=[]
    for i,base in enumerate((old,new)):
        target=Path(tmp)/f'text{i}'
        subprocess.run(['aarch64-linux-gnu-objcopy','--dump-section',f'.text={target}',
                        str(base/'builds/spm_g4/guest.export-safe'),str(Path(tmp)/f'elf{i}')],check=True)
        blobs.append(target.read_bytes())
    assert len(blobs[0])==len(blobs[1])
    offsets=[i for i in range(0,len(blobs[0]),4) if blobs[0][i:i+4]!=blobs[1][i:i+4]]
    assert len(offsets)==6,offsets
    (out/'text_comparison.json').write_text(json.dumps(dict(size=len(blobs[0]),
        changed_instruction_offsets=offsets,meaning='Only six address-mask instructions changed'),indent=2))
cmd=json.loads((old/'runs/spm_t16384_br16_bc128_g4_p1/command.json').read_text())
cmd=[f'--outdir={out}' if x.startswith('--outdir=') else x for x in cmd]
cmd[cmd.index('--cmd')+1]=str(new/'builds/spm_g4/guest.export-safe')
cmd+=['--abs-max-tick=13000000000']
(out/'command.json').write_text(json.dumps(cmd,indent=2))
with (out/'run.log').open('w') as log:
    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
text=(out/'run.log').read_text()
assert 'simulate() limit reached' in text,text[-500:]
(out/'passed.json').write_text(json.dumps(dict(passed=True,stop_tick=13000000000,
    original_crash_tick=11628680000,full_numerical_validation=False),indent=2))
