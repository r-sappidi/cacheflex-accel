"""FA-53: map the exact FA-51c executable; no guest/simulator rebuild."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
R = Path('/data2/rsappidi/cacheflex_mc')
H = Path(__file__).resolve().parent
OUT = R / 'results/cacheflex_flash_attention/winner_profile_20261007'
G = R / 'results/cacheflex_flash_attention/softmax_tune_20261005/guest_v1'
CONTROL = R / 'results/cacheflex_flash_attention/softmax_tune_20261005/fa51c_qwen/runs/t4096_spm_bc256_r256_f15_cr16'
B = R / 'results/cacheflex_flash_attention/phase_profile_20261005/build_v1'
NAMES = {0:'Startup barrier',1:'K packing',2:'K-pack barrier',3:'Q packing and task initialization',4:'V staging',5:'QK and row-max epilogue',6:'Softmax and online state',7:'P packing',11:'Output accumulation/rescaling epilogue',12:'SPM release',21:'Loop/control/dispatch',22:'PV dot product and function setup',23:'Final output normalization',24:'Final barrier',25:'Reporting/ROI exit'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')

def mapping(disasm):
    # Exact binary addresses, checked against disassembly. The map changes host
    # observation only; labels and instruction stream remain those of the winner.
    rows=[
      (0x406bf0,1,'kv_phase_pack_k: start cooperative K packing'),
      (0x406cf8,2,'K-pack barrier argument setup, before call'),
      (0x406d00,21,'return from K-pack barrier'),
      (0x406e20,3,'kv_phase_pack_q: Q packing and task initialization'),
      (0x406f24,21,'KV loop dispatch after Q packing/init'),
      (0x40904c,4,'kv_phase_stage: copy loop'),
      (0x406f44,21,'return from copy loop; inner-loop setup'),
      (0x406fd8,5,'kv_phase_qk: boundary-block QK path'),
      (0x40826c,5,'kv_phase_qk: nonboundary QK/max path'),
      (0x407088,6,'kv_phase_softmax'),
      (0x407394,21,'kv_phase_pv: packing/PV dispatch'),
      (0x403120,7,'cf_vector_pack_A entry'),
      (0x4077f0,21,'return from cf_vector_pack_A'),
      (0x4019c0,22,'SPM paired no-alpha PV function entry'),
      (0x402980,22,'SPM paired alpha PV function entry'),
      (0x401a8c,11,'no-alpha loop exit: conversion and output accumulation'),
      (0x402a4c,11,'alpha loop exit: conversion, rescaling and output accumulation'),
      (0x408cc0,21,'return from alpha PV'),
      (0x408d34,21,'return from no-alpha PV'),
      (0x4078ac,23,'kv_phase_normalize'),
      (0x40913c,21,'normalization/drain complete, advance task'),
      (0x409460,12,'kv_phase_release'),
      (0x409498,21,'release complete, record work counts'),
      (0x4094ac,24,'kv_phase_final_wait: final arrival'),
      (0x4094b4,25,'final barrier return / ROI exit'),
    ]
    ins={int(a,16):(word,op.strip()) for a,word,op in re.findall(r'^\s*([0-9a-f]+):\s+([0-9a-f]{8})\s+(.+)$',disasm,re.M)}
    symbols={name:int(a,16) for a,name in re.findall(r'^([0-9a-f]+) <([^>]+)>:',disasm,re.M)}
    assert len({a for a,_,_ in rows})==len(rows)
    for a,_,why in rows:assert a in ins,(hex(a),why)
    for a in [0x4077f0,0x408cc0,0x408d34]:assert ins[a-4][1].startswith('bl\t'),hex(a)
    for a in [0x401a8c,0x402a4c]:
        assert ins[a-4][1].startswith('b.ne\t'),hex(a)
        assert ins[a][1].startswith('mov\tx7, x12'),hex(a)
    for name,a in [('pack_k_11258',0x406bf0),('pack_q_11591',0x406e20),('softmax_12081',0x407088),('final_wait_20174',0x4094ac)]:
        assert symbols['kv_phase_'+name]==a
    return [dict(pc=f'{a:016x}',phase=p,name=NAMES[p],reason=why,instruction=ins[a][1],encoding=ins[a][0]) for a,p,why in rows]

def main():
    OUT.mkdir(exist_ok=False);(OUT/'runs').mkdir()
    assert sha(B/'gem5.opt')==json.loads((B/'ready.json').read_text())['binary_sha256']
    disasm=subprocess.check_output(['aarch64-linux-gnu-objdump','-d','-C',str(G/'guest.export-safe')],text=True)
    rows=mapping(disasm)
    (OUT/'guest.disassembly.txt').write_text(disasm)
    (OUT/'phase_map.txt').write_text(''.join(f"{r['pc']} {r['phase']}\n" for r in rows))
    save(OUT/'phase_map_audit.json',dict(guest_sha256=sha(G/'guest.export-safe'),events=rows,phase_names=NAMES))
    save(OUT/'plan.json',dict(ledger='FA-53',control=str(CONTROL),case=json.loads((CONTROL/'case.json').read_text()),simulator=str(B/'gem5.opt'),guest=str(G/'guest.export-safe'),runs=1,acceptance='Exact accepted output and all non-host stats; full oracle/config/work/copy/resource/frozen-input checks plus phase partition, counts, request conservation',scope='Latest Qwen SPM winner only; no retuning; no claim of global optimum',hypothesis='Determine whether softmax/packing remain dominant on the latest winner before proposing an ingress LUT; full ROI must remain 8398.4216 us exactly'))
    print('Prepared',len(rows),'passive PC events; unchanged guest and existing FA-49 observer')
if __name__=='__main__':main()
