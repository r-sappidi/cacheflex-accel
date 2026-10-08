"""Read-only saved-run analysis; instruction-verified DWARF, no new simulations.

ROB-head attribution is not a causal latency decomposition. pcNotExecuted is
a subset of pcNoRetire, not an independent stall reason or memory miss count.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
import analyze_attribution as a


def phase(frames,shift=0):
    names=' '.join(f for f,_ in frames)
    for needle,label in [('cf_pv16','PV_microkernel'),('cf_wide_qk','QK_microkernel'),
        ('cf_vector_pack_A','Q_P_packing'),('pack_A_fp16','Q_P_packing'),
        ('expf','scalar_exp'),('with_errnof','scalar_exp'),
        ('scale_f32','output_rescale'),('acc_f16','PV_output_accumulate'),
        ('sve_exp_f32','vector_exp'),('stage_kv','V_staging'),
        ('pthread_barrier','synchronization'),('futex','synchronization')]:
        if needle in names:return label
    for _,loc in frames:
        m=re.search(r'/driver.cpp:(\d+)',loc)
        if not m:continue
        n=int(m[1])
        if 362<=n<=385:return 'V_staging_control'
        if 386<=n<=422:return 'QK_other'
        if 423<=n<=464:return 'softmax_rescale_other'
        if 465<=n<=507:return 'PV_other'
        if 508<=n<=532:return 'phase_loop_control'
        if 533<=n<=537:return 'final_normalize_sync'
    if any(n in names for n in ('memset','memcpy','memmove')):return 'memory_helpers'
    return 'other'


def instruction_bytes(data,va):
    ph=struct.unpack_from('<Q',data,32)[0];size,n=struct.unpack_from('<HH',data,54)
    for i in range(n):
        typ,flags,off,addr,_,filesz,_,_=struct.unpack_from('<IIQQQQQQ',data,ph+i*size)
        if typ==1 and addr<=va and va+4<=addr+filesz:return data[off+va-addr:off+va-addr+4]
    raise ValueError(hex(va))


def main():
    out=Path(sys.argv[1]);out.mkdir()
    root=ROOT/'results/cacheflex_flash_attention/llc_retain_20260915/runs'
    builds=ROOT/'results/cacheflex_flash_attention/pv_improvements_20260913/builds'
    allrows={}
    for name,directory,build in [('cache','cache_t16384_retain0_small0_old0','fce06cc784b2725b'),
        ('spm_off','spm_t16384_retain0_small0_old0','ea1c7516d839b578'),
        ('spm_on','spm_t16384_retain1_small0_old0','ea1c7516d839b578')]:
        run=root/directory;assert (run/'accepted.json').exists()
        cmd=json.loads((run/'command.json').read_text());guest=Path(cmd[cmd.index('--cmd')+1])
        debug=builds/build/'guest.debug'
        assert guest.read_bytes()==(builds/build/'guest.export-safe').read_bytes()
        a.phase=phase
        result=a.analyze(run,debug)
        gd,dd=guest.read_bytes(),debug.read_bytes()
        for pc in result['pcs']:
            addr=int(pc['pc'],16)
            assert instruction_bytes(gd,addr)==instruction_bytes(dd,addr),('debug opcode mismatch',name,pc)
        result['guest_sha256']=hashlib.sha256(gd).hexdigest()
        result['debug_sha256']=hashlib.sha256(dd).hexdigest()
        # Also retain a mnemonic breakdown, not just broad instruction classes.
        mnemonics=defaultdict(lambda:defaultdict(int))
        for pc in result['pcs']:
            key=pc['phase']+' / '+pc['instruction'].split()[0]
            for m in ('attributed_cycles','pcNoRetireCycles','pcNotExecutedCycles','pcRetiredMicroops'):
                mnemonics[key][m]+=pc.get(m,0)
        result['phase_mnemonics']=dict(mnemonics)
        lookup={int(p['pc'],16):p['phase'] for p in result['pcs']}
        cores=defaultdict(lambda:defaultdict(int))
        for line in (run/'stats.txt').read_text().split('End Simulation Statistics')[0].splitlines():
            w=line.split()
            if len(w)<2:continue
            m=re.fullmatch(r'system.cpu(\d+).commit.(pcHeadCycles|pcSleepHeadCycles)::(\d+)',w[0])
            if m:cores[m[1]][lookup[int(m[3])]]+=int(w[1])
        result['per_core_attributed_cycles']=dict(cores)
        (out/(name+'.json')).write_text(json.dumps(result,indent=2))
        allrows[name]=result
        print(name,result['traffic']['roi_us'],flush=True)
        for p,r in sorted(result['phases'].items(),key=lambda x:-x[1]['attributed_cycles']):
            print(p,'mean_core_us',round(r['attributed_cycles']/20000,3),
                  'no_retire_us',round(r.get('pcNoRetireCycles',0)/20000,3),
                  'not_exec_us',round(r.get('pcNotExecutedCycles',0)/20000,3),flush=True)
    (out/'summary.json').write_text(json.dumps({n:dict(roi_us=r['traffic']['roi_us'],phases=r['phases'],phase_classes=r['phase_classes'],phase_mnemonics=r['phase_mnemonics']) for n,r in allrows.items()},indent=2))


if __name__=='__main__':main()
