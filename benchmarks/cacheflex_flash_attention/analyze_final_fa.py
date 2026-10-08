"""Read-only winner analysis; verified sampled instructions/current source labels."""
from pathlib import Path
import json,re,struct
import analyze_attribution as a
from run_causal import audit
from validate_attribution import read

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/cacheflex_flash_attention'
RUNS=dict(cache=BASE/'optuna_20260912/runs/90670efef713dce4',
          spm=BASE/'best_corrected_20260912/runs/dc31c793e4b22029')

def phase(frames,shift=0):
    # These labels are line-checked against each winner's saved driver, not
    # inherited from historical K-staging drivers or shifted debug ELFs.
    for name,loc in frames:
        if 'driver.cpp:' in loc:
            n=int(re.search(r'driver.cpp:(\d+)',loc)[1])-shift
            for lo,hi,label in [(295,311,'prepare_K'),(333,349,'query_pack_initialize'),
               (350,384,'stage_V_control'),(385,419,'QK'),(420,462,'softmax_rescale'),
               (463,498,'PV_pack_accumulate'),(522,524,'final_normalize'),(526,528,'final_sync')]:
                if lo<=n<=hi:return label
    names=' '.join(x[0] for x in frames)
    for token,label in [('cf_wide_qk','QK'),('cache_qk','QK'),('spm_kernel_1vl','QK'),
      ('cf_vector_pack_A','shared_Q_P_pack'),('spm_pv','PV'),('cache_pv','PV'),
      ('stage_kv','stage_V'),('copy_line','stage_V'),('acc_f16','PV_accumulate'),
      ('scale_f32','output_rescale'),('exp','softmax_exp'),('barrier','synchronization'),
      ('futex','synchronization'),('memset','shared_memory_helpers'),('transpose_K','prepare_K')]:
        if token in names:return label
    return 'other_or_shared'

def instruction(data,va):
    phoff=struct.unpack_from('<Q',data,32)[0];ents,n=struct.unpack_from('<HH',data,54)
    for i in range(n):
        typ,flags,off,vaddr,_,size,_,_=struct.unpack_from('<IIQQQQQQ',data,phoff+i*ents)
        if typ==1 and vaddr<=va< vaddr+size:return data[off+va-vaddr:off+va-vaddr+4]
    raise ValueError(hex(va))

def main():
    out=BASE/'winner_analysis_20260913';out.mkdir(exist_ok=True)
    results={};a.phase=phase
    for name,run in RUNS.items():
        args=json.loads((run/'command.json').read_text());guest=Path(args[args.index('--cmd')+1]);debug=guest.with_name('guest.debug')
        lines=guest.with_name('driver.cpp').read_text().splitlines()
        shift=next(i+1 for i,line in enumerate(lines) if 'auto qk_phase' in line)-385
        for n,text in [(385,'auto qk_phase'),(420,'auto softmax_phase'),(463,'auto pv_phase'),(522,'CF_PHASE(final_normalize)')]:
            assert text in lines[n+shift-1],(n,lines[n+shift-1])
        result=a.analyze(run,debug,shift);g,d=guest.read_bytes(),debug.read_bytes()
        for row in result['pcs']:
            pc=int(row['pc'],16);assert instruction(g,pc)==instruction(d,pc),('sampled PC differs',name,hex(pc))
        result['symbol_gate']='All sampled PC instructions match debug ELF; source phase anchors verified'
        results[name]=result;(out/(name+'.json')).write_text(json.dumps(result,indent=2))
    audit(read(RUNS['cache'],16)[2],read(RUNS['spm'],16)[2],0)
    for name,r in results.items():
        print(name,'ROI',r['traffic']['roi_us'])
        print('phases',[(k,round(v['roi_core_cycle_percent'],3),round(v['attributed_cycles']*.0004/8,3)) for k,v in sorted(r['phases'].items(),key=lambda kv:-kv[1]['attributed_cycles'])])
        print('empty/suspended us',r['empty_rob_cycles']*.0004/8,r['suspended_cycles']*.0004/8)
    deltas=[]
    for k in set(results['cache']['messages'])|set(results['spm']['messages']):
        c=results['cache']['messages'].get(k,{}).get('packets',0);s=results['spm']['messages'].get(k,{}).get('packets',0)
        deltas.append((k,c,s,s-c))
    print('message deltas',sorted(deltas,key=lambda x:-abs(x[3]))[:12])

if __name__=='__main__':main()
