"""Symbol-verified corrected-driver PC/route corroboration, not causal sums."""
from pathlib import Path
import json,re,subprocess,sys,tempfile
import analyze_attribution as a
def phase(frames,shift=0):
    names=' '.join(f for f,_ in frames)
    for key,label in [('cf_vector_pack_A','vector_Q_P_pack'),('pack_A_fp16','legacy_Q_P_pack'),('cache_qk','QK'),('spm_kernel_1vl','QK'),('cache_pv','PV'),('stage_kv','stage_K'),('acc_f16','PV_accumulate'),('scale_f32','output_rescale'),('exp','softmax_exp'),('barrier','synchronization'),('futex','synchronization')]:
        if key in names:return label
    for _,loc in frames:
        match=re.search(r'correct_driver.cpp:(\d+)',loc)
        if match:
            n=int(match[1])
            for lo,hi,label in [(295,311,'prepare_K'),(333,349,'query_pack_initialize'),(350,384,'stage_K_control'),(385,419,'QK'),(420,462,'softmax_rescale'),(463,498,'PV_pack_accumulate'),(522,524,'final_normalize')]:
                if lo<=n<=hi:return label
    return 'unresolved_or_shared'
def analyze(run,elf):
    args=json.loads((run/'command.json').read_text());guest=Path(args[args.index('--cmd')+1])
    with tempfile.TemporaryDirectory(prefix='cf-numeric-text-') as d:
        for i,p in enumerate([guest,elf]):subprocess.run(['aarch64-linux-gnu-objcopy','--dump-section',f'.text={d}/{i}.text',str(p),f'{d}/{i}.elf'],check=True)
        assert Path(d,'0.text').read_bytes()==Path(d,'1.text').read_bytes()
    a.phase=phase;result=a.analyze(run,elf)
    (run/'correct_attribution.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
    run=Path(sys.argv[1]);args=json.loads((run/'command.json').read_text());elf=Path(args[args.index('--cmd')+1]).with_name('guest.debug')
    r=analyze(run,elf)
    print(json.dumps({k:dict(core_us_div8=v['attributed_cycles']*.0004/8,retired=v.get('pcRetiredMicroops',0)) for k,v in r['phases'].items()},indent=2))
