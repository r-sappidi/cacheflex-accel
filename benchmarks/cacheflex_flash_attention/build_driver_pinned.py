"""Build scalar per-core pinned-K variants with aligned native GEMM loops.

An identical one-time branch skips alignment padding; loop bodies are unchanged.
No source in the micro repo or active driver sweep is edited.
"""
from pathlib import Path
import json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve();v=sys.argv[2];g=int(sys.argv[3]);mask=int(sys.argv[4]);out.mkdir()
def align(s):
    token='"1:\\n"'
    assert token in s
    return s.replace(token,'"b 1f\\n.p2align 6\\n1:\\n"')
micro=ROOT.parent/'cacheflex_micro/experiments/end2end/src'
subprocess.run(['python3',str(HERE/'make_search_kernels.py'),str(out/'kernels.hpp'),'1','0','0','0'],check=True)
(out/'kernels.hpp').write_text(align((out/'kernels.hpp').read_text()))
flags=f'-std=c++17 -O3 -g -march=armv8.2-a+sve+fp16 -mbranch-protection=none -DGEM5 -DGEM5_SE -DVL_16 -DCACHEFLEX_FLASH_VL16=1 -DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_SKIP_UNUSED_TIMERS=1 -I"{HERE}" -I"$M5_INCLUDE" -I"{ROOT.parent}/cacheflex_micro/kernels/llama_bench_spm" -I"{ROOT.parent}/cacheflex_micro/kernels/attention_kernels_spm"'
if '--rich' in sys.argv:flags+=' -DCACHEFLEX_FLASH_RICH_INPUT=1'
if '--correct' in sys.argv:flags+=f' -include "{HERE}/correct_fa_conversions.hpp"'
if v=='original':
    # Local copies retain original driver, arithmetic and helpers.
    (out/'pv_kernel_8x64_vl16.hpp').write_text(align((micro/'pv_kernel_8x64_vl16.hpp').read_text()))
    source=align((micro/'cache_flash_attn.cpp').read_text())
    (out/'cache_flash_attn.cpp').write_text(source)
    wrapper=(HERE/'cache_multicore.cpp').read_text().replace('"../../../cacheflex_micro/experiments/end2end/src/cache_flash_attn.cpp"','"cache_flash_attn.cpp"')
    (out/'driver.cpp').write_text(wrapper)
    cmd=f'"$CROSS_CXX" {flags} -I"{micro}" -static -pthread "{out}/driver.cpp" "$M5OP_OBJ" -lm -o "{out}/guest.debug"'
else:
    assert v=='spm' and mask & 64
    source=(HERE/'spm_multicore.cpp').read_text()
    source=source.replace('"../../../cacheflex_micro/',f'"{ROOT.parent}/cacheflex_micro/')
    anchor='    for(size_t group_begin=0;group_begin<n_ib;group_begin+=ib_group){'
    assert source.count(anchor)==1
    stage='''    // Pin every K panel once per core using existing scalar SPMCP.
    // This consumes (ceil(T/128)*d*4*64) bytes, within four eligible ways.
    if (((size_t)T+Bc-1)/Bc*kbank_stride>1024) std::abort();
    CF_PHASE(stage_K_initial);
    for(int pj=0;pj<T;pj+=Bc) {
      int pbc=std::min(Bc,T-pj);size_t pbb=((size_t)pbc+127)/128;
      stage_kv(s.KT.data(),s.V.data(),T,d,pj,pbc,pbb,false,
               ((size_t)pj/Bc)*kbank_stride,false);
    }
'''
    source=source.replace(anchor,stage+anchor)
    source=source.replace('if(CACHEFLEX_FLASH_K_OVERLAP && !CACHEFLEX_FLASH_MULTICAST_PIN_ALL) {','if(false) {')
    source=source.replace('size_t kset_base=CACHEFLEX_FLASH_MULTICAST_PIN_ALL?','size_t kset_base=true?')
    source=source.replace('if(CACHEFLEX_FLASH_MULTICAST_PIN_ALL) {\n        // All panels were installed before the compute loop.','if(true) {\n        // All panels were installed before the compute loop.')
    if '--correct' in sys.argv:
        assert source.count('return 0;\n}')==1
        source=source.replace('return 0;\n}','return cf_audit_output(s.O.data(),s.V.data(),(size_t)8*T*d,d);\n}')
    (out/'pinned_driver.cpp').write_text(source)
    flags+=f' -DCACHEFLEX_FLASH_SEARCH=1 -DCACHEFLEX_FLASH_MATCHED=1 -DCACHEFLEX_FLASH_BOUNDARIES=1 -DCACHEFLEX_FLASH_STAGE_K={int(v=="spm")} -DCACHEFLEX_FLASH_STAGE_V=0 -DCACHEFLEX_FLASH_K_OVERLAP=1 -DCACHEFLEX_FLASH_IB_GROUP={g} -DCACHEFLEX_FLASH_DRIVER_ABLATION={mask} -DCACHEFLEX_FLASH_SEARCH_KERNELS=\\\"{out}/kernels.hpp\\\"'
    cmd=f'"$CROSS_CXX" {flags} -c "{HERE}/cache_qk_wrapper.cpp" -o "{out}/qk.o"\n"$CROSS_CXX" {flags} -S "{out}/pinned_driver.cpp" -o "{out}/guest.s"\npython3 "$SPM_COMPILER" "{out}/guest.s" "{out}/guest.enc.s"\n"$CROSS_CXX" -static -pthread "{out}/guest.enc.s" "{out}/qk.o" "$M5OP_OBJ" -lm -o "{out}/guest.debug"'
script=f'set -e\nsource "{ROOT}/setup_spm_env.sh" >/dev/null\n'+cmd
(out/'build_command.sh').write_text(script+'\n')
subprocess.run(['bash','-c',script],check=True)
subprocess.run(['aarch64-linux-gnu-objcopy','--strip-debug',str(out/'guest.debug'),str(out/'guest')],check=True)
(out/'experiment.json').write_text(json.dumps(dict(variant=v,group=g,mask=mask,alignment=64),indent=2))
