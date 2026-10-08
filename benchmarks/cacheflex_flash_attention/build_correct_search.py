"""Isolated corrected FA search builder. No shared guest/hardware edits."""
from pathlib import Path
import json, subprocess, sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def build(out, c):
    out.mkdir()
    placement=c['placement']; k=placement in ('k','kv'); v=placement in ('v','kv')
    subprocess.run(['python3',str(HERE/'make_search_kernels.py'),str(out/'kernels.hpp'),
                    str(c['unroll']),str(c['order']),str(c['rows']),str(c['pipeline'])],check=True)
    kernels=(out/'kernels.hpp').read_text()
    assert '"1:\\n"' in kernels
    kernels=kernels.replace('"1:\\n"','"b 1f\\n.p2align 6\\n1:\\n"')
    # A constant set offset is applied to both scalar copies and SPM load
    # bases, never to coherent addresses. No wraparound is allowed by runner.
    offset=c.get('offset',0)
    if offset:
        import re
        kernels,n=re.subn(r'(size_t base_set = 0\) \{)',r'\1\n base_set += '+str(offset)+';',kernels)
        kernels,n2=re.subn(r'(size_t\s+spm_base_v\)\s*\{)',r'\1\n spm_base_v += '+str(offset)+';',kernels)
        kernels=re.sub(r'(size_t base\) \{)',r'\1\n base += '+str(offset)+';',kernels)
        # Inspect exact native argument spellings instead of silently missing a load.
        if n != 1 or n2 != 1:
            raise RuntimeError('SPM offset transform did not cover QK and PV')
    (out/'kernels.hpp').write_text(kernels)
    source=(HERE/'spm_multicore.cpp').read_text().replace('"../../../cacheflex_micro/',f'"{ROOT.parent}/cacheflex_micro/')
    assert source.count('return 0;\n}')==1
    source=source.replace('return 0;\n}','return cf_audit_output(s.O.data(),s.V.data(),(size_t)8*T*d,d);\n}')
    if offset:
        anchor='static inline void copy_line(size_t slot,const void *src,uint64_t mask=0)\n{'
        assert source.count(anchor)==1
        source=source.replace(anchor,anchor+f'\n    slot += {offset}u << 6;')
    if c.get('pinned',0):
        assert placement=='k'
        anchor='    for(size_t group_begin=0;group_begin<n_ib;group_begin+=ib_group){'
        assert source.count(anchor)==1
        source=source.replace(anchor,'''    if (((size_t)T+Bc-1)/Bc*kbank_stride>1024) std::abort();
    CF_PHASE(stage_K_initial);
    for(int pj=0;pj<T;pj+=Bc) {
      int pbc=std::min(Bc,T-pj);size_t pbb=((size_t)pbc+127)/128;
      stage_kv(s.KT.data(),s.V.data(),T,d,pj,pbc,pbb,false,
               ((size_t)pj/Bc)*kbank_stride,false);
    }
'''+anchor)
        source=source.replace('if(CACHEFLEX_FLASH_K_OVERLAP && !CACHEFLEX_FLASH_MULTICAST_PIN_ALL) {','if(false) {')
        source=source.replace('size_t kset_base=CACHEFLEX_FLASH_MULTICAST_PIN_ALL?','size_t kset_base=true?')
        source=source.replace('if(CACHEFLEX_FLASH_MULTICAST_PIN_ALL) {\n        // All panels were installed before the compute loop.','if(true) {\n        // All panels were installed before the compute loop.')
    (out/'driver.cpp').write_text(source)
    defs=dict(GEM5=1,GEM5_SE=1,VL_16=1,CACHEFLEX_FLASH_VL16=1,
        CACHEFLEX_FLASH_SEARCH=1,CACHEFLEX_FLASH_MATCHED=1,CACHEFLEX_FLASH_BOUNDARIES=1,
        CACHEFLEX_FLASH_PRETOUCH=1,CACHEFLEX_FLASH_SKIP_UNUSED_TIMERS=1,
        CACHEFLEX_FLASH_STAGE_K=int(k),CACHEFLEX_FLASH_STAGE_V=int(v),
        CACHEFLEX_FLASH_IB_GROUP=c['group'],CACHEFLEX_FLASH_K_OVERLAP=c['overlap'],
        CACHEFLEX_FLASH_DRIVER_ABLATION=3,CACHEFLEX_FLASH_WIDE_ROWS=c['rows'],
        CACHEFLEX_FLASH_FUSE_RESCALE=c['fusion'],CACHEFLEX_FLASH_RICH_INPUT=c.get('rich',0),
        CACHEFLEX_FLASH_INTERLEAVE_IB=c.get('interleave',0))
    flags=' '.join(f'-D{a}={b}' for a,b in defs.items())
    flags+=f' -std=c++17 -O3 -g -march=armv8.2-a+sve+fp16 -mbranch-protection=none -include "{HERE}/correct_fa_conversions.hpp" -I"{HERE}" -I"$M5_INCLUDE" -I"{ROOT.parent}/cacheflex_micro/kernels/llama_bench_spm" -I"{ROOT.parent}/cacheflex_micro/kernels/attention_kernels_spm"'
    flags+=f' -DCACHEFLEX_FLASH_SEARCH_KERNELS=\\\"{out}/kernels.hpp\\\"'
    script=f'''set -e
source "{ROOT}/setup_spm_env.sh" >/dev/null
"$CROSS_CXX" {flags} -c "{HERE}/cache_qk_wrapper.cpp" -o "{out}/qk.o"
"$CROSS_CXX" {flags} -S "{out}/driver.cpp" -o "{out}/guest.s"
python3 "$SPM_COMPILER" "{out}/guest.s" "{out}/guest.enc.s"
"$CROSS_CXX" -static -pthread "{out}/guest.enc.s" "{out}/qk.o" "$M5OP_OBJ" -lm -o "{out}/guest.debug"
aarch64-linux-gnu-objcopy --strip-debug "{out}/guest.debug" "{out}/guest"
'''
    (out/'build_command.sh').write_text(script)
    subprocess.run(['bash','-c',script],check=True)
    (out/'experiment.json').write_text(json.dumps(c,indent=2))

if __name__=='__main__':
    build(Path(sys.argv[1]).resolve(),json.loads(Path(sys.argv[2]).read_text()))
