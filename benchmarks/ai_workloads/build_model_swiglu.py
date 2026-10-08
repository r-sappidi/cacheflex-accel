"""Isolated FP16-storage/FP32-accumulation custom and pristine ACL guests."""
import os
from pathlib import Path
import subprocess
import sys
from repair_screen_export import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def build(out):
    out.mkdir()
    for combined in (0,1):
        dest=out/f'custom_g{combined}'
        env=dict(os.environ,MLP_ROWS='8',MLP_ACC32='1',MLP_COMBINED=str(combined))
        subprocess.run([sys.executable,str(HERE/'build_screen_mlp.py'),str(dest)],env=env,check=True)
        for arch in ('cache','spm'):patch(dest/arch)
    # Keep ACL's operator implementation, packing, scheduler and library pristine.
    # Extend only the benchmark harness: precision request, repeated ROIs/export.
    source=(HERE/'mlp_acl.cpp').read_text()
    source=source.replace('argc!=5','argc!=6')
    source=source.replace('const bool warm=', 'const int repeats=atoi(argv[5]); if(repeats<1)return 2;\n    const bool warm=')
    old='GEMMInfo info(false,false,true,0,false,false,GEMMLowpOutputStageInfo(),false,true);'
    assert source.count(old)==1
    source=source.replace(old,'GEMMInfo info(false,false,true,0,false,false,GEMMLowpOutputStageInfo(),false,false);\n    info.set_use_fp32_acc(true);')
    # This pinned NEGEMM does not forward use_fp32_acc; use the public low-level
    # wrapper that does, with ACL's ordinary memory helpers and scheduler.
    source=source.replace('using namespace arm_compute;', '''
#include "arm_compute/runtime/experimental/low_level/CpuGemmAssemblyDispatch.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "src/core/helpers/MemoryHelpers.h"
using namespace arm_compute;
struct ModelACLGemm {
 experimental::op::ll::CpuGemmAssemblyDispatch op;
 MemoryGroup mg;
 ITensorPack run_pack,prep_pack;
 WorkspaceData<Tensor> ws;
 experimental::MemoryRequirements req;
 bool prepared=false;
 void configure(const ITensor* a,const ITensor* b,const ITensor* c,ITensor* d,
                float alpha,float beta,const GEMMInfo& info) {
  if(alpha!=1.f||beta!=0.f||c)std::abort();
  auto valid=decltype(op)::validate(a->info(),b->info(),nullptr,d->info(),info);
  if(!valid)std::abort();
  op.configure(a->info(),b->info(),nullptr,d->info(),info);
  run_pack={{ACL_SRC_0,a},{ACL_SRC_1,b},{ACL_SRC_2,c},{ACL_DST,d}};
  prep_pack={{ACL_SRC_1,b},{ACL_SRC_2,c}};
  req=op.workspace();
  ws=manage_workspace<Tensor>(req,mg,run_pack,prep_pack,false);
 }
 void run() {
  if(!prepared) {
   allocate_tensors(req,ws);op.prepare(prep_pack);
   release_temporaries<Tensor>(req,ws);prepared=true;
  }
  MemoryGroupResourceScope scope(mg);op.run(run_pack);
 }
};
''').replace('NEGEMM gate_gemm','ModelACLGemm gate_gemm')
    # Diagnostic output hashing/check reads between calls is matched to custom.
    check_start=source.index('    size_t errors=0,nonfinite=0;')
    check_end=source.index('    return nonfinite?1:0;',check_start)
    check=source[check_start:check_end]
    check=check.replace('backend=ACL mode=%s','backend=ACL iteration=%d mode=%s').replace('argv[4],t,d,f,','iteration,argv[4],t,d,f,')
    start=source.index('    // A full unmeasured invocation')
    source=source[:start]+'''    auto check=[&](int iteration) {
'''+check+'''        return nonfinite;
    };
    if(warm) run();
    size_t nonfinite=0;
    for(int iteration=0;iteration<repeats;++iteration) {
        printf("MLP_ROI begin backend=ACL epoch=%d T=%d D=%d F=%d acc32=1\\n",iteration,t,d,f);
        m5_reset_stats(0,0);m5_work_begin(0,0);
        run();
        m5_work_end(0,0);m5_dump_stats(0,0);puts("MLP_ROI end");
        nonfinite+=check(iteration);
    }
    dump_tensor(gate,f,t,"gate.bin");dump_tensor(up,f,t,"up.bin");
    dump_tensor(act,f,t,"silu.bin");dump_tensor(hidden,f,t,"hidden.bin");
    dump_tensor(y,d,t,"output.bin");
    return nonfinite?1:0;
}
'''
    helper='''
static uint64_t model_export(const void* data,uint64_t bytes,uint64_t offset,const char* name) {
 alignas(64) uint16_t scratch[32];
 const auto* src=reinterpret_cast<const volatile uint16_t*>(data);
 for(uint64_t i=0;i<bytes;i+=64) {
  uint64_t n=std::min<uint64_t>(64,bytes-i);
  for(uint64_t j=0;j<n/2;++j)scratch[j]=src[i/2+j];
  asm volatile("dsb sy":::"memory");
  if(m5_write_file(scratch,n,offset+i,name)!=n)return i;
 }
 return bytes;
}
'''
    source=source.replace('static void init(',helper+'\nstatic void init(')
    source=source.replace('m5_write_file(a.ptr_to_element','model_export(a.ptr_to_element')
    src=out/'mlp_acl_model.cpp';src.write_text(source)
    aclroot=ROOT/'results/ai_workloads/mlp_acl_20260909/upstream/source'
    env=dict(os.environ,MLP_ACL_SOURCE=str(src),ACL_ROOT=str(aclroot),ACL_LIB=str(aclroot/'build/libarm_compute-static.a'))
    subprocess.run(['bash',str(HERE/'build_mlp_acl.sh'),str(out/'acl')],env=env,check=True)
    patch(out/'acl/acl')

if __name__=='__main__':build(Path(sys.argv[1]).resolve())
