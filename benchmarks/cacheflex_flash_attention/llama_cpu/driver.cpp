// Public ggml CPU API harness. No replacement of upstream attention scheduling.
#include <ggml.h>
#include <ggml-cpu.h>
#include <gem5/m5ops.h>
#include <arm_sve.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

static uint16_t pattern(size_t i,uint64_t salt) {
    uint64_t x=i+salt*0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;i=(x^(x>>31))>>29;
    const uint16_t b[]={0x2c00,0xac00,0x3000,0xb000,0x2800,0xa800,0x3400,0xb400};
    return b[(i*5+salt*3)&7];
}
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    const int t=atoi(argv[1]),threads=atoi(argv[2]);
    if(t<1||threads<1||threads>8||svcntb()!=256)return 2;
    setvbuf(stdout,nullptr,_IONBF,0);
    ggml_cpu_init();
    ggml_init_params ip={size_t(t)*64*16+16*1024*1024,nullptr,false};
    ggml_context* ctx=ggml_init(ip);if(!ctx)return 3;
    auto* q=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,64,t,1,1);
    auto* k=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,64,t,1,1);
    auto* v=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,64,t,1,1);
    auto* out=ggml_flash_attn_ext(ctx,q,k,v,nullptr,.125f,0,0);
    auto* graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
    std::vector<__fp16> q16(size_t(t)*64),out16(size_t(t)*64);
    std::memset(q->data,0,ggml_nbytes(q));std::memset(out->data,0,ggml_nbytes(out));
    for(size_t i=0;i<q16.size();++i) {
        uint16_t qb=pattern(i,1);std::memcpy(&q16[i],&qb,2);
        static_cast<uint16_t*>(k->data)[i]=pattern(i,2);
        static_cast<uint16_t*>(v->data)[i]=pattern(i,3);
    }
    auto tp=ggml_threadpool_params_default(threads);
    auto* pool=ggml_threadpool_new(&tp);if(!pool)return 4;
    // Complete worker startup outside ROI, as in the custom pthread harness.
    // A disjoint add graph touches no attention inputs or attention kernel.
    auto* boot=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,8192);
    std::memset(boot->data,0,ggml_nbytes(boot));
    auto* boot_out=ggml_add(ctx,boot,boot);auto* boot_graph=ggml_new_graph(ctx);
    ggml_build_forward_expand(boot_graph,boot_out);
    auto boot_plan=ggml_graph_plan(boot_graph,threads,pool);
    std::vector<uint8_t> boot_scratch(boot_plan.work_size);boot_plan.work_data=boot_scratch.data();
    if(ggml_graph_compute(boot_graph,&boot_plan)!=GGML_STATUS_SUCCESS)return 8;
    std::printf("LLAMA_WORKERS_READY no_attention_warmup=1\n");
    auto plan=ggml_graph_plan(graph,threads,pool);
    std::vector<uint8_t> scratch(plan.work_size);plan.work_data=scratch.data();
    std::printf("LLAMA_CONFIG T=%d threads=%d vl_bits=%lu q_api=fp32 kv=fp16 accumulation=fp32 output_api=fp32 poll=%u work_bytes=%zu\n",
        t,threads,svcntb()*8,tp.poll,plan.work_size);
    std::printf("LLAMA_ROI_BEGIN\n");m5_reset_stats(0,0);m5_work_begin(0,0);
    // Match external FP16 operand/output contract; charge all API conversions.
    for(size_t i=0;i<q16.size();++i)static_cast<float*>(q->data)[i]=float(q16[i]);
    const auto status=ggml_graph_compute(graph,&plan);
    for(size_t i=0;i<out16.size();++i)out16[i]=static_cast<float*>(out->data)[i];
    m5_work_end(0,0);m5_dump_stats(0,0);std::printf("LLAMA_ROI_END status=%d\n",int(status));
    size_t bad=0;uint64_t hash=1469598103934665603ULL;
    for(auto x:out16){uint16_t b;std::memcpy(&b,&x,2);hash=(hash^b)*1099511628211ULL;bad+=!std::isfinite(float(x));}
    std::printf("LLAMA_RESULT hash=%016llx nonfinite=%zu elements=%zu\n",(unsigned long long)hash,bad,out16.size());
    if(m5_write_file(out16.data(),out16.size()*2,0,"output.fp16")!=out16.size()*2)return 5;
    if(m5_write_file(out->data,ggml_nbytes(out),0,"output.fp32")!=ggml_nbytes(out))return 6;
    ggml_threadpool_free(pool);ggml_free(ctx);
    return status==GGML_STATUS_SUCCESS&&!bad?0:7;
}
