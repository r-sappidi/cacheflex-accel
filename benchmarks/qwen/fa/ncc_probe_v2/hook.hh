// Functional arithmetic proxy. No claim of timed engine/coherence implementation.
#include "mem/ruby/structures/CacheMemory.hh"
#include "mem/ruby/slicc_interface/AbstractCacheEntry.hh"
#include "sim/sim_object.hh"
extern "C" {
#include "softfloat/softfloat.h"
}
#include <cstdlib>
namespace gem5 { namespace pseudo_inst {
static uint64_t nccProbe(ThreadContext *tc,uint64_t op,uint64_t a,uint64_t b,uint64_t c,uint64_t mode) {
 panic_if(!std::getenv("CACHEFLEX_NCC_PROBE"),"NCC diagnostic disabled");
 SETranslatingPortProxy proxy(tc);
 static std::array<uint64_t,8> pv{},kp{},pp{},service{};
 unsigned id=tc->getCpuPtr()->cpuId();panic_if(id>=8,"NCC CPU bounds");
 if(op==9){std::cout<<"NCC_COUNTS core="<<id<<" pv="<<pv[id]<<" kpack="<<kp[id]<<" ppack="<<pp[id]<<" service_cycles="<<service[id]<<std::endl;return 0;}
 panic_if(mode>8,"NCC invalid mode");
 if(op==1 || op==2){
  const size_t rows=op==1?128:16,cols=op==1?64:256;
  std::vector<uint16_t> src(rows*cols),dst(rows*cols);
  proxy.readBlob(a,src.data(),src.size()*2);
  for(size_t r=0;r<rows;++r)for(size_t k=0;k<cols;++k)
   dst[op==1?k*rows+r:(r/8)*cols*8+k*8+r%8]=src[r*cols+k];
  proxy.writeBlob(b,dst.data(),dst.size()*2);
  if(op==1)++kp[id];else ++pp[id];return 0;
 }
 panic_if(op!=3,"NCC unknown operation");
 // Packed P has two eight-row blocks; V is the already-claimed private panel.
 std::vector<uint16_t> p(16*256),v(256*64),out(16*64,0);
 proxy.readBlob(a,p.data(),p.size()*2);
 std::string name="system.ruby.l1_cntrl"+std::to_string(id)+".cache";
 auto *cache=dynamic_cast<ruby::CacheMemory*>(SimObject::find(name.c_str()));panic_if(!cache,"NCC missing cache");
 for(unsigned k=0;k<256;++k)for(unsigned w=0;w<2;++w){
  Addr addr=0x60000000ULL|(Addr(w)<<16)|(Addr(k)<<6);
  auto *entry=cache->lookup(addr);panic_if(!entry||!entry->isScratchpad(),"NCC requires claimed V");
  auto data=cache->readSPMData(addr,64);std::memcpy(v.data()+k*64+w*32,data.getData(0,64),64);
 }
 auto savedMode=softfloat_roundingMode;auto savedFlags=softfloat_exceptionFlags;
 softfloat_roundingMode=softfloat_round_near_even;softfloat_exceptionFlags=0;
 for(unsigned r=0;r<16;++r)for(unsigned k=0;k<256;++k){
  float16_t x{p[(r/8)*256*8+k*8+r%8]};
  for(unsigned d=0;d<64;++d)out[r*64+d]=f16_mulAdd(x,float16_t{v[k*64+d]},float16_t{out[r*64+d]}).v;
 }
 softfloat_roundingMode=savedMode;softfloat_exceptionFlags=savedFlags;
 proxy.writeBlob(b,out.data(),out.size()*2);++pv[id];
 // Blocking service sensitivity, not a bank/network simulation. Transfer costs
 // are explicit additions after CPU-side memory probes, conservatively serialized.
 uint64_t cycles=0;
 if(mode>=5){
  unsigned lanes=mode==5?64:mode==7?256:128;
  cycles=(16*64*256+lanes-1)/lanes + 32 + 8192/32 + 32768/32 + 2048/32;
  if(mode==8)cycles+=(8192+32768+2048)/16+40; // pessimistic L3 delivery per command
 }
 service[id]+=cycles;
 if(cycles)quiesceCycles(tc,cycles);
 return 0;
}
}}
