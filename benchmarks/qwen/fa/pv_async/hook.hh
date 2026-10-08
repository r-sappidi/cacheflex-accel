// Diagnostic only. Coherent payloads cross architectural registers, never Ruby
// functional memory. CPU real loads/stores provide normal coherent visibility.
#include "mem/ruby/structures/CacheMemory.hh"
#include "mem/ruby/slicc_interface/AbstractCacheEntry.hh"
#include "sim/sim_object.hh"
#include "arch/arm/regs/vec.hh"
extern "C" {
#include "softfloat/softfloat.h"
}
#include <cstdlib>
namespace gem5 { namespace pseudo_inst {
struct NccBuffer {
 std::array<uint8_t,16384> input{},output{};
 size_t inputSize=0,loaded=0,outputSize=0,sent=0;
 Tick ready=0;bool computing=false;
 Tick workBegin=0,workReady=0;bool workActive=false;
 uint64_t workTicks=0,overlapTicks=0,workRegions=0,readyPolls=0,notReadyPolls=0,vChecks=0;
 uint64_t pv=0,kp=0,pp=0,service=0,inBytes=0,outBytes=0,polls=0;
};
static uint64_t nccProbe(ThreadContext *tc,uint64_t op,uint64_t a,uint64_t b,uint64_t c,uint64_t mode) {
 panic_if(!std::getenv("CACHEFLEX_NCC_PROBE"),"NCC diagnostic disabled");
 static std::array<NccBuffer,8> buffers;
 unsigned id=tc->getCpuPtr()->cpuId();panic_if(id>=8,"NCC CPU bounds");auto& buf=buffers[id];
 if(op==13){panic_if(buf.workActive,"Nested CPU work region");buf.workActive=true;buf.workBegin=curTick();buf.workReady=buf.computing?buf.ready:curTick();return 0;}
 if(op==14){panic_if(!buf.workActive,"Missing CPU work begin");buf.workTicks+=curTick()-buf.workBegin;
  buf.overlapTicks+=std::min(curTick(),buf.workReady)-std::min(buf.workBegin,buf.workReady);++buf.workRegions;buf.workActive=false;return 0;}
 if(op==15){panic_if(buf.computing||buf.sent!=buf.outputSize,"V reused before completion");++buf.vChecks;return 0;}
 if(op==9){
  panic_if(buf.workActive,"Unclosed CPU work region");
  std::cout<<"NCC_OVERLAP core="<<id<<" work_ticks="<<buf.workTicks<<" overlap_ticks="<<buf.overlapTicks<<" regions="<<buf.workRegions<<" ready_polls="<<buf.readyPolls<<" blocked_polls="<<buf.notReadyPolls<<" v_checks="<<buf.vChecks<<std::endl;
  panic_if(buf.computing||buf.sent!=buf.outputSize,"NCC incomplete command at exit");
  std::cout<<"NCC_COUNTS core="<<id<<" pv="<<buf.pv<<" kpack="<<buf.kp<<" ppack="<<buf.pp<<" service_cycles="<<buf.service<<std::endl;
  std::cout<<"NCC_TRANSPORT core="<<id<<" input_bytes="<<buf.inBytes<<" output_bytes="<<buf.outBytes<<" polls="<<buf.polls<<std::endl;return 0;
 }
 if(op==10){
  panic_if(buf.computing||buf.sent!=buf.outputSize,"NCC command overlap");
  panic_if(a!=8192&&a!=16384,"NCC input bounds");buf.inputSize=a;buf.loaded=0;buf.outputSize=buf.sent=0;return 0;
 }
 if(op==11){
  panic_if(buf.computing||a!=buf.loaded||a+1024>buf.inputSize,"NCC input chunk order/bounds");
  for(unsigned i=0;i<4;++i){ArmISA::VecRegContainer reg;tc->getReg(ArmISA::vecRegClass[28+i],&reg);std::memcpy(buf.input.data()+a+256*i,reg.as<uint8_t>(),256);}
  buf.loaded+=1024;buf.inBytes+=1024;return 0;
 }
 if(op==4){++buf.polls;bool ready=curTick()>=buf.ready;if(ready)++buf.readyPolls;else ++buf.notReadyPolls;return ready;}
 if(op==12){
  panic_if(!buf.computing||curTick()<buf.ready||a!=buf.sent||a+1024>buf.outputSize,"NCC output readiness/order/bounds");
  for(unsigned i=0;i<4;++i){ArmISA::VecRegContainer reg;reg.zero();std::memcpy(reg.as<uint8_t>(),buf.output.data()+a+256*i,256);tc->setReg(ArmISA::vecRegClass[28+i],&reg);}
  buf.sent+=1024;buf.outBytes+=1024;if(buf.sent==buf.outputSize)buf.computing=false;return 0;
 }
 panic_if(mode>8||buf.computing||buf.loaded!=buf.inputSize,"NCC invalid mode or incomplete input");
 auto* src=reinterpret_cast<const uint16_t*>(buf.input.data());auto* dst=reinterpret_cast<uint16_t*>(buf.output.data());
 if(op==1 || op==2){
  const size_t rows=op==1?128:16,cols=op==1?64:256;
  panic_if(buf.inputSize!=rows*cols*2,"NCC packing geometry");
  for(size_t r=0;r<rows;++r)for(size_t k=0;k<cols;++k)dst[op==1?k*rows+r:(r/8)*cols*8+k*8+r%8]=src[r*cols+k];
  if(op==1)++buf.kp;else ++buf.pp;buf.outputSize=rows*cols*2;buf.ready=curTick();buf.computing=true;return 0;
 }
 panic_if(op!=3||buf.inputSize!=8192,"NCC unknown operation/PV input");
 std::array<uint16_t,256*64> v{};std::fill(dst,dst+16*64,uint16_t(0));
 std::string name="system.ruby.l1_cntrl"+std::to_string(id)+".cache";
 auto *cache=dynamic_cast<ruby::CacheMemory*>(SimObject::find(name.c_str()));panic_if(!cache,"NCC missing cache");
 for(unsigned k=0;k<256;++k)for(unsigned w=0;w<2;++w){
  Addr addr=0x60000000ULL|(Addr(w)<<16)|(Addr(k)<<6);auto *entry=cache->lookup(addr);
  panic_if(!entry||!entry->isScratchpad(),"NCC requires claimed V");auto data=cache->readSPMData(addr,64);std::memcpy(v.data()+k*64+w*32,data.getData(0,64),64);
 }
 auto savedMode=softfloat_roundingMode;auto savedFlags=softfloat_exceptionFlags;
 softfloat_roundingMode=softfloat_round_near_even;softfloat_exceptionFlags=0;
 for(unsigned r=0;r<16;++r)for(unsigned k=0;k<256;++k){float16_t x{src[(r/8)*256*8+k*8+r%8]};
  for(unsigned d=0;d<64;++d)dst[r*64+d]=f16_mulAdd(x,float16_t{v[k*64+d]},float16_t{dst[r*64+d]}).v;
 }
 softfloat_roundingMode=savedMode;softfloat_exceptionFlags=savedFlags;++buf.pv;
 uint64_t cycles=0;
 if(mode>=5){unsigned lanes=mode==5?64:mode==7?256:128;cycles=262144/lanes+32+8192/32+32768/32+2048/32;if(mode==8)cycles+=43008/16+40;}
 buf.service+=cycles;buf.ready=tc->getCpuPtr()->clockEdge(Cycles(cycles));buf.outputSize=2048;buf.computing=true;return 0;
}
}}
