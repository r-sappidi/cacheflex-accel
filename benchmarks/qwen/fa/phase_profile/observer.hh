// Passive observers only: no modeled events, register/cache/queue modifications.
#include "cpu/pc_event.hh"
#include "sim/system.hh"
#include "phase_api.hh"
#include <fstream>
#include <set>
#include <unordered_map>
namespace gem5 {
namespace {
constexpr unsigned FaPhases=26;
struct FaRequest {Tick start;unsigned phase,bytes;bool spm;};
struct FaPhase {
 Tick ticks=0,copyBusy=0,copyArea=0,rawBusy=0,rawArea=0,cacheBusy=0,cacheArea=0,spmBusy=0,spmArea=0;
 uint64_t entries=0,copyIssue=0,copyDone=0,copyLatency=0,rawStart=0,rawDone=0,rawSquash=0,rawLatency=0;
 uint64_t cacheIssue=0,cacheDone=0,cacheBytes=0,cacheLatency=0,spmIssue=0,spmDone=0,spmBytes=0,spmLatency=0;
 uint64_t cachePortBlock=0,spmPortBlock=0;
};
struct FaCore {
 unsigned phase=0;Tick last=0,arrival=0;
 std::array<FaPhase,FaPhases> phases{};
 std::unordered_map<uint64_t,FaRequest> copies,raw,memory;
 uint64_t cachePending=0,spmPending=0,copyMax=0,rawMax=0,cacheMax=0,spmMax=0;
};
std::array<FaCore,8> faCores;
bool faActive=false;Tick faBegin=0;
FaCore& faFlush(unsigned id) {
 panic_if(id>=8,"FA phase core bounds");auto& c=faCores[id];Tick dt=curTick()-c.last;auto& p=c.phases[c.phase];
 p.ticks+=dt;p.copyBusy+=c.copies.empty()?0:dt;p.copyArea+=dt*c.copies.size();
 p.rawBusy+=c.raw.empty()?0:dt;p.rawArea+=dt*c.raw.size();
 p.cacheBusy+=c.cachePending?dt:0;p.cacheArea+=dt*c.cachePending;
 p.spmBusy+=c.spmPending?dt:0;p.spmArea+=dt*c.spmPending;c.last=curTick();return c;
}
}
void faCopyBegin(unsigned id,uint64_t key) {
 if(!faActive)return;auto& c=faFlush(id);auto& p=c.phases[c.phase];
 panic_if(!c.copies.emplace(key,FaRequest{curTick(),c.phase,64,false}).second,"FA duplicate copy");
 ++p.copyIssue;c.copyMax=std::max(c.copyMax,uint64_t(c.copies.size()));
}
void faCopyEnd(unsigned id,uint64_t key) {
 if(!faActive)return;auto& c=faFlush(id);auto it=c.copies.find(key);panic_if(it==c.copies.end(),"FA unknown copy completion");
 ++c.phases[c.phase].copyDone;c.phases[it->second.phase].copyLatency+=curTick()-it->second.start;c.copies.erase(it);
}
void faRawBegin(unsigned id,uint64_t key) {
 if(!faActive)return;auto& c=faFlush(id);panic_if(!c.raw.emplace(key,FaRequest{curTick(),c.phase,0,true}).second,"FA duplicate RAW wait");
 ++c.phases[c.phase].rawStart;c.rawMax=std::max(c.rawMax,uint64_t(c.raw.size()));
}
void faRawEnd(unsigned id,uint64_t key,bool squashed) {
 if(!faActive)return;auto& c=faFlush(id);auto it=c.raw.find(key);panic_if(it==c.raw.end(),"FA unknown RAW completion");
 auto& p=c.phases[c.phase];if(squashed)++p.rawSquash;else {++p.rawDone;p.rawLatency+=curTick()-it->second.start;}c.raw.erase(it);
}
void faMemBegin(unsigned id,uint64_t key,unsigned bytes,bool spm) {
 if(!faActive)return;auto& c=faFlush(id);panic_if(!c.memory.emplace(key,FaRequest{curTick(),c.phase,bytes,spm}).second,"FA duplicate data packet");
 auto& p=c.phases[c.phase];if(spm){++p.spmIssue;p.spmBytes+=bytes;++c.spmPending;c.spmMax=std::max(c.spmMax,c.spmPending);}
 else{++p.cacheIssue;p.cacheBytes+=bytes;++c.cachePending;c.cacheMax=std::max(c.cacheMax,c.cachePending);}
}
void faMemEnd(unsigned id,uint64_t key) {
 if(!faActive)return;auto& c=faFlush(id);auto it=c.memory.find(key);
 // Untracked responses include stores and requests admitted before the ROI.
 if(it==c.memory.end())return;auto& p=c.phases[c.phase];auto& origin=c.phases[it->second.phase];
 if(it->second.spm){++p.spmDone;origin.spmLatency+=curTick()-it->second.start;--c.spmPending;}
 else{++p.cacheDone;origin.cacheLatency+=curTick()-it->second.start;--c.cachePending;}c.memory.erase(it);
}
void faPortBlock(unsigned id,bool spm) {
 if(!faActive)return;auto& c=faFlush(id);auto& p=c.phases[c.phase];if(spm)++p.spmPortBlock;else++p.cachePortBlock;
}
namespace pseudo_inst {
class FaPhaseEvent:public PCEvent {
 unsigned phase;
 public:
 FaPhaseEvent(PCEventScope* scope,Addr pc,unsigned p):PCEvent(scope,"FA phase",pc),phase(p){}
 void process(ThreadContext* tc) override {
  if(!faActive)return;auto& c=faFlush(tc->getCpuPtr()->cpuId());c.phase=phase;++c.phases[phase].entries;
  if(phase==24&&!c.arrival)c.arrival=curTick();
 }
};
static void pvDiagStart(ThreadContext* tc) {
 const char* path=std::getenv("CACHEFLEX_PV_PHASE_MAP");if(!path||!*path)return;
 panic_if(faActive,"Repeated FA ROI");std::ifstream f(path);panic_if(!f,"FA phase map unavailable");
 std::string address;unsigned phase,count=0;std::set<Addr> seen;
 while(f>>address>>phase){Addr pc=std::stoull(address,nullptr,16);panic_if(phase>=FaPhases||!seen.insert(pc).second,"Invalid FA phase map");new FaPhaseEvent(tc->getSystemPtr(),pc,phase);++count;}
 panic_if(!count,"Empty FA phase map");faBegin=curTick();for(auto& c:faCores){c=FaCore{};c.last=curTick();}faActive=true;
}
static void pvDiagEnd(ThreadContext*) {
 if(!faActive)return;
 for(unsigned id=0;id<8;++id){auto& c=faFlush(id);
  std::cout<<"PV_TIMELINE core="<<id<<" start="<<faBegin<<" end="<<curTick()<<" final_arrival="<<c.arrival<<std::endl;
  std::cout<<"FA_PENDING core="<<id<<" copies="<<c.copies.size()<<" raw="<<c.raw.size()<<" cache="<<c.cachePending<<" spm="<<c.spmPending<<" copy_max="<<c.copyMax<<" raw_max="<<c.rawMax<<" cache_max="<<c.cacheMax<<" spm_max="<<c.spmMax<<std::endl;
  for(unsigned i=0;i<FaPhases;++i){auto& p=c.phases[i];
   std::cout<<"PV_PHASE core="<<id<<" phase="<<i<<" ticks="<<p.ticks<<" entries="<<p.entries<<std::endl;
   std::cout<<"FA_DETAIL core="<<id<<" phase="<<i
    <<" copy_busy="<<p.copyBusy<<" copy_area="<<p.copyArea<<" copy_issue="<<p.copyIssue<<" copy_done="<<p.copyDone<<" copy_latency="<<p.copyLatency
    <<" raw_busy="<<p.rawBusy<<" raw_area="<<p.rawArea<<" raw_start="<<p.rawStart<<" raw_done="<<p.rawDone<<" raw_squash="<<p.rawSquash<<" raw_latency="<<p.rawLatency
    <<" cache_busy="<<p.cacheBusy<<" cache_area="<<p.cacheArea<<" cache_issue="<<p.cacheIssue<<" cache_done="<<p.cacheDone<<" cache_bytes="<<p.cacheBytes<<" cache_latency="<<p.cacheLatency
    <<" spm_busy="<<p.spmBusy<<" spm_area="<<p.spmArea<<" spm_issue="<<p.spmIssue<<" spm_done="<<p.spmDone<<" spm_bytes="<<p.spmBytes<<" spm_latency="<<p.spmLatency
    <<" cache_port_block="<<p.cachePortBlock<<" spm_port_block="<<p.spmPortBlock<<std::endl;
  }
 }faActive=false;
}
}}
