// Passive commit-PC observer. Never modifies guest state, queues or timing.
#include "cpu/pc_event.hh"
#include "sim/system.hh"
#include <fstream>
#include <set>
namespace gem5 { namespace pseudo_inst {
struct PvPhaseState {
 unsigned phase=0;Tick last=0,arrival=0;
 std::array<Tick,26> ticks{};
 std::array<uint64_t,26> entries{};
};
static std::array<PvPhaseState,8> pvPhases;
static bool pvObserve=false;static Tick pvBegin=0;
class PvPhaseEvent:public PCEvent {
 unsigned phase;
 public:
 PvPhaseEvent(PCEventScope* scope,Addr pc,unsigned p):PCEvent(scope,"PV phase",pc),phase(p){}
 void process(ThreadContext* tc) override {
  if(!pvObserve)return;unsigned id=tc->getCpuPtr()->cpuId();panic_if(id>=8,"PV phase core bounds");auto& s=pvPhases[id];
  s.ticks[s.phase]+=curTick()-s.last;s.last=curTick();s.phase=phase;++s.entries[phase];
  if(phase==24 && !s.arrival)s.arrival=curTick();
 }
};
static void pvDiagStart(ThreadContext* tc) {
 const char* path=std::getenv("CACHEFLEX_PV_PHASE_MAP");if(!path || !*path)return;
 panic_if(pvObserve,"PV observer ROI repeated");std::ifstream f(path);panic_if(!f,"PV phase map missing");
 std::string address;unsigned phase,count=0;std::set<Addr> seen;
 while(f>>address>>phase){Addr pc=std::stoull(address,nullptr,16);panic_if(phase>=26||!seen.insert(pc).second,"PV phase map invalid");new PvPhaseEvent(tc->getSystemPtr(),pc,phase);++count;}
 panic_if(!count,"Empty PV phase map");pvBegin=curTick();for(auto& s:pvPhases){s=PvPhaseState{};s.last=curTick();}pvObserve=true;
}
static void pvDiagEnd(ThreadContext*) {
 if(!pvObserve)return;
 for(unsigned id=0;id<8;++id){auto& s=pvPhases[id];s.ticks[s.phase]+=curTick()-s.last;
  std::cout<<"PV_TIMELINE core="<<id<<" start="<<pvBegin<<" end="<<curTick()<<" final_arrival="<<s.arrival<<std::endl;
  for(unsigned p=0;p<26;++p)std::cout<<"PV_PHASE core="<<id<<" phase="<<p<<" ticks="<<s.ticks[p]<<" entries="<<s.entries[p]<<std::endl;
 }pvObserve=false;
}
}}
