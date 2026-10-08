#pragma once
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include "base/statistics.hh"
#include "cpu/o3/cpu.hh"
#include "cpu/o3/dyn_inst.hh"
namespace gem5 { namespace o3 { namespace dsbdiag {
inline bool enabled(){static bool e=std::getenv("CACHEFLEX_DSB_DIAG")!=nullptr;return e;}
inline bool isDsb(const DynInstPtr &i){return i && i->staticInst->getName()=="dsb";}
// Return bridge=4, private SPM=2, ordinary coherent=1. Mnemonic fallback
// matters before execute has populated dynamic request flags.
inline unsigned domain(const DynInstPtr &i){
 auto f=i->memReqFlags;auto n=i->staticInst->getName();
 if((f&Request::CP_TO_SPM)||(f&Request::SPM_WRITEBACK)||n.find("spmcp")!=std::string::npos||n.find("spmwb")!=std::string::npos)return 4;
 if((f&Request::SPMLD)||(f&Request::SPMST)||n.find("spm")!=std::string::npos)return 2;
 return 1;
}
struct Counters:statistics::Group {
 statistics::Scalar arrivals,retired,headSamples,storeWaitSamples,widthDeferredSamples;
 statistics::Vector waitMask,arrivalRequests;
 statistics::Scalar youngerCoherentDependencies,youngerSpmDependencies,youngerBridgeDependencies;
 statistics::Scalar renameWaitSamples,renameCoherentFrontSamples,renameCoherentFrontUnique;
 Counters(CPU *c):statistics::Group(c,"dsbDiag"),
 ADD_STAT(arrivals,statistics::units::Count::get(),"Unique DSBs observed unexecuted at ROB head"),
 ADD_STAT(retired,statistics::units::Count::get(),"Retired DSB instructions"),
 ADD_STAT(headSamples,statistics::units::Cycle::get(),"Distinct CPU ticks attempting unexecuted DSB commit; sampled not additive runtime"),
 ADD_STAT(storeWaitSamples,statistics::units::Cycle::get(),"DSB commit attempts blocked by store/bridge/reduction drain, excluding commit-width deferral"),
 ADD_STAT(widthDeferredSamples,statistics::units::Cycle::get(),"DSB commit attempts deferred after another instruction committed this cycle"),
 ADD_STAT(waitMask,statistics::units::Cycle::get(),"Exclusive drain categories: coherent1 SPM2 bridge4;0 is unclassified"),
 ADD_STAT(arrivalRequests,statistics::units::Count::get(),"Older incomplete LSQ/shadow operations summed over DSB arrivals; coherent/SPM/bridge, deduplicated by sequence"),
 ADD_STAT(youngerCoherentDependencies,statistics::units::Count::get(),"Ordinary memrefs inserted with a DSB dependency; includes speculative, not removable cycles"),
 ADD_STAT(youngerSpmDependencies,statistics::units::Count::get(),"SPM memrefs inserted with a DSB dependency"),
 ADD_STAT(youngerBridgeDependencies,statistics::units::Count::get(),"Bridge memrefs inserted with a DSB dependency"),
 ADD_STAT(renameWaitSamples,statistics::units::Cycle::get(),"Distinct rename ticks stalled by serialization attributed to a preceding DSB"),
 ADD_STAT(renameCoherentFrontSamples,statistics::units::Cycle::get(),"DSB serialization samples with ordinary coherent memref at rename front; overlap with commit samples"),
 ADD_STAT(renameCoherentFrontUnique,statistics::units::Count::get(),"Distinct visible coherent memrefs stalled at rename by DSB serialization") {
 waitMask.init(8);arrivalRequests.init(3);
 arrivalRequests.subname(0,"coherent");arrivalRequests.subname(1,"spm");arrivalRequests.subname(2,"bridge");
 }
};
struct State {Counters *stats=nullptr;InstSeqNum seen=0,probe=0,serialize=0,front=0;ThreadID tid=0;Tick headTick=~Tick(0),renameTick=~Tick(0);unsigned mask=0;unsigned counts[3]={0,0,0};};
inline std::map<CPU*,State> states;
inline State &state(CPU*c){auto &s=states[c];if(!s.stats)s.stats=new Counters(c);return s;}
inline std::map<std::string,CPU*> memdepCpu;
inline void begin(CPU*c,const DynInstPtr&i){auto&s=state(c);s.probe=i->seqNum;s.tid=i->threadNumber;s.mask=0;for(auto &v:s.counts)v=0;}
inline void finish(CPU*c,const DynInstPtr&i,unsigned instNum,bool waiting){
 auto&s=state(c);s.probe=0;
 if(s.headTick==curTick())return;s.headTick=curTick();++s.stats->headSamples;
 if(s.seen!=i->seqNum){s.seen=i->seqNum;++s.stats->arrivals;for(int j=0;j<3;++j)s.stats->arrivalRequests[j]+=s.counts[j];}
 if(instNum)++s.stats->widthDeferredSamples;
 else if(waiting){++s.stats->storeWaitSamples;++s.stats->waitMask[s.mask];}
}
} } }
