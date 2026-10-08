#pragma once
#include <cstdlib>
#include <map>
#include "base/statistics.hh"
#include "cpu/o3/cpu.hh"
namespace gem5 { namespace o3 { namespace spmisolation {
inline unsigned budget() {
 static const unsigned value=[](){
  const char *s=std::getenv("CACHEFLEX_COPY_SHADOW_BUDGET");
  unsigned v=s?std::strtoul(s,nullptr,10):0;
  fatal_if(v!=0 && v!=96 && v!=4096,"Invalid diagnostic copy-shadow budget");
  return v;
 }();return value;
}
struct Counters:statistics::Group {
 statistics::Scalar samples,baselineFull,relieved,maxCopyShadows,maxTotalOccupancy;
 Counters(CPU*c):statistics::Group(c,"spmIsolation"),
 ADD_STAT(samples,statistics::units::Cycle::get(),"Distinct ticks querying SPM SQ availability; not exposed stall cycles"),
 ADD_STAT(baselineFull,statistics::units::Cycle::get(),"Sampled ticks where original shared SPM SQ budget would be full"),
 ADD_STAT(relieved,statistics::units::Cycle::get(),"Sampled ticks with original budget full but diagnostic admission possible"),
 ADD_STAT(maxCopyShadows,statistics::units::Count::get(),"Peak copy shadows, dependency records retained until original completion"),
 ADD_STAT(maxTotalOccupancy,statistics::units::Count::get(),"Peak real stores plus all bridge shadows, not charged occupancy") {}
};
struct State { Counters *stats=nullptr;Tick tick=~Tick(0); };
inline std::map<CPU*,State> states;
inline State& state(CPU*c){auto&s=states[c];if(!s.stats)s.stats=new Counters(c);return s;}
} } }
