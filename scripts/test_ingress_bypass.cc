#include <cassert>
#include <iostream>
#include "mem/ruby/network/IngressBypass.hh"
using namespace gem5::ruby;
int main() {
    const IngressRequestKey ld{true,false,0x1000,0};
    const IngressRequestKey independentLoad{true,false,0x2000,0};
    const IngressRequestKey cp{true,true,0x2000,0x8000};
    const IngressRequestKey cpOther{true,true,0x3000,0x9000};
    const IngressRequestKey sameSlot{true,true,0x4000,0x8000};
    const IngressRequestKey sameSource{true,true,0x2000,0x9000};
    const IngressRequestKey storeOrBarrier{false,false,0x5000,0};
    assert(ingressIndependent(independentLoad,ld));
    assert(!ingressIndependent(ld,ld));
    assert(ingressIndependent(cp,ld));
    assert(ingressIndependent(ld,cp));
    assert(ingressIndependent(cpOther,cp));
    assert(!ingressIndependent(sameSlot,cp));
    assert(!ingressIndependent(sameSource,cp));
    assert(!ingressIndependent(ld,storeOrBarrier));
    assert(!ingressIndependent(cp,storeOrBarrier));
    assert(!ingressIndependent(storeOrBarrier,ld));
    assert(!ingressIndependent(storeOrBarrier,cp));
    assert(ingressCopyHead(cp));
    assert(!ingressCopyHead(ld));
    assert(!ingressCopyHead(storeOrBarrier));
    assert(ingressDemandCandidate(ld));
    assert(!ingressDemandCandidate(cp));
    assert(!ingressDemandCandidate(storeOrBarrier));
    std::cout << "INGRESS_DEPENDENCY_TEST PASS checks=17\n";
}
