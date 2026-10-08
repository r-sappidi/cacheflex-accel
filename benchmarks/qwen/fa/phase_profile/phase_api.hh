#pragma once
#include <cstdint>
namespace gem5 {
void faCopyBegin(unsigned core,uint64_t key);
void faCopyEnd(unsigned core,uint64_t key);
void faRawBegin(unsigned core,uint64_t key);
void faRawEnd(unsigned core,uint64_t key,bool squashed);
void faMemBegin(unsigned core,uint64_t key,unsigned bytes,bool spm);
void faMemEnd(unsigned core,uint64_t key);
void faPortBlock(unsigned core,bool spm);
}
