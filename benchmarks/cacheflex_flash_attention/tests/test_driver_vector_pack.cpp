#include <algorithm>
#include <arm_sve.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "../driver_vector_pack.hpp"

int main() {
    if (svcnth()!=128) return 2;
    size_t errors=0,cases=0;
    for(size_t rows: {size_t(1),size_t(7),size_t(8),size_t(9),size_t(16)})
    for(size_t kc: {size_t(64),size_t(127),size_t(128),size_t(129),size_t(256)}) {
        size_t lda=kc+3, blocks=(rows+7)/8;
        std::vector<__fp16> src(rows*lda),out(blocks*8*kc);
        for(size_t i=0;i<src.size();++i) {
            uint16_t bits=uint16_t((i*11939+71)%65536);
            std::memcpy(&src[i],&bits,2);
        }
        cf_vector_pack_A(src.data(),lda,out.data(),rows,0,kc+1,1,rows,kc);
        for(size_t b=0;b<blocks;++b) for(size_t k=0;k<kc;++k) for(size_t r=0;r<8;++r) {
            uint16_t expected=0,actual;
            if(b*8+r<rows)std::memcpy(&expected,&src[(b*8+r)*lda+k+1],2);
            std::memcpy(&actual,&out[(b*kc+k)*8+r],2);
            errors+=actual!=expected;
        }
        ++cases;
    }
    std::printf("PACK_REFERENCE cases=%zu errors=%zu\n",cases,errors);
    return errors?1:0;
}
