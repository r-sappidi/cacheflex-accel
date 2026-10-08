#include <cmath>
#define sqrtf sqrt
#define main cacheflex_micro_cache_flash_single_main
#ifdef CACHEFLEX_FLASH_VL16
#include "../../../cacheflex_micro/experiments/end2end/src/cache_flash_attn.cpp"
#define NT_VL4 get_NT()
#else
#include "../../../cacheflex_micro/experiments/flash_attention/src/cache_flash_attn.cpp"
#endif
#undef main
#undef sqrtf

#include <pthread.h>
#include <atomic>

#ifndef CACHEFLEX_FLASH_PRETOUCH
#define CACHEFLEX_FLASH_PRETOUCH 0
#endif

namespace {
constexpr int kCores = 8;

struct Shared {
    int T, d, Br, Bc;
    float scale;
    AlignedBuffer<__fp16> Q, K, V, O, KT, Kknm;
    pthread_barrier_t barrier;
    std::atomic<int> failures{0};

    Shared(int t, int dim, int br, int bc)
        : T(t), d(dim), Br(br), Bc(bc),
          scale(1.0f / std::sqrt((float)dim)),
          Q((size_t)kCores*t*dim), K((size_t)t*dim),
          V((size_t)t*dim), O((size_t)kCores*t*dim),
          KT((size_t)dim*t),
          Kknm(((size_t)t + NT_VL4 - 1) / NT_VL4 * dim * NT_VL4)
    {
        pthread_barrier_init(&barrier, nullptr, kCores);
    }
    ~Shared() { pthread_barrier_destroy(&barrier); }
};

static __fp16 pattern(size_t i, unsigned salt)
{
    // Select from exact, finite FP16 constants without guest FP conversion.
    static const uint16_t bits[] = {
        0x2c00, 0xac00, 0x3000, 0xb000, 0x2800, 0xa800, 0x3400, 0xb400
    };
    uint16_t b = bits[(i * 5 + salt * 3) & 7];
    __fp16 v;
    std::memcpy(&v, &b, sizeof(v));
    return v;
}

struct ThreadCtx { Shared *s; int tid; };

static void *worker(void *opaque)
{
    auto &ctx = *static_cast<ThreadCtx *>(opaque);
    Shared &s = *ctx.s;
    const int T=s.T, d=s.d, Br=s.Br, Bc=s.Bc;
    const size_t ntiles=((size_t)T+NT_VL4-1)/NT_VL4;
    const size_t ablocks=((size_t)Br+MT-1)/MT;

    AlignedBuffer<__fp16> qtile((size_t)Br*d), spanel(ablocks*MT*Bc),
        ptile(ablocks*MT*Bc), pvpanel(8*d), qkap(ablocks*MT*d),
        pvap(8*Bc);
    AlignedBuffer<float> oacc((size_t)Br*d), mvec(Br), lvec(Br);

    if (CACHEFLEX_FLASH_PRETOUCH) {
        std::memset(qtile.data(),0,(size_t)Br*d*2);
        std::memset(spanel.data(),0,ablocks*MT*Bc*2);
        std::memset(ptile.data(),0,ablocks*MT*Bc*2);
        std::memset(pvpanel.data(),0,8*d*2);
        std::memset(qkap.data(),0,ablocks*MT*d*2);
        std::memset(pvap.data(),0,8*Bc*2);
        std::memset(oacc.data(),0,(size_t)Br*d*4);
        std::memset(mvec.data(),0,(size_t)Br*4);
        std::memset(lvec.data(),0,(size_t)Br*4);
    }

    std::printf("FLASH_ADDR tid=%d qtile=%p spanel=%p ptile=%p pvpanel=%p qkap=%p pvap=%p oacc=%p m=%p l=%p\n",
        ctx.tid, qtile.data(), spanel.data(), ptile.data(), pvpanel.data(),
        qkap.data(), pvap.data(), oacc.data(), mvec.data(), lvec.data());

    std::printf("FLASH_THREAD_READY tid=%d\n", ctx.tid);
    pthread_barrier_wait(&s.barrier);
    if (ctx.tid == 0) {
        std::printf("FLASH_ROI begin rpns=%llu\n", (unsigned long long)m5_rpns());
        m5_reset_stats(0, 0);
        m5_work_begin(0, 0);
        transpose_K(s.K.data(), s.KT.data(), T, d);
        prepack_B_knm(s.KT.data(), s.Kknm.data(), d, T, d, T);
    }
    pthread_barrier_wait(&s.barrier);

    flash_attn_head_nospm(
        s.Q.data()+(size_t)ctx.tid*T*d, s.K.data(), s.V.data(),
        s.O.data()+(size_t)ctx.tid*T*d, T, d, s.scale, Br, Bc,
        s.KT.data(), s.Kknm.data(), qtile.data(), spanel.data(),
        ptile.data(), pvpanel.data(), qkap.data(), pvap.data(),
        oacc.data(), mvec.data(), lvec.data(), nullptr);

    pthread_barrier_wait(&s.barrier);
    if (ctx.tid == 0) {
        m5_work_end(0, 0);
        m5_dump_stats(0, 0);
        std::printf("FLASH_ROI end rpns=%llu\n", (unsigned long long)m5_rpns());
    }
    return nullptr;
}
}

int main(int argc, char **argv)
{
    setbuf(stdout, nullptr);
#ifdef CACHEFLEX_FLASH_VL16
    if (svcnth() != 128) return 2;
#endif
    const int T=argc>1?std::atoi(argv[1]):256;
    const int d=argc>2?std::atoi(argv[2]):64;
    const int Br=argc>3?std::atoi(argv[3]):64;
    int Bc=argc>4?std::atoi(argv[4]):96;
    Bc=((Bc+(int)NT_VL4-1)/(int)NT_VL4)*(int)NT_VL4;
    if (d != 64 || T <= 0 || Br <= 0) return 2;

    Shared s(T,d,Br,Bc);
    std::printf("FLASH_SHARED Q=%p K=%p V=%p O=%p KT=%p Kpack=%p\n",
        s.Q.data(),s.K.data(),s.V.data(),s.O.data(),s.KT.data(),s.Kknm.data());
    for (size_t i=0;i<(size_t)kCores*T*d;++i) s.Q.data()[i]=pattern(i,1);
    for (size_t i=0;i<(size_t)T*d;++i) {
        s.K.data()[i]=pattern(i,2); s.V.data()[i]=pattern(i,3);
    }
    std::memset(s.O.data(),0,(size_t)kCores*T*d*sizeof(__fp16));

    pthread_t threads[kCores]; ThreadCtx ctx[kCores];
    ctx[0]={&s,0};
    for(int i=1;i<kCores;++i){ctx[i]={&s,i};pthread_create(&threads[i],nullptr,worker,&ctx[i]);}
    worker(&ctx[0]);
    for(int i=1;i<kCores;++i)pthread_join(threads[i],nullptr);
    double checksum=0;
    uint64_t hash=1469598103934665603ULL;
    size_t nonfinite=0;
    for(size_t i=0;i<(size_t)kCores*T*d;++i) {
        uint16_t bits;
        std::memcpy(&bits, &s.O.data()[i], sizeof(bits));
        checksum+=(float)s.O.data()[i];
        hash=(hash^bits)*1099511628211ULL;
        nonfinite+=((bits&0x7c00)==0x7c00);
    }
    std::printf("FLASH_RESULT variant=cache cores=8 T=%d Hq=8 Hkv=1 d=%d Br=%d Bc=%d checksum=%.10g hash=%016llx nonfinite=%zu\n",
                T,d,Br,Bc,checksum,(unsigned long long)hash,nonfinite);
    return 0;
}
