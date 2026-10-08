// Minimal scalable-vector port of ggml's existing C += A*B helper.
// Same FP32 arithmetic, panel layout, loop traversal and upstream threadpool.
// Named registers avoid arrays of sizeless SVE types. Predicate handles tails.
#elif defined(GGML_SIMD) && defined(__ARM_FEATURE_SVE)
static void simd_gemm(float * GGML_RESTRICT C,
                     const float * GGML_RESTRICT A,
                     const float * GGML_RESTRICT B,int M,int K,int N) {
    int i=0;
    for(;i+8<=M;i+=8) for(int j=0;j<N;j+=svcntw()) {
        const svbool_t pg=svwhilelt_b32(j,N);
        svfloat32_t c0=svld1_f32(pg,C+(i+0)*N+j),c1=svld1_f32(pg,C+(i+1)*N+j);
        svfloat32_t c2=svld1_f32(pg,C+(i+2)*N+j),c3=svld1_f32(pg,C+(i+3)*N+j);
        svfloat32_t c4=svld1_f32(pg,C+(i+4)*N+j),c5=svld1_f32(pg,C+(i+5)*N+j);
        svfloat32_t c6=svld1_f32(pg,C+(i+6)*N+j),c7=svld1_f32(pg,C+(i+7)*N+j);
        for(int k=0;k<K;++k) {
            const auto b=svld1_f32(pg,B+k*N+j);
            c0=svmla_n_f32_m(pg,c0,b,A[(i+0)*K+k]);c1=svmla_n_f32_m(pg,c1,b,A[(i+1)*K+k]);
            c2=svmla_n_f32_m(pg,c2,b,A[(i+2)*K+k]);c3=svmla_n_f32_m(pg,c3,b,A[(i+3)*K+k]);
            c4=svmla_n_f32_m(pg,c4,b,A[(i+4)*K+k]);c5=svmla_n_f32_m(pg,c5,b,A[(i+5)*K+k]);
            c6=svmla_n_f32_m(pg,c6,b,A[(i+6)*K+k]);c7=svmla_n_f32_m(pg,c7,b,A[(i+7)*K+k]);
        }
        svst1_f32(pg,C+(i+0)*N+j,c0);svst1_f32(pg,C+(i+1)*N+j,c1);
        svst1_f32(pg,C+(i+2)*N+j,c2);svst1_f32(pg,C+(i+3)*N+j,c3);
        svst1_f32(pg,C+(i+4)*N+j,c4);svst1_f32(pg,C+(i+5)*N+j,c5);
        svst1_f32(pg,C+(i+6)*N+j,c6);svst1_f32(pg,C+(i+7)*N+j,c7);
    }
    for(;i<M;++i) for(int j=0;j<N;j+=svcntw()) {
        const auto pg=svwhilelt_b32(j,N);auto c=svld1_f32(pg,C+i*N+j);
        for(int k=0;k<K;++k)c=svmla_n_f32_m(pg,c,svld1_f32(pg,B+k*N+j),A[i*K+k]);
        svst1_f32(pg,C+i*N+j,c);
    }
}
