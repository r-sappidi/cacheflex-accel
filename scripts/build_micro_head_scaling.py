"""Preserve MICRO head kernels; add contiguous KV-group-owned CPU scheduling.

Original corrected single-core drivers are built as controls. The common
pthread harness uses identical per-head bodies at 1/4/8 cores, with independent
workspaces, shared read-only inputs, and disjoint output heads. No hardware edits.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FA = ROOT / 'benchmarks/cacheflex_flash_attention'
BASE = ROOT / 'results/cacheflex_flash_attention'
UP = BASE / 'micro_upstream_20260918'
OLD = BASE / 'micro_gcc15_corrected_gate_20260918/builds'
CXX = BASE / 'micro_toolchain_audit_20260918/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def transform(source, arch):
    prefix, main = source.split('int main(int argc, char** argv)', 1)
    alloc = main.index('    AlignedBuffer<__fp16> Q_buf(')
    start = main.rfind('    //', 0, alloc)
    setup = main[:start]
    shared_end = main.index('\n', main.index('    AlignedBuffer<__fp16> O_buf(')) + 1
    shared = main[alloc:shared_end]
    if arch == 'spm':
        init = main.index('    // Init random data')
        init_end = main.index('    // Flush cache', init)
        loop_start = main.index('        for (int hq = 0; hq < H_q; ++hq)')
        loop_end = main.index('\n        double us = us_since(t0);', loop_start)
        times = 'double t_spmcp=0,t_qk=0,t_sm=0,t_pv=0,t_fin=0,t_pack=0; auto _now=[]{return Clock::now();};'
    else:
        init = main.index('    {\n        // Same seed pattern')
        init_end = main.index('    // ── Prefill:', init)
        loop_start = main.index('        for (int hq = 0; hq < H_q; ++hq)')
        loop_end = main.index('        ROI_END();', loop_start)
        times = 'FlashTiming cold_ft; double t_kprep=0;'
    workspace = main[shared_end:init]
    loop = main[loop_start:loop_end]
    assert loop.count('for (int hq = 0; hq < H_q; ++hq)') == 1
    loop = loop.replace('for (int hq = 0; hq < H_q; ++hq)',
                        'for (int hq = head_begin; hq < head_end; ++hq)')
    # Whole KV groups stay on one worker, preserving transpose/pack reuse.
    result = '#include <thread>\n#include <vector>\n#include <pthread.h>\n' + prefix
    result += 'int main(int argc, char** argv)' + setup
    result += '''
    const int cores = argc > 9 ? std::atoi(argv[9]) : 1;
    if (T<=0 || d!=64 || Br<=0 || Br%8 || Bc<=0 || Bc%128 ||
        H_kv<=0 || (cores!=1 && cores!=4 && cores!=8) || H_kv%cores ||
        H_q%H_kv || N_ITER!=1) return 2;
    printf("CF_SCALING cores=%d T=%d Hq=%d Hkv=%d Br=%d Bc=%d\\n",cores,T,H_q,H_kv,Br,Bc);
'''
    result += shared + main[init:init_end]
    result += '''
    pthread_barrier_t barrier;
    if(pthread_barrier_init(&barrier,nullptr,cores)) return 3;
    auto worker = [&](int tid) {
        const int head_begin = tid*(H_q/cores), head_end=(tid+1)*(H_q/cores);
'''
    result += workspace + '\n' + times
    result += '''
        printf("CF_WORKER tid=%d begin=%d end=%d\\n",tid,head_begin,head_end);
        pthread_barrier_wait(&barrier);
        if(tid==0) { m5_reset_stats(0,0); ROI_BEGIN(); }
        pthread_barrier_wait(&barrier);
'''
    result += loop
    result += '''
        asm volatile("dsb sy" ::: "memory");
        pthread_barrier_wait(&barrier);
        if(tid==0) { ROI_END(); m5_dump_stats(0,0); }
        pthread_barrier_wait(&barrier);
        printf("CF_DONE tid=%d heads=%d\\n",tid,head_end-head_begin);
    };
    std::vector<std::thread> workers;
    for(int tid=1;tid<cores;++tid) workers.emplace_back(worker,tid);
    worker(0);
    for(auto& thread:workers) thread.join();
    pthread_barrier_destroy(&barrier);
    cf_dump("Q.fp16",Q_buf.data(),(size_t)H_q*T*d*2);
    cf_dump("K.fp16",K_buf.data(),(size_t)H_kv*T*d*2);
    cf_dump("V.fp16",V_buf.data(),(size_t)H_kv*T*d*2);
    cf_dump("O.fp16",O_buf.data(),(size_t)H_q*T*d*2);
    return 0;
}
'''
    return result, dict(original_head_body_sha256=hashlib.sha256(main[loop_start:loop_end].encode()).hexdigest(),
                       changed_head_body='Only hq loop bounds; inner arithmetic, packing, timing and staging unchanged',
                       mapping='Contiguous whole KV groups; shared Q/K/V arrays, private workspace, disjoint O',
                       roi='First invocation including packing, all head computation and start/end barriers; excludes thread/workspace creation and exports')


def build(out):
    out.mkdir()
    manifest = {}
    for arch in ('cache','spm'):
        srcdir = OLD / f'{arch}_vl16'
        for mode in ('original','parallel'):
            dest = out / f'{arch}_{mode}'; dest.mkdir()
            for p in srcdir.glob('*.hpp'):
                shutil.copy2(p,dest/p.name); manifest[str(p)] = sha(p)
            src = srcdir/'driver.cpp'; source=src.read_text(); manifest[str(src)] = sha(src)
            if mode=='parallel':
                source, audit = transform(source,arch)
                (dest/'transform.json').write_text(json.dumps(audit,indent=2))
            (dest/'driver.cpp').write_text(source)
            # Same proven filename refresh as the old binary patch, linked
            # directly at build time; executes exclusively after the ROI.
            exporter=(FA/'fa_export_hook.S').read_text()
            assert exporter.count('    .inst 0xff4f0110')==2
            exporter=exporter.replace('    .inst 0xff4f0110','    bl filename_each_chunk',1)
            (dest/'export.S').write_text(exporter+'\n'+(FA/'export_filename_each_chunk.S').read_text())
            common=[str(CXX),'-O3','-g','-std=c++17','-march=armv8.2-a+sve+fp16',
                    '-mbranch-protection=none','-DGEM5','-DGEM5_SE','-DVL_16',
                    '-include',str(FA/'correct_fa_conversions.hpp'),'-I'+str(ROOT/'gem5/include'),
                    '-I'+str(dest),'-I'+str(UP/'kernels/llama_bench_spm'),
                    '-I'+str(UP/'kernels/attention_kernels_spm')]
            m5 = ROOT/'gem5/util/m5/build/arm64/out/m5op.o'
            if not m5.exists(): m5=ROOT/'gem5/util/m5/build/arm64/abi/arm64/m5op.o'
            commands=[common+['-S',str(dest/'driver.cpp'),'-o',str(dest/'guest.s')],
                      ['python3',str(ROOT/'spm_tools/spm_compiler.py'),str(dest/'guest.s'),str(dest/'guest.enc.s')],
                      [str(CXX),'-c',str(dest/'export.S'),'-o',str(dest/'export.o')],
                      [str(CXX),'-static','-pthread',str(dest/'guest.enc.s'),str(dest/'export.o'),str(m5),'-lm','-o',str(dest/'guest')]]
            (dest/'commands.json').write_text(json.dumps(commands,indent=2))
            with (dest/'build.log').open('w') as log:
                for command in commands: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            manifest[str(dest/'guest')]=sha(dest/'guest')
    for p in (CXX,FA/'correct_fa_conversions.hpp',FA/'fa_export_hook.S',FA/'export_filename_each_chunk.S',ROOT/'spm_tools/spm_compiler.py',Path(__file__)):
        manifest[str(p)]=sha(p)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    import sys
    build(Path(sys.argv[1]).resolve())
