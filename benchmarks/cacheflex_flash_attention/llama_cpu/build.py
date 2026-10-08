"""Build untouched pinned ggml/SVE and separately identified SVE GEMM port."""
import json,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=ROOT/'results/cacheflex_flash_attention/llama_cpu_20260925'
UP=BASE/'upstream'
PIN='4b1a27fa0eb875bbca4f6cfe936e3d65adc685c0'
TC=ROOT/'results/cacheflex_flash_attention/micro_toolchain_audit_20260918/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/bin'
sys.path.insert(0,str(ROOT/'scripts'))
from build_access_timing import sha,save

def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip()==PIN
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=UP,text=True).strip()
    for mode in ('upstream_sve','ported_sve'):
        out=BASE/mode;out.mkdir()
        shutil.copytree(UP/'ggml',out/'ggml')
        (out/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.14)\nproject(cacheflex_ggml C CXX ASM)\nadd_subdirectory(ggml)\n')
        shutil.copy2(HERE/'driver.cpp',out/'driver.cpp')
        src=out/'ggml/src/ggml-cpu/simd-gemm.h'
        before=sha(src)
        if mode=='ported_sve':
            text=src.read_text();needle='#elif defined(GGML_SIMD) && defined(__riscv_v_intrinsic)'
            assert text.count(needle)==1
            src.write_text(text.replace(needle,(HERE/'sve_gemm.inc').read_text()+'\n'+needle))
        compiler=str(TC/'aarch64-none-linux-gnu-g++')
        flags='-O3 -g -march=armv8.2-a+sve+fp16 -mbranch-protection=none'
        cmd=['cmake','-S',str(out),'-B',str(out/'cmake'),'-DCMAKE_SYSTEM_NAME=Linux',
             '-DCMAKE_SYSTEM_PROCESSOR=aarch64','-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_C_COMPILER='+str(TC/'aarch64-none-linux-gnu-gcc'),'-DCMAKE_CXX_COMPILER='+compiler,
             '-DCMAKE_C_FLAGS='+flags,'-DCMAKE_CXX_FLAGS='+flags,'-DBUILD_SHARED_LIBS=OFF',
             '-DGGML_NATIVE=OFF','-DGGML_CPU_ARM_ARCH=armv8.2-a+sve+fp16',
             '-DGGML_OPENMP=OFF','-DGGML_BLAS=OFF','-DGGML_CPU_KLEIDIAI=OFF',
             '-DGGML_BUILD_TESTS=OFF','-DGGML_BUILD_EXAMPLES=OFF','-DGGML_BACKEND_DL=OFF']
        save(out/'configure.json',cmd)
        with (out/'build.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run(['cmake','--build',str(out/'cmake'),'-j','8'],stdout=log,stderr=subprocess.STDOUT,check=True)
            libs=sorted((out/'cmake').rglob('*.a'))
            cmd=[compiler,*flags.split(),'-std=c++17','-static','-pthread',
                 '-I'+str(out/'ggml/include'),'-I'+str(ROOT/'gem5/include'),str(out/'driver.cpp'),
                 str(ROOT/'gem5/util/m5/build/arm64/abi/arm64/m5op.o'),
                 '-Wl,--start-group',*map(str,libs),'-Wl,--end-group','-lm','-ldl','-o',str(out/'guest')]
            save(out/'link.json',cmd);subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (out/'guest.dis').open('w') as f:
            subprocess.run([str(TC/'aarch64-none-linux-gnu-objdump'),'-d','-C',str(out/'guest')],stdout=f,check=True)
        save(out/'provenance.json',dict(upstream_commit=PIN,mode=mode,original_gemm_sha=before,
             compiled_gemm_sha=sha(src),compiler=subprocess.check_output([compiler,'--version'],text=True)))
        paths=[out/'guest',out/'driver.cpp',out/'provenance.json',*list((out/'ggml').rglob('*'))]
        save(out/'frozen.json',{str(p):sha(p) for p in paths if p.is_file()})
        print('BUILT',mode,flush=True)
if __name__=='__main__':main()
