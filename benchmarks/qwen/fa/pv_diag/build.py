import json,sys,subprocess,shutil,hashlib,re
from pathlib import Path
R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent
ROOT=R/'results/cacheflex_flash_attention/pv_diag_20261005';G=ROOT/'guest_v1';B=ROOT/'build_v1';OLD=R/'results/cacheflex_flash_attention/pv_async_20261005/guest_v1'
G.mkdir(parents=True);B.mkdir()
for p in OLD.glob('*.hpp'):shutil.copy2(p,G/p.name)
s=(OLD/'driver.cpp').read_text().replace('static int async_pv=0;','static int async_pv=0;\nstatic int diag_flags=0;\n#define DIAG(n) asm volatile(".global pvdiag_" #n "_%=\\npvdiag_" #n "_%= ":\\n" ::: "memory")')
# Numeric phase labels; no additional guest instructions.
s=s.replace('"_%= ":\\n"','"_%=:\\n"')
s=s.replace('if(argc!=19)','if(argc!=20)').replace('    async_pv=atoi(argv[18]);','    async_pv=atoi(argv[18]);\n    diag_flags=atoi(argv[19]);if(diag_flags<0||diag_flags>5)return 2;\n    std::printf("PV_DIAG flags=%d\\n",diag_flags);')
s=s.replace(' ncc_sum(0x4e434350,10,bytes,0,0,0);',' DIAG(16);ncc_sum(0x4e434350,10,bytes,0,0,0);').replace('  auto* ptr=static_cast<const char*>(src)+off;','  DIAG(17);auto* ptr=static_cast<const char*>(src)+off;')
s=s.replace(' while(!ncc_sum(0x4e434350,4,0,0,0,0))',' DIAG(8);while(!ncc_sum(0x4e434350,4,0,0,0,0))').replace(' for(size_t off=0;off<bytes;off+=1024){\n  register uint64_t x0 asm("x0")=0x4e434350,x1 asm("x1")=12',' DIAG(9);for(size_t off=0;off<bytes;off+=1024){\n  register uint64_t x0 asm("x0")=0x4e434350,x1 asm("x1")=12')
a=s.index('static void* legacy_worker(');b=s.index('static void* ',a+15);x=s[a:b]
x=x.replace('        ncc_output(pv.p,2048);','        DIAG(8);ncc_output(pv.p,2048);').replace('        asm volatile("dsb sy\\nisb\\n":::"memory");','        DIAG(10);if(!(diag_flags&4))asm volatile("dsb sy\\nisb\\n":::"memory");',1).replace('        for(int a=0;a<16;++a){auto* dst=o+(pending_row+a)*64;','        DIAG(11);for(int a=0;a<16;++a){auto* dst=o+(pending_row+a)*64;').replace('        pending=false;++completed;','        DIAG(12);pending=false;++completed;')
for name,num in [('pack_k',1),('pack_q',3),('qk',5),('softmax',6),('pv',7),('normalize',22),('final_wait',24)]:x=x.replace('PHASE('+name+');',f'DIAG({num});PHASE('+name+');')
x=x.replace('    pthread_barrier_wait(&s.barrier);\n    const int groups=', '    DIAG(2);pthread_barrier_wait(&s.barrier);\n    const int groups=')
x=x.replace('            if(pending)std::abort();','            DIAG(4);if(pending)std::abort();')
x=x.replace('                        if(ncc_mode>=4)pv_read_traffic(pa.p,0,bc);','                        DIAG(13);if(ncc_mode>=4 && !(diag_flags&1))pv_read_traffic(pa.p,0,bc);').replace('                        asm volatile("dsb sy\\nisb\\n":::"memory");','                        DIAG(14);if(!(diag_flags&4))asm volatile("dsb sy\\nisb\\n":::"memory");',1).replace('                        ncc_input(pa.p,8192);','                        DIAG(15);ncc_input(pa.p,8192);\n                        DIAG(19);')
x=x.replace('                        if(!async_pv)finish_pv();\n                        continue;','                        if(!async_pv)finish_pv();\n                        DIAG(20);continue;')
x=x.replace('                    if(kernel_variant) {','                    DIAG(21);if(kernel_variant) {',1).replace('                        continue;\n                    }\n                    if(s.spm) cf_pv16_spm','                        DIAG(20);continue;\n                    }\n                    if(s.spm) cf_pv16_spm')
x=x.replace('    if(s.spm)release(', '    DIAG(23);if(s.spm)release(').replace('    if(pending || submitted!=completed)', '    DIAG(25);if(pending || submitted!=completed)')
s=s[:a]+x+s[b:];(G/'driver.cpp').write_text(s)
(G/'build.sh').write_text((OLD/'build.sh').read_text().replace(str(OLD),str(G)))
subprocess.run(['bash',str(G/'build.sh')],cwd=R,check=True)
sys.path.insert(0,str(R/'benchmarks/cacheflex_flash_attention'));from export_safe_elf import patch
patch(G/'guest',G/'guest.export-safe')
nm=subprocess.check_output(['nm','-n',str(G/'guest.export-safe')],text=True);mapping={}
for address,p in re.findall(r'^([0-9a-f]+) \w pvdiag_(\d+)_\d+$',nm,re.M):
 assert address not in mapping or mapping[address]==int(p),('ambiguous phase',address,mapping.get(address),p)
 mapping[address]=int(p)
assert {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25}<=set(mapping.values()),mapping
(G/'phase_map.txt').write_text(''.join(f'{a} {p}\n' for a,p in mapping.items()))
(G/'frozen.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in G.iterdir() if p.is_file()},indent=2))
s=(R/'results/cacheflex_flash_attention/pv_async_20261005/build_v1/pseudo_inst.cc').read_text();s=s.replace('#include "hook.hh"','#include "hook.hh"\n#include "observer.hh"')
for name,fn in [('workbegin','pvDiagStart'),('workend','pvDiagEnd')]:
 needle=name+'(ThreadContext *tc, uint64_t workid, uint64_t threadid)\n{';assert needle in s;s=s.replace(needle,needle+'\n    '+fn+'(tc);')
(B/'pseudo_inst.cc').write_text(s)
for name in ['hook.hh','observer.hh']:shutil.copy2(H/name,B/name)
print('Diagnostic guest built; passive PC map',len(mapping),'entries')
