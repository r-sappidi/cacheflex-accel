import json,sys,subprocess,shutil,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent
ROOT=R/'results/cacheflex_flash_attention/pv_async_20261005';G=ROOT/'guest_v1';B=ROOT/'build_v1';OLD=R/'results/cacheflex_flash_attention/ncc_probe_20261005_v5/guest_v1'
G.mkdir(parents=True);B.mkdir()
for p in OLD.glob('*.hpp'):shutil.copy2(p,G/p.name)
s=(OLD/'driver.cpp').read_text().replace('static int ncc_mode=0;','static int ncc_mode=0;\nstatic int async_pv=0;')
a=s.index('static void* legacy_worker(');b=s.index('static void* ',a+15);body=s[a:b]
marker='    pthread_barrier_wait(&s.barrier);'
helper='''    bool pending=false;int pending_row=0;float pending_alpha[16];
    size_t submitted=0,completed=0;
    auto finish_pv=[&]() {
        if(!pending)return;
        ncc_output(pv.p,2048);
        asm volatile("dsb sy\\nisb\\n":::"memory");
        for(int a=0;a<16;++a){auto* dst=o+(pending_row+a)*64;
            auto contribution=cf_dense_f16_to_f32(pf,svld1_f16(ph,pv.p+a*64));
            auto scaled=svmul_n_f32_x(pf,svld1_f32(pf,dst),pending_alpha[a]);
            svst1_f32(pf,dst,svadd_f32_x(pf,scaled,contribution));
        }
        pending=false;++completed;
    };
'''
body=body.replace(marker,helper+marker,1)
old='''            if(s.spm && !s.kv) stage(vhead+j*64,bc,s.sets);'''
new='''            if(pending)std::abort(); // V cannot be replaced while an engine owns it.
            if(ncc_mode>=3 && ncc_mode<=8)ncc_sum(0x4e434350,15,0,0,0,ncc_mode);
'''+old
assert body.count(old)==1;body=body.replace(old,new)
old='''                PHASE(qk);''';assert body.count(old)==1
body=body.replace(old,'''                if(ncc_mode>=3 && ncc_mode<=8)ncc_sum(0x4e434350,13,0,0,0,ncc_mode);
'''+old)
start=body.index('                    if(ncc_mode>=3 && ncc_mode<=8) {');end=body.index('                    if(ncc_mode==9)',start)
body=body[:start]+'''                    if(ncc_mode>=3 && ncc_mode<=8) {
                        // QK, softmax and P packing above ran while the previous
                        // row block's PV was pending. Retire before buffer reuse.
                        ncc_sum(0x4e434350,14,0,0,0,ncc_mode);
                        finish_pv();
                        if(ncc_mode>=4)pv_read_traffic(pa.p,0,bc);
                        asm volatile("dsb sy\\nisb\\n":::"memory");
                        ncc_input(pa.p,8192);
                        for(int a=0;a<16;++a)pending_alpha[a]=alpha.p[r+a];
                        pending_row=bi+r;
                        ncc_sum(0x4e434350,3,0,0,0,ncc_mode);
                        pending=true;++submitted;
                        if(!async_pv)finish_pv();
                        continue;
                    }
'''+body[end:]
old='''                }
            }
        }
        if(s.kv) {'''
assert body.count(old)==1
body=body.replace(old,'''                }
            }
            finish_pv(); // Drain before staging another V tile / normalization.
        }
        if(s.kv) {''')
old='    if(ncc_mode)ncc_sum(0x4e434350,9,0,0,0,ncc_mode);';assert old in body
body=body.replace(old,'''    if(pending || submitted!=completed)std::abort();
    std::printf("ASYNC_WORK core=%d submitted=%zu completed=%zu\\n",c.id,submitted,completed);
'''+old)
s=s[:a]+body+s[b:];s=s.replace('if(argc!=18)','if(argc!=19)')
old='    ncc_mode=atoi(argv[17]);';s=s.replace(old,old+'''
    async_pv=atoi(argv[18]);
    if(async_pv<0||async_pv>1||(async_pv && ncc_mode!=6 && ncc_mode!=7))return 2;
    std::printf("ASYNC_CONFIG enabled=%d depth=1 alpha_bytes=64\\n",async_pv);
''')
(G/'driver.cpp').write_text(s);(G/'build.sh').write_text((OLD/'build.sh').read_text().replace(str(OLD),str(G)))
subprocess.run(['bash',str(G/'build.sh')],cwd=R,check=True)
sys.path.insert(0,str(R/'benchmarks/cacheflex_flash_attention'));from export_safe_elf import patch
patch(G/'guest',G/'guest.export-safe')
(G/'frozen.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in G.iterdir() if p.is_file()},indent=2))
old=R/'results/cacheflex_flash_attention/ncc_probe_20261005_v5/build_v1/pseudo_inst.cc'
(B/'pseudo_inst.cc').write_text(old.read_text());shutil.copy2(H/'hook.hh',B/'hook.hh')
print('Asynchronous guest and hook prepared')
