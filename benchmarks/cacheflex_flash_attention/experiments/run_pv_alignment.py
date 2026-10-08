"""PV loop placement factorial: same loop body and fixed-size code region.

Two branches per invocation skip unreachable padding. Every variant has the
same branch count and all code after the 128-byte region stays at the same PC.
Only the hot loop's offset within a 64-byte fetch block changes (8 vs 24).
"""
from _paths import FA_DIR
from pathlib import Path
import concurrent.futures as cf
import hashlib,json,os,re,subprocess,sys,time
from validate_attribution import read
from run_causal import audit
ROOT=FA_DIR.parents[1];B=ROOT/'results/cacheflex_flash_attention'
old=B/'residual_retry_20260911';study=B/'driver_causes_20260911'
out=Path(sys.argv[1]).resolve();out.mkdir()
state=dict(stage='waiting_for_launch_controls',completed=[],failed=None);rows=[]
def status(s):
    state['stage']=s;(out/'status.json').write_text(json.dumps(state,indent=2));print(s,flush=True)
def build(v,g,m,offset):
    b=study/f'build_{v}_g{g}_m{m}';p=out/f'build_{v}_m{m}_o{offset}';p.mkdir()
    text=(b/'1.enc.s').read_text();pattern=r'(?m)^1:\nld1rqh \{ z0.h \}, p0/z, \[x21\]\nld1h   \{ z2.h \}, p2/z, \[x22\]\n.*?^bne 1b\n'
    loops=[]
    def replace(match):
        n=len(loops);body=match[0];loops.append(body)
        return (f'b .Lcf_pv_entry{n}\n.balign 64\n.Lcf_pv_base{n}:\n.space {offset},0\n.Lcf_pv_entry{n}:\n'+body+
                f'b .Lcf_pv_end{n}\n.space 128-(.-.Lcf_pv_base{n}),0\n.Lcf_pv_end{n}:\n')
    text,n=re.subn(pattern,replace,text,flags=re.S)
    assert n==1,('expected exactly one PV inline body',n)
    (p/'guest.s').write_text(text)
    (p/'loop.json').write_text(json.dumps(dict(body=loops[0],offset=offset,region_bytes=128),indent=2))
    cmd=f'source "{ROOT}/setup_spm_env.sh" >/dev/null\n"$CROSS_CXX" -static -pthread "{p}/guest.s" "{b}/qk.o" "$M5OP_OBJ" -lm -o "{p}/guest.debug"'
    subprocess.run(['bash','-c',cmd],check=True)
    subprocess.run(['aarch64-linux-gnu-objcopy','--strip-debug',str(p/'guest.debug'),str(p/'guest')],check=True)
    dis=subprocess.check_output(['aarch64-linux-gnu-objdump','-d',str(p/'guest')],text=True)
    starts=[];lines=dis.splitlines()
    for i,l in enumerate(lines[:-1]):
        if re.search(r'ld1rqh.*z0.*p0/z.*\[x21\]',l) and re.search(r'ld1h.*z2.*p2/z.*\[x22\]',lines[i+1]):
            pc=int(l.split(':')[0],16);assert pc%64==offset;starts.append(pc)
    assert len(starts)==1
    (p/'loop_pc.json').write_text(json.dumps(starts))
    return p
def run(case):
    v,g,m,offset=case;b=build(v,g,m,offset);p=out/f'{v}_m{m}_o{offset}';p.mkdir()
    args=json.loads((old/f'original_{v}_t1024_p0/command.json').read_text())
    args=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in args];args[args.index('--cmd')+1]=str(b/'guest')
    args+=['-P',"system.cpu[0].workload[0].cmd=['cacheflex-fa','1024','64','64','128']"]
    (p/'command.json').write_text(json.dumps(args,indent=2))
    with (p/'run.log').open('w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200,
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
    o,c,cfg=read(p,16);ref=read(old/'original_cache_t1024_p0',16)
    assert o['hash']==ref[0]['hash'];audit(ref[2],cfg,0)
    assert c['m_spmcpRequestsIssued']==c['m_spmcpRequestsCompleted']==(65536 if v=='spm' else 0)
    return dict(variant=v,mask=m,offset=offset,counters=c,output=o,path=str(p))
try:
    status('waiting_for_launch_controls')
    while not (B/'driver_launch_20260911/summary.json').exists():time.sleep(10)
    for m in [0,1]:
        for v,g in [('spm',4),('cache',1)]:
            status(f'{v}_m{m}_offset_pair')
            with cf.ThreadPoolExecutor(max_workers=2) as pool:rows+=list(pool.map(run,[(v,g,m,o) for o in [8,24]]))
            (out/'progress.json').write_text(json.dumps(rows,indent=2));state['completed'].append(state['stage'])
    (out/'summary.json').write_text(json.dumps(rows,indent=2));status('complete')
except BaseException as e:
    state['failed']=repr(e);status('failed');raise
