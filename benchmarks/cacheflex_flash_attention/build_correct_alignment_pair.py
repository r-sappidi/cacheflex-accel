"""Preserve corrected arithmetic while isolating fixed-region loop placement."""
from pathlib import Path
import json,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
src=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();mode=sys.argv[3];out.mkdir()
if mode=='qk':
    pattern=r'(?m)^1:\nlsl x16, x22, #6\n\.inst 0xA5F08602\nld1rqh \{ z0.h \}, p0/z, \[x21\]\n.*?^bne 1b\n';offsets=[0,16]
else:
    assert mode=='pv'
    pattern=r'(?m)^1:\nld1rqh \{ z0.h \}, p0/z, \[x21\]\nld1h   \{ z2.h \}, p2/z, \[x22\]\n.*?^bne 1b\n';offsets=[8,24]
pcs=[];texts=[]
for offset in offsets:
    p=out/f'o{offset}';p.mkdir();bodies=[]
    def replace(m):
        bodies.append(m[0])
        return f'b .Lcf_entry\n.balign 64\n.Lcf_base:\n.space {offset},0\n.Lcf_entry:\n'+m[0]+'b .Lcf_end\n.space 128-(.-.Lcf_base),0\n.Lcf_end:\n'
    asm,n=re.subn(pattern,replace,(src/'guest.enc.s').read_text(),flags=re.S);assert n==1,n
    (p/'guest.s').write_text(asm)
    command=f'source "{ROOT}/setup_spm_env.sh" >/dev/null\n"$CROSS_CXX" -static -pthread "{p}/guest.s" "{src}/qk.o" "$M5OP_OBJ" -lm -o "{p}/guest.debug"'
    subprocess.run(['bash','-c',command],check=True)
    subprocess.run(['aarch64-linux-gnu-objcopy','--strip-debug',str(p/'guest.debug'),str(p/'guest')],check=True)
    subprocess.run(['aarch64-linux-gnu-objcopy','--dump-section',f'.text={p}/text.bin',str(p/'guest'),str(p/'text.elf')],check=True)
    dis=subprocess.check_output(['aarch64-linux-gnu-objdump','-d',str(p/'guest')],text=True).splitlines();starts=[]
    for i,line in enumerate(dis[:-1]):
        if mode=='qk':
            hit='a5f08602' in line and re.search(r'ld1rqh.*z0.*p0/z.*\[x21\]',dis[i+1]);delta=-4
        else:
            hit=re.search(r'ld1rqh.*z0.*p0/z.*\[x21\]',line) and re.search(r'ld1h.*z2.*p2/z.*\[x22\]',dis[i+1]);delta=0
        if hit:starts.append(int(line.split(':')[0],16)+delta)
    assert len(starts)==1 and starts[0]%64==offset,starts
    pcs.append(starts[0]-offset);texts.append((p/'text.bin').read_bytes())
    (p/'loop.json').write_text(json.dumps({'mode':mode,'offset':offset,'pc':starts[0],'body':bodies[0]},indent=2))
assert pcs[0]==pcs[1] and len(texts[0])==len(texts[1])
hdr=subprocess.check_output(['aarch64-linux-gnu-objdump','-h',str(out/f'o{offsets[0]}/guest')],text=True)
start=int(re.search(r'\s+\d+\s+\.text\s+[0-9a-f]+\s+([0-9a-f]+)',hdr)[1],16)
different=[start+i for i,(a,b) in enumerate(zip(*texts)) if a!=b]
outside=sorted(set(pc&~3 for pc in different if not pcs[0]<=pc<pcs[0]+128))
# The original aligned source has its own once-per-call b 1f. Both that
# branch and the inserted skip branch retarget the moved loop entry.
for pc in outside:
    assert pc<pcs[0]
    for data,offset in zip(texts,offsets):
        word=int.from_bytes(data[pc-start:pc-start+4],'little')
        assert word&0xfc000000==0x14000000,hex(word)
        imm=word&0x3ffffff
        if imm&(1<<25):imm-=1<<26
        assert pc+4*imm==pcs[0]+offset
(out/'pair_audit.json').write_text(json.dumps({'loop_base':pcs[0],'changed_bytes':len(different),'fixed_downstream_code':True,'offsets':offsets,'outside_region_changes_only_entry_branch_targets':outside},indent=2))
