"""Isolated guard-only 32K guest; preserve all existing live/frozen guests."""
import json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc')
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'benchmarks/cacheflex_flash_attention')]
from build_access_timing import sha,save
from export_safe_elf import patch
OLD=ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928/guest_v4'
BASE=ROOT/'results/cacheflex_l3_spm/qwen_long_context_20260928'
OUT=BASE/'guest_v2'
def main():
    OUT.mkdir()
    source=(OLD/'driver.cpp').read_text();assert source.count('t>16384')==1
    modified=source.replace('t>16384','t>32768')
    assert modified.replace('t>32768','t>16384')==source
    (OUT/'driver.cpp').write_text(modified)
    for p in OLD.glob('*.hpp'):shutil.copy2(p,OUT/p.name)
    (OUT/'build.sh').write_text((OLD/'build.sh').read_text().replace(str(OLD),str(OUT)))
    with (OUT/'build.log').open('w') as f:subprocess.run(['bash',str(OUT/'build.sh')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    patch(OUT/'guest',OUT/'guest.export-safe')
    # Compare executable .text bytes: only the range-guard immediate may differ.
    for label,p in [('old',OLD/'guest.export-safe'),('new',OUT/'guest.export-safe')]:
        subprocess.run(['aarch64-linux-gnu-objcopy','-O','binary','--only-section=.text',str(p),str(OUT/(label+'.text'))],check=True)
    a=(OUT/'old.text').read_bytes();b=(OUT/'new.text').read_bytes();assert len(a)==len(b)
    diffs=[i for i in range(0,len(a),4) if a[i:i+4]!=b[i:i+4]]
    assert len(diffs)==1,('unexpected instruction changes',diffs[:30])
    # Compiler folds positive bound into unsigned (t-1) <= limit-1.
    # MOV W0, #16383 -> MOV W0, #32767; same compare and branch.
    i=diffs[0];oldword=int.from_bytes(a[i:i+4],'little');newword=int.from_bytes(b[i:i+4],'little')
    assert oldword==0x5287ffe0 and newword==0x528fffe0
    audit=dict(prior=str(OLD),prior_sha256=sha(OLD/'guest.export-safe'),new_sha256=sha(OUT/'guest.export-safe'),
        source_change='Only t>16384 -> t>32768',text_bytes=len(a),changed_instruction_offset=i,
        old_instruction=hex(oldword),new_instruction=hex(newword),all_other_text_identical=True,
        bounds=dict(max_array_elements=32*32768*64,max_causal_index=32767,configured_memory_bytes=2*1024**3),
        headers={p.name:sha(p)==sha(OLD/p.name) for p in OUT.glob('*.hpp')})
    assert all(audit['headers'].values());save(OUT/'audit.json',audit)
    save(OUT/'frozen.json',{str(p):sha(p) for p in OUT.iterdir() if p.is_file()})
    save(OUT/'status.json',dict(stage='built_requires_validation',sha256=sha(OUT/'guest.export-safe')))
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
