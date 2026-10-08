import json,sys,subprocess,shutil,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent
ROOT=R/'results/cacheflex_flash_attention/ncc_probe_20261005_v5';G=ROOT/'guest_v1';B=ROOT/'build_v1';OLD=R/'results/cacheflex_flash_attention/ncc_probe_20261005_v4/guest_v1'
G.mkdir(parents=True);B.mkdir()
for p in OLD.glob('*.hpp'):shutil.copy2(p,G/p.name)
s=(OLD/'driver.cpp').read_text();start=s.index('static void touch_read(');end=s.index('static void* legacy_worker',start)
s=s[:start]+(H/'transport.hpp').read_text()+'''
static void ncc_pack(int op,const void* src,void* dst,size_t bytes) {
 ncc_input(src,bytes);ncc_sum(0x4e434350,op,0,0,0,ncc_mode);ncc_output(dst,bytes);
}
'''+s[end:]
marker='                        ncc_sum(0x4e434350,3,reinterpret_cast<size_t>(pa.p),reinterpret_cast<size_t>(pv.p),0,ncc_mode);'
assert marker in s;s=s.replace(marker,'                        ncc_input(pa.p,8192);ncc_sum(0x4e434350,3,0,0,0,ncc_mode);ncc_output(pv.p,2048);')
(G/'driver.cpp').write_text(s);(G/'build.sh').write_text((OLD/'build.sh').read_text().replace(str(OLD),str(G)))
subprocess.run(['bash',str(G/'build.sh')],cwd=R,check=True)
sys.path.insert(0,str(R/'benchmarks/cacheflex_flash_attention'));from export_safe_elf import patch
patch(G/'guest',G/'guest.export-safe')
(G/'frozen.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in G.iterdir() if p.is_file()},indent=2))
old=R/'results/cacheflex_flash_attention/ncc_probe_20261005/build_v1/pseudo_inst.cc'
(B/'pseudo_inst.cc').write_text(old.read_text());shutil.copy2(H/'hook.hh',B/'hook.hh')
print('Guest and hook prepared')
