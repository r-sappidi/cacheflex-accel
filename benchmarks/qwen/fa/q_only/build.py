"""FA-52 guest: FA-51 guest v2 sources + Q-only head layout (mode 12). Isolated build.

Mode 12 = qo_worker with Q resident in SPM, O and V coherent (FA-35 modes all paired Q with O
or V). Modes 2 (nothing staged; cache control) and 3 (V only) are the matched controls in the
same ELF. Head layouts require sm_flags==0 (FA-50 flags patch only the legacy worker).
"""
import hashlib, json, os, resource, shutil, subprocess, sys
from pathlib import Path
R = Path('/data2/rsappidi/cacheflex_mc'); H = Path(__file__).resolve().parent
SRC = R / 'results/cacheflex_flash_attention/softmax_tune_20261005/guest_v1'
G = R / 'results/cacheflex_flash_attention/q_only_20261006/guest_v1'
sys.path.insert(0, str(R / 'benchmarks/cacheflex_flash_attention'))
from export_safe_elf import patch
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def one(s, a, b):
    assert s.count(a) == 1, (a[:70], s.count(a)); return s.replace(a, b)
G.mkdir(parents=True, exist_ok=False)
for p in SRC.glob('*.hpp'): shutil.copy2(p, G / p.name)
s = (SRC / 'driver.cpp').read_text()
s = one(s, 'const bool qresident=layout_mode==4||layout_mode==5||layout_mode==6,',
           'const bool qresident=layout_mode==4||layout_mode==5||layout_mode==6||layout_mode==12,')
s = one(s, '    if(layout_mode>1||sets!=1024||((sm_flags&4)&&br!=16)||(sm_flags&1)&&br%4)return 2;\n',
           '    if(!(layout_mode<=3||layout_mode==12)||sets!=1024||((sm_flags&4)&&br!=16)||(sm_flags&1)&&br%4)return 2;\n'
           '    if(layout_mode>=2&&sm_flags)return 2;\n')
s = one(s, 'layout_mode<0||layout_mode>11||', 'layout_mode<0||(layout_mode>11&&layout_mode!=12)||')
s = one(s, 'if(layout_mode>=2&&layout_mode<=7&&(t%256', 'if(((layout_mode>=2&&layout_mode<=7)||layout_mode==12)&&(t%256')
s = one(s, 'auto worker=layout_mode>=9?kv_worker:((layout_mode>=2&&layout_mode<=7)?qo_worker:legacy_worker);',
           'auto worker=(layout_mode>=9&&layout_mode<=11)?kv_worker:(((layout_mode>=2&&layout_mode<=7)||layout_mode==12)?qo_worker:legacy_worker);')
(G / 'driver.cpp').write_text(s)
(G / 'build.sh').write_text((SRC / 'build.sh').read_text().replace(str(SRC), str(G)))
def limits():
    resource.setrlimit(resource.RLIMIT_AS, (3 * 2**30, 3 * 2**30)); os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
subprocess.run(['bash', str(G / 'build.sh')], cwd=R, check=True, preexec_fn=limits)
patch(G / 'guest', G / 'guest.export-safe')
(G / 'frozen.json').write_text(json.dumps({str(p): sha(p) for p in [*G.iterdir(), SRC / 'driver.cpp', H / 'build.py'] if p.is_file()}, indent=2) + '\n')
print('built', G / 'guest.export-safe')
