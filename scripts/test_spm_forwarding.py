"""Host geometry test and reproducible guest build for finite-SQ forwarding."""
import subprocess
import os
from pathlib import Path
from build_access_timing import ROOT
OUT=ROOT/'results/cacheflex_flash_attention/spm_forwarding_20260924'/os.environ.get('CACHEFLEX_FORWARD_TESTS','tests')
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    src=OUT/'ranges.cc'
    src.write_text('''#include "cpu/o3/spm_forwarding.hh"
#include <cassert>
using gem5::o3::SpmForwardRange;
int main(){
 auto r=SpmForwardRange::decode(3<<6,128);
 assert(r.covers(SpmForwardRange::decode((1<<16)|(3<<6),64)));
 assert(!r.overlaps(SpmForwardRange::decode(4<<6,64)));
 assert(!r.covers(SpmForwardRange::decode((1<<16)|(3<<6)|32,64)));
 assert(r.overlaps(SpmForwardRange::decode((1<<16)|(3<<6)|32,64)));
 assert(!SpmForwardRange::decode((7<<16)|(3<<6),128).valid());
}''')
    subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'gem5/src'),str(src),'-o',str(OUT/'ranges')],check=True)
    subprocess.run([str(OUT/'ranges')],check=True)
    old=ROOT/'results/cacheflex_flash_attention/output_resident_20260924/roundtrip'
    script=(old/'build.sh').read_text().replace(str(old),str(OUT))
    script=script.replace('tests/test_output_resident.cpp','tests/test_spm_forwarding.cpp')
    (OUT/'build.sh').write_text(script)
    subprocess.run(['bash',str(OUT/'build.sh')],cwd=ROOT,check=True)
if __name__=='__main__':main()
