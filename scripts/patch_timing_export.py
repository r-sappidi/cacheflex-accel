"""Reuse the established filename-refresh repair without altering FA ROI bytes."""
from pathlib import Path
import struct
import subprocess
from build_access_timing import ROOT,sha,save

def patch(src,dst):
    src=Path(src);dst=Path(dst)
    symbols={w[-1]:(int(w[0],16),int(w[1],16)) for line in
        subprocess.check_output(['aarch64-linux-gnu-nm','-S',str(src)],text=True).splitlines()
        if len(w:=line.split())==4}
    start,size=symbols['_Z37cacheflex_micro_spm_flash_single_mainiPPc']
    data=src.read_bytes()
    phoff=struct.unpack_from('<Q',data,32)[0];ent,n=struct.unpack_from('<HH',data,54)
    def offset(va,length):
        for i in range(n):
            typ,flags,off,addr,_,length_file,_,_=struct.unpack_from('<IIQQQQQQ',data,phoff+i*ent)
            if typ==1 and addr<=va and va+length<=addr+length_file:
                assert flags&1
                return off+va-addr
        raise ValueError(hex(va))
    begin=offset(start,176)
    op=struct.pack('<I',0xff4f0110)
    assert [i for i in range(0,176,4) if data[begin+i:begin+i+4]==op]==[108,168]
    # This unused old-main region is already the post-ROI exporter. The only
    # new instructions occupy its unused tail; the timed worker is untouched.
    obj=dst.with_suffix('.o');blob=dst.with_suffix('.bin')
    helper=ROOT/'benchmarks/cacheflex_flash_attention/export_filename_each_chunk.S'
    subprocess.run(['aarch64-linux-gnu-as',str(helper),'-o',str(obj)],check=True)
    subprocess.run(['aarch64-linux-gnu-objcopy','-O','binary','--only-section=.text',str(obj),str(blob)],check=True)
    code=blob.read_bytes();assert 176+len(code)<=size
    target=start+176;call=start+108
    out=bytearray(data);out[begin+176:begin+176+len(code)]=code
    out[begin+108:begin+112]=struct.pack('<I',0x94000000|(((target-call)//4)&0x3ffffff))
    ranges=[(begin+108,begin+112),(begin+176,begin+176+len(code))]
    assert len(out)==len(data)
    assert all(a==b or any(lo<=i<hi for lo,hi in ranges) for i,(a,b) in enumerate(zip(data,out)))
    dst.write_bytes(out);dst.chmod(0o755)
    save(dst.with_suffix('.audit.json'),dict(original=str(src),original_sha256=sha(src),
        patched_sha256=sha(dst),helper_sha256=sha(helper),changed_file_ranges=ranges,
        invariant='Only post-ROI export call and unused old-main tail changed; all other ELF bytes/layout identical'))

if __name__=='__main__':
    base=ROOT/'results/cacheflex_flash_attention'
    out=base/'timing_alignment_20260922/fa_export_guests';out.mkdir()
    for arch in ('cache','spm'):
        patch(base/f'micro_resources_20260921/experiment_v6/preserved/{arch}',out/arch)
