"""Patch only unused legacy main and post-ROI export stub; preserve ELF layout."""
from pathlib import Path
import hashlib,json,struct,subprocess
HERE=Path(__file__).resolve().parent

def patch(guest):
    guest=Path(guest);out=guest.with_name('guest.export-safe')
    if out.exists():return out
    debug=guest.with_name('guest.debug')
    lines=subprocess.check_output(['aarch64-linux-gnu-nm','-S',str(debug)],text=True).splitlines()
    legacy=next(x.split() for x in lines if x.endswith(' _Z37cacheflex_micro_spm_flash_single_mainiPPc'))
    writer=next(x.split() for x in lines if x.endswith(' m5_write_file'))
    addr,size=int(legacy[0],16),int(legacy[1],16);entry=int(writer[0],16)
    obj=guest.with_name('export_hook.o');blob=guest.with_name('export_hook.bin')
    subprocess.run(['aarch64-linux-gnu-as',str(HERE/'fa_export_hook.S'),'-o',str(obj)],check=True)
    subprocess.run(['aarch64-linux-gnu-objcopy','-O','binary','--only-section=.text',str(obj),str(blob)],check=True)
    hook=blob.read_bytes();assert len(hook)<=size
    data=guest.read_bytes();assert data[:6]==b'\x7fELF\x02\x01'
    phoff=struct.unpack_from('<Q',data,32)[0];ents,n=struct.unpack_from('<HH',data,54)
    def file_offset(va):
        for i in range(n):
            typ,flags,off,vaddr,_,filesz,_,_=struct.unpack_from('<IIQQQQQQ',data,phoff+i*ents)
            if typ==1 and vaddr<=va<vaddr+filesz:
                assert flags&1
                return off+va-vaddr
        raise ValueError(hex(va))
    start=file_offset(addr);stub=file_offset(entry)
    assert data[stub:stub+8]==struct.pack('<II',0xff4f0110,0xd65f03c0)
    # Legacy renamed main must have no direct call/jump reference in original.
    dis=subprocess.check_output(['aarch64-linux-gnu-objdump','-d',str(guest)],text=True)
    assert not any(('bl\t'+format(addr,'x')+' ' in x or 'b\t'+format(addr,'x')+' ' in x) for x in dis.splitlines())
    delta=(addr-entry)//4;assert -(1<<25)<=delta<(1<<25)
    patched=bytearray(data);patched[start:start+len(hook)]=hook
    patched[stub:stub+4]=struct.pack('<I',0x14000000|(delta&0x3ffffff))
    assert patched[:start]==data[:start] if start<stub else True
    for i,(a,b) in enumerate(zip(data,patched)):
        if a!=b:assert start<=i<start+len(hook) or stub<=i<stub+4
    out.write_bytes(patched);out.chmod(0o755)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    out.with_suffix('.json').write_text(json.dumps(dict(original=sha(data),patched=sha(patched),
      original_path=str(guest),hook_va=hex(addr),writer_va=hex(entry),
      changed_file_ranges=[[start,start+len(hook)],[stub,stub+4]],
      invariant='All other ELF bytes including worker/ROI code, entry points, segment layout unchanged'),indent=2))
    return out

def verify_dump(row):
    import numpy as np
    p=Path(row['path']);h=1469598103934665603
    for word in np.fromfile(p/'output.fp16',dtype=np.uint16):h=((h^int(word))*1099511628211)&((1<<64)-1)
    assert f'{h:016x}'==row['output']['hash'],('guest/dump mismatch',p,f'{h:016x}',row['output']['hash'])
    return row
