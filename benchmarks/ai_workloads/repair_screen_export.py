"""Patch unused m5 stubs and writer only; no ROI/layout/simulator changes."""
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

HERE = Path(__file__).resolve().parent


def patch(guest):
    guest = Path(guest)
    original = guest.read_bytes()
    symbols = {}
    for line in subprocess.check_output(['aarch64-linux-gnu-nm', str(guest)], text=True).splitlines():
        fields = line.split()
        if len(fields) == 3:
            symbols[fields[2]] = int(fields[0], 16)
    start, end, writer = (symbols[n] for n in ('m5_arm', 'm5_sum', 'm5_write_file'))
    dis = subprocess.check_output(['aarch64-linux-gnu-objdump', '-d', str(guest)], text=True)
    # Reject direct branches/ADR references. ADRP names an entire 4KiB page,
    # not the symbol objdump happens to associate with that page boundary.
    for line in dis.splitlines():
        if not re.match(r'^\s*[0-9a-f]+:', line):
            continue
        if '\tadrp\t' in line:
            continue
        for target in re.findall(r'\b([0-9a-f]+) <[^>]+>', line):
            assert not start <= int(target, 16) < end, line
    obj, blob = guest.with_suffix('.filename.o'), guest.with_suffix('.filename.bin')
    subprocess.run(['aarch64-linux-gnu-as', str(HERE/'export_filename_hook.S'), '-o', str(obj)], check=True)
    subprocess.run(['aarch64-linux-gnu-objcopy', '-O', 'binary', '--only-section=.text', str(obj), str(blob)], check=True)
    hook = blob.read_bytes()
    assert len(hook) <= end-start
    assert original[:6] == b'\x7fELF\x02\x01'
    phoff = struct.unpack_from('<Q', original, 32)[0]
    ents, count = struct.unpack_from('<HH', original, 54)
    def offset(va, length):
        for i in range(count):
            typ, flags, off, addr, _, size, _, _ = struct.unpack_from('<IIQQQQQQ', original, phoff+i*ents)
            if typ == 1 and addr <= va and va+length <= addr+size:
                assert flags & 1
                return off+va-addr
        raise ValueError(hex(va))
    a, b = offset(start, len(hook)), offset(writer, 8)
    assert original[b:b+8] == struct.pack('<II', 0xff4f0110, 0xd65f03c0)
    delta = (start-writer)//4
    assert -(1<<25) <= delta < (1<<25)
    data = bytearray(original)
    data[a:a+len(hook)] = hook
    data[b:b+4] = struct.pack('<I', 0x14000000 | (delta & 0x3ffffff))
    assert len(data) == len(original)
    assert all(x == y or a <= i < a+len(hook) or b <= i < b+4
               for i, (x, y) in enumerate(zip(original, data)))
    backup = guest.with_suffix('.original')
    assert not backup.exists()
    backup.write_bytes(original)
    guest.write_bytes(data)
    report = dict(original_sha256=hashlib.sha256(original).hexdigest(),
                  patched_sha256=hashlib.sha256(data).hexdigest(),
                  ranges=[[a, a+len(hook)], [b, b+4]],
                  invariant='All other ELF bytes and segment layout unchanged; hook called only by post-ROI writer.')
    guest.with_suffix('.repair.json').write_text(json.dumps(report, indent=2))
    return report
