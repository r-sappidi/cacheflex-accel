"""Install the existing timing-read exporter without relocating kernel code.

Only for static little-endian AArch64 ELFs with unused file padding after an
executable PT_LOAD. Extend that segment within its existing final 4KiB page;
never overwrite a section or move any original code/data/entry point.
"""
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def patch(src, dst):
    src, dst = Path(src), Path(dst)
    assert not dst.exists(), dst
    data = src.read_bytes()
    assert data[:6] == b'\x7fELF\x02\x01'
    assert struct.unpack_from('<H', data, 18)[0] == 183  # AArch64
    symbols = subprocess.check_output(['aarch64-linux-gnu-nm', str(src)], text=True)
    writer = int(re.search(r'^([0-9a-f]+) T m5_write_file$', symbols, re.M)[1], 16)
    dst.parent.mkdir(parents=True, exist_ok=True)
    obj, blob = dst.with_suffix('.hook.o'), dst.with_suffix('.hook.bin')
    subprocess.run(['aarch64-linux-gnu-as', str(HERE/'fa_export_refresh.S'), '-o', str(obj)], check=True)
    subprocess.run(['aarch64-linux-gnu-objcopy', '-O', 'binary', '--only-section=.text', str(obj), str(blob)], check=True)
    hook = blob.read_bytes()
    phoff = struct.unpack_from('<Q', data, 32)[0]
    ents, count = struct.unpack_from('<HH', data, 54)
    loads = []
    for i in range(count):
        pos = phoff+i*ents
        typ, flags, off, va, pa, filesz, memsz, align = struct.unpack_from('<IIQQQQQQ', data, pos)
        if typ == 1:
            loads.append((pos, flags, off, va, filesz, memsz))
    seg = next(x for x in loads if x[1]&1 and x[3] <= writer < x[3]+x[4])
    pos, flags, off, va, filesz, memsz = seg
    assert filesz == memsz
    start = (off+filesz+3)&~3
    end = start+len(hook)
    hook_va = va+start-off
    newsize = end-off
    assert (memsz+4095)//4096 == (newsize+4095)//4096, 'Would change mapped page count'
    assert end <= len(data) and not any(data[off+filesz:end]), 'Not zero padding'
    for other in loads:
        if other == seg:
            continue
        assert end <= other[2] or start >= other[2]+other[4]
        assert va+newsize <= other[3] or va >= other[3]+other[5]
    # Reject overlap with any file-backed section, including non-ALLOC sections.
    shoff = struct.unpack_from('<Q', data, 40)[0]
    shents, shnum = struct.unpack_from('<HH', data, 58)
    for i in range(shnum):
        _, typ, _, _, so, size = struct.unpack_from('<IIQQQQ', data, shoff+i*shents)
        if typ != 8 and size:  # SHT_NOBITS has no file payload
            assert end <= so or start >= so+size
    stub = off+writer-va
    assert data[stub:stub+8] == struct.pack('<II', 0xff4f0110, 0xd65f03c0)
    delta = (hook_va-writer)//4
    assert -(1<<25) <= delta < (1<<25)
    patched = bytearray(data)
    patched[start:end] = hook
    patched[stub:stub+4] = struct.pack('<I', 0x14000000 | (delta&0x3ffffff))
    struct.pack_into('<QQ', patched, pos+32, newsize, newsize)
    ranges = [(start, end), (stub, stub+4), (pos+32, pos+48)]
    assert len(data) == len(patched)
    assert all(a == b or any(lo <= i < hi for lo, hi in ranges)
               for i, (a, b) in enumerate(zip(data, patched)))
    dst.write_bytes(patched)
    dst.chmod(0o755)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    report = dict(original=str(src), original_sha256=sha(data), patched_sha256=sha(patched),
                  changed_ranges=ranges, hook_va=hex(hook_va), writer_va=hex(writer),
                  invariant='All original kernel code/data addresses and bytes unchanged; same mapped pages; only post-ROI writer branch, padding and PT_LOAD sizes changed')
    dst.with_suffix('.audit.json').write_text(json.dumps(report, indent=2))
    return dst


def export_hash(path, log):
    """Fail on transport corruption BEFORE interpreting it as numerical error."""
    import numpy as np
    h = 1469598103934665603
    for x in np.fromfile(path, dtype=np.uint16):
        h = ((h ^ int(x))*1099511628211)&((1<<64)-1)
    expected = re.search(r'(?:KV|LLAMA)_RESULT hash=([0-9a-f]+) nonfinite=0', log)
    assert expected and f'{h:016x}' == expected[1], ('guest/export hash mismatch', str(path), f'{h:016x}', expected[1] if expected else None)
    return f'{h:016x}'
