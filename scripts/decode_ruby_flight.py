#!/usr/bin/env python3
"""Decode bounded Ruby flight recordings, including after abort/SIGKILL."""
import argparse
import mmap
from pathlib import Path
import struct
import sys


def decode(path):
    records, rejected = [], 0
    with open(path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if len(data) < 64 or data[:8] != b'CFRUBY01':
            raise ValueError(f'{path}: invalid header')
        endian = '<' if data[24:32] == bytes(range(8, 0, -1)) else '>'
        capacity, size, marker = struct.unpack_from(endian + 'QQQ', data, 8)
        if marker != 0x0102030405060708 or size != 1024 or len(data) != 64 + capacity * size:
            raise ValueError(f'{path}: invalid format/size')
        for index in range(capacity):
            offset = 64 + index * size
            slot = data[offset:offset + size]
            seq, tick, length, checksum = struct.unpack_from(endian + 'QQII', slot)
            end, = struct.unpack_from(endian + 'Q', slot, 1016)
            now, = struct.unpack_from(endian + 'Q', data, offset)
            if seq == end == now == 0:
                continue
            if not seq or seq != end or seq != now or length > 992 or (seq - 1) % capacity != index:
                rejected += 1
                continue
            value = 2166136261
            for byte in slot[8:20] + slot[24:24 + length]:
                value = ((value ^ byte) * 16777619) & 0xffffffff
            if value != checksum:
                rejected += 1
                continue
            records.append((tick, Path(path).name.removesuffix('.flight.bin'), seq,
                            slot[24:24 + length].decode('utf-8', errors='replace').rstrip('\0')))
    return sorted(records), rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path, help='Recording files or output directories')
    parser.add_argument('--contains', help='Only show records containing this text')
    args = parser.parse_args()
    paths = [f for p in args.paths for f in (sorted(p.glob('*.flight.bin')) if p.is_dir() else [p])]
    if not paths:
        parser.error('No .flight.bin recordings found')
    records = []
    for path in paths:
        rows, rejected = decode(path)
        print(f'{path}: {len(rows)} retained records; {rejected} incomplete/corrupt slots', file=sys.stderr)
        records.extend(rows)
    for tick, controller, seq, message in sorted(records):
        if not args.contains or args.contains in message:
            print(f'{tick}: {controller} #{seq}: {message}')


if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError:
        # Allow piping a large decoded history to head without a traceback.
        sys.stdout = open('/dev/null', 'w')
