#!/usr/bin/env python3
"""Join Ruby NP/GETX transitions with Sequencer vaddr/PC mappings."""

import collections
import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 6:
        print("usage: analyze_flash_getx_regions.py RUBY_TRACE MAP_TRACE RUN_LOG T BR", file=sys.stderr)
        return 2
    ruby_path, map_path, log_path = map(Path, sys.argv[1:4])
    T, Br = map(int, sys.argv[4:6])
    Bc, d, cores = 96, 64, 8
    log = log_path.read_text()
    roi = [int(x) * 1000 for x in re.findall(r"FLASH_ROI (?:begin|end) rpns=(\d+)", log)]
    if len(roi) != 2:
        raise RuntimeError("missing ROI timestamps")

    ranges = []
    shared = re.search(r"FLASH_SHARED Q=(\S+) K=(\S+) V=(\S+) O=(\S+) KT=(\S+)", log)
    if not shared:
        raise RuntimeError("missing shared buffer addresses")
    shared_sizes = [cores*T*d*2, T*d*2, T*d*2, cores*T*d*2, T*d*2]
    for name, text, size in zip(("Q", "K", "V", "O", "KT"), shared.groups(), shared_sizes):
        base = int(text, 16); ranges.append((base, base + size, name))

    n_ib = (T + Br - 1) // Br
    abmax = (Br + 7) // 8
    local_sizes = {
        "qap": n_ib*abmax*8*d*2, "S": Br*Bc*2, "P": Br*Bc*2,
        "PV": 8*d*2, "pap": 8*Bc*2, "Oacc": T*d*4,
        "m": T*4, "l": T*4,
        "qtile": Br*d*2, "spanel": abmax*8*Bc*2,
        "ptile": abmax*8*Bc*2, "pvpanel": 8*d*2,
        "qkap": abmax*8*d*2, "pvap": 8*Bc*2,
        "oacc": Br*d*4, "mvec": Br*4, "lvec": Br*4,
    }
    addr_re = re.compile(r"FLASH_ADDR tid=(\d+) (.*)")
    for match in addr_re.finditer(log):
        tid = int(match.group(1))
        for name, text in re.findall(r"(\w+)=(0x[0-9a-f]+)", match.group(2)):
            if name in local_sizes:
                base = int(text, 16)
                ranges.append((base, base + local_sizes[name], f"{name}"))

    mapping = {}
    map_re = re.compile(r"CacheFlexStoreMap paddr=(0x[0-9a-f]+) vaddr=(0x[0-9a-f]+) pc=(0x[0-9a-f]+)")
    with map_path.open() as stream:
        for line in stream:
            match = map_re.search(line)
            if match:
                paddr, vaddr, pc = (int(x, 16) for x in match.groups())
                mapping[paddr & ~63] = (vaddr & ~63, pc)

    region_counts = collections.Counter()
    pc_counts = collections.Counter()
    unmatched = 0
    event_re = re.compile(r"^\s*(\d+):.*Addr: (0x[0-9a-f]+) State: NP .*Type: GETX")
    with ruby_path.open() as stream:
        for line in stream:
            match = event_re.search(line)
            if not match:
                continue
            tick, paddr = int(match.group(1)), int(match.group(2), 16)
            if not roi[0] <= tick <= roi[1]:
                continue
            item = mapping.get(paddr & ~63)
            if item is None:
                unmatched += 1
                continue
            vaddr, pc = item
            region = "other"
            for begin, end, name in ranges:
                if begin <= vaddr < end:
                    region = name
                    break
            region_counts[region] += 1
            pc_counts[(pc, region)] += 1

    total = sum(region_counts.values()) + unmatched
    print("region,count,percent")
    for region, count in region_counts.most_common():
        print(f"{region},{count},{100*count/total if total else 0:.4f}")
    print(f"unmatched,{unmatched},{100*unmatched/total if total else 0:.4f}")
    print(f"total,{total},100.0000")
    print("\npc,region,count")
    for (pc, region), count in pc_counts.most_common(20):
        print(f"0x{pc:x},{region},{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
