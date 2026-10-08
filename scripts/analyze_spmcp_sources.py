#!/usr/bin/env python3
"""Report an additive source-path breakdown for single-target SPMCP fetches."""

import re
import sys
from pathlib import Path


def first_section(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    active = False
    result = []
    for line in lines:
        if "Begin Simulation Statistics" in line:
            if active:
                break
            active = True
            continue
        if active and "End Simulation Statistics" in line:
            break
        if active:
            result.append(line)
    return result


def total(lines: list[str], pattern: str) -> int:
    rx = re.compile(pattern)
    value = 0
    for line in lines:
        match = rx.match(line)
        if match:
            value += int(float(match.group(1)))
    return value


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} RESULT_DIR_OR_STATS", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if path.is_dir():
        path /= "stats.txt"
    lines = first_section(path)

    issued = total(lines, r".*sequencer\.m_spmcpDataRequestsIssued\s+(\d+)")
    l1_resident = total(
        lines,
        r"system\.ruby\.L1Cache_Controller\.(?:S|SS|E|EE|M|MM)\.SPMCP_fetch::total\s+(\d+)",
    )
    l1_absent = total(
        lines, r"system\.ruby\.L1Cache_Controller\.I\.SPMCP_fetch::total\s+(\d+)"
    )
    llc_resident = total(
        lines,
        r"system\.ruby\.L2Cache_Controller\.(?:SS|M)\.SPM_GETS_SILENT::total\s+(\d+)",
    )
    owner = total(
        lines, r"system\.ruby\.L2Cache_Controller\.MT\.SPM_GETS_SILENT::total\s+(\d+)"
    )
    memory = total(
        lines, r"system\.ruby\.L2Cache_Controller\.NP\.SPM_GETS_SILENT::total\s+(\d+)"
    )

    downstream = llc_resident + owner + memory
    reprocessed = max(0, downstream - l1_absent)
    categories = [
        ("requester_private_L2", l1_resident),
        ("shared_LLC", llc_resident),
        ("forwarded_owner", owner),
        ("memory", memory),
    ]
    classified = l1_resident + l1_absent
    print("source,count,percent")
    for name, value in categories:
        pct = 100.0 * value / issued if issued else 0.0
        print(f"{name},{value},{pct:.4f}")
    print(f"issued,{issued},100.0000")
    print(f"l1_absent_forwarded,{l1_absent},{100.0*l1_absent/issued if issued else 0:.4f}")
    print(f"downstream_reprocessed,{reprocessed},0.0000")
    print(f"classified,{classified},{100.0*classified/issued if issued else 0:.4f}")
    if classified != issued or downstream < l1_absent:
        print("ERROR: source categories do not add to issued SPMCPs", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
