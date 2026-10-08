#!/usr/bin/env python3
"""Print the additive CacheFlex microkernel cycle breakdown from gem5 stats."""

import argparse
import re
from pathlib import Path


FIELDS = {
    "kernelComputeCycles": "compute",
    "kernelSpmDataCycles": "spm_b_data",
    "kernelCacheDataCycles": "cache_a_data",
    "kernelCacheBDataCycles": "cache_b_data",
    "kernelStoreCycles": "c_store_writeback",
    "kernelFrontendControlCycles": "frontend_control",
    "kernelOtherBackendCycles": "other_backend",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stats", type=Path)
    args = parser.parse_args()

    totals = {label: 0 for label in FIELDS.values()}
    marked = 0
    classified_formula = 0
    pattern = re.compile(
        r"^system\.cpu\d+\.commit\."
        r"(kernel(?:Profile|Compute|SpmData|CacheData|CacheBData|Store|"
        r"FrontendControl|OtherBackend|Classified)Cycles)\s+(\d+)"
    )

    # The runner dumps ROI statistics first and final/post-ROI statistics
    # second. Only consume the first block.
    for line in args.stats.read_text().splitlines():
        if "End Simulation Statistics" in line:
            break
        match = pattern.match(line)
        if not match:
            continue
        field, value_text = match.groups()
        value = int(value_text)
        if field == "kernelProfileCycles":
            marked += value
        elif field == "kernelClassifiedCycles":
            classified_formula += value
        else:
            totals[FIELDS[field]] += value

    classified = sum(totals.values())
    if not marked:
        raise SystemExit("no marked kernel cycles found")
    if classified != marked or classified_formula != marked:
        raise SystemExit(
            f"non-additive result: marked={marked} bins={classified} "
            f"formula={classified_formula}"
        )

    for label, cycles in totals.items():
        print(f"{label},{cycles},{100.0 * cycles / marked:.4f}")
    print(f"total,{marked},100.0000")


if __name__ == "__main__":
    main()
