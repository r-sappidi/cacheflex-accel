#!/usr/bin/env python3
"""Aggregate W1-W7 GEMM ROI compute counters into one CSV row/workload."""

import csv
import math
import re
from pathlib import Path


ROOT = Path("results/acl_sve_gemm")
OUTPUT = ROOT / "compute_utilization_actual.csv"
DIR_RE = re.compile(
    r".*_vl(?P<vl>\d+)_garnet_8c_m(?P<m>\d+)_n(?P<n>\d+)_k(?P<k>\d+)_i(?P<i>\d+)_w(?P<w>[1-7])$"
)
STAT_RE = re.compile(r"system\.cpu(?P<cpu>[0-7])\.(?P<name>\S+)")


def value(line):
    fields = line.split()
    try:
        return fields[0], float(fields[1])
    except (IndexError, ValueError):
        return None, None


def analyze(directory, match):
    m, n, k = (int(match.group(x)) for x in ("m", "n", "k"))
    vl = int(match.group("vl"))
    cores = 8
    lanes = vl // 32
    fma_units_per_core = 2
    peak_flops_per_core_cycle = fma_units_per_core * 2 * lanes
    tile_n = 3 * lanes
    kernel_sections = int(match.group("i")) * math.ceil(n / tile_n)

    totals = {
        "elapsed_cycles": 0.0,
        "committed_insts": 0.0,
        "executed_insts": 0.0,
        "sve_fma_committed": 0.0,
        "sve_fma_fu_busy_events": 0.0,
        "all_fu_busy_events": 0.0,
    }
    section = 0
    active = False
    sim_ticks = 0.0

    with (directory / "stats.txt").open(errors="replace") as stats:
        for line in stats:
            if line.startswith("---------- Begin Simulation Statistics"):
                section += 1
                active = section <= kernel_sections
                sim_ticks = 0.0
                continue
            if not active:
                continue
            name, number = value(line)
            if name is None:
                continue
            if name == "simTicks":
                sim_ticks = number
                continue
            cpu_stat = STAT_RE.fullmatch(name)
            if not cpu_stat:
                continue
            stat = cpu_stat.group("name")
            if stat == "commitStats0.numInsts":
                totals["committed_insts"] += number
            elif stat == "executeStats0.numInsts":
                totals["executed_insts"] += number
            elif stat == "commit.committedInstType_0::SimdFloatMultAcc":
                totals["sve_fma_committed"] += number
            elif stat == "statFuBusy::SimdFloatMultAcc":
                totals["sve_fma_fu_busy_events"] += number
            elif stat == "fuBusy":
                totals["all_fu_busy_events"] += number
            elif stat == "numCycles":
                # All cores share the epoch duration; account CPU0 once.
                if cpu_stat.group("cpu") == "0":
                    totals["elapsed_cycles"] += number

    cycles = totals["elapsed_cycles"]
    fmas = totals["sve_fma_committed"]
    useful_flops = 2 * m * n * k
    executed_fma_flops = fmas * 2 * lanes
    peak_system_flops_cycle = cores * peak_flops_per_core_cycle

    def ratio(a, b):
        return a / b if b else 0.0

    return {
        "workload": f"W{match.group('w')}",
        "M": m, "K": k, "N": n,
        "sve_vl_bits": vl,
        "roi_cycles": round(cycles),
        "committed_insts": round(totals["committed_insts"]),
        "aggregate_ipc": ratio(totals["committed_insts"], cycles),
        "mean_ipc_per_core": ratio(totals["committed_insts"], cycles * cores),
        "sve_fma_committed": round(fmas),
        "sve_fma_per_cycle_system": ratio(fmas, cycles),
        "sve_fma_per_cycle_per_core": ratio(fmas, cycles * cores),
        "sve_fma_issue_util_pct": 100 * ratio(fmas, cycles * cores * fma_units_per_core),
        "executed_fma_flops_per_cycle": ratio(executed_fma_flops, cycles),
        "useful_flops_per_cycle": ratio(useful_flops, cycles),
        "useful_compute_util_pct": 100 * ratio(useful_flops, cycles * peak_system_flops_cycle),
        "sve_fma_fu_busy_events": round(totals["sve_fma_fu_busy_events"]),
        "sve_fma_fu_busy_events_per_ki": 1000 * ratio(
            totals["sve_fma_fu_busy_events"], totals["committed_insts"]),
        "all_fu_busy_events": round(totals["all_fu_busy_events"]),
    }


def main():
    rows = []
    for directory in ROOT.iterdir():
        match = DIR_RE.fullmatch(directory.name)
        if match and (directory / "stats.txt").is_file():
            rows.append(analyze(directory, match))
    rows.sort(key=lambda row: int(row["workload"][1:]))
    if len(rows) != 7:
        raise SystemExit(f"expected 7 workloads, found {len(rows)}")
    with OUTPUT.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
