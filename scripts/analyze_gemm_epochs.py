#!/usr/bin/env python3
"""Convert multi-section ACL SVE GEMM gem5 stats into one CSV row/epoch."""

import argparse
import csv
import math
import re
from pathlib import Path


WORKLOADS = {
    (128, 4096, 11008): "W1",
    (256, 4096, 4096): "W2",
    (256, 2048, 2048): "W3",
    (512, 768, 768): "W4",
    (2048, 2048, 2048): "W5",
    (2048, 64, 2048): "W6",
    (784, 256, 1024): "W7",
}

DIR_RE = re.compile(
    r"^mesi3_(?P<mode>spm_|spmsnap_|spmproto_)?(?P<cpu>o3|derivo3(?:paper(?:pf)?)?|neoversev2(?:_fdp[01])?)"
    r"(?P<dtype>_fp16)?"
    r"(?:_vl(?P<vl>\d+))?_garnet_(?P<cores>[1248])c_"
    r"m(?P<m>\d+)_n(?P<n>\d+)_k(?P<k>\d+)_i(?P<iterations>\d+)$"
)
CPU_RE = re.compile(r"system\.cpu([0-7]?)\.(.+)")
MEM_RE = re.compile(r"system\.mem_ctrls(\d*)\.(.+)")
ROUTER_RE = re.compile(r"system\.ruby\.network\.routers(\d+)\.(.+)")


def ratio(num, den):
    return num / den if den else 0.0


def parse_value(words):
    try:
        return float(words[1])
    except (IndexError, ValueError):
        return None


def blank_section():
    return {
        "scalar": {}, "cpu": {}, "mem": {}, "router": {},
        "dcache_hits": 0.0, "dcache_misses": 0.0,
        "l0_replacements": 0.0,
    }


def finish_section(
        data, workload, cpu_model, sve_vl_bits,
        m, k, n, iterations, section):
    scalar = data["scalar"]
    cpu = data["cpu"]
    mem = data["mem"]
    router = data["router"]
    lane_bits = 16 if "FP16" in cpu_model else 32
    tile_n = 3 * (sve_vl_bits // lane_bits)
    n_tiles = math.ceil(n / tile_n)
    kernel_sections = iterations * n_tiles
    is_kernel = section <= kernel_sections
    zero_panel = section - 1

    core_cycles = [v.get("numCycles", 0.0) for v in cpu.values()]
    core_ipc = [v.get("ipc", 0.0) for v in cpu.values()]
    committed = sum(v.get("commitStats0.numInsts", 0.0) for v in cpu.values())
    idle = sum(v.get("idleCycles", 0.0) for v in cpu.values())
    iq_full = sum(v.get("iew.iqFullEvents", 0.0) for v in cpu.values())
    lsq_full = sum(v.get("iew.lsqFullEvents", 0.0) for v in cpu.values())
    cache_blocked = sum(v.get("lsq0.blockedByCache", 0.0) for v in cpu.values())
    spm_lsq_full = sum(v.get("iew.spmLsqFullEvents", 0.0)
                       for v in cpu.values())
    spm_cache_blocked = sum(v.get("spm_lsq0.blockedByCache", 0.0)
                            for v in cpu.values())

    def sum_suffix(suffix):
        return sum(value for name, value in scalar.items()
                   if name.endswith(suffix))

    sim_ticks = scalar.get("simTicks", 0.0)
    cpu_hz = 2.5e9 if "PaperCore" in cpu_model else 1.5e9
    elapsed_cycles = sim_ticks / (1.0e12 / cpu_hz)
    max_cycles = max(core_cycles, default=0.0)
    min_cycles = min(core_cycles, default=0.0)

    d_hits = data["dcache_hits"]
    d_misses = data["dcache_misses"]
    d_accesses = d_hits + d_misses

    requests = sum(v.get("sa_request_events::total", 0.0) for v in router.values())
    grants = sum(v.get("sa_grant_events::total", 0.0) for v in router.values())
    ready = sum(v.get("sa_ready_events::total", 0.0) for v in router.values())
    no_credit = sum(v.get("sa_no_credit_events::total", 0.0) for v in router.values())
    no_vc = sum(v.get("sa_no_free_vc_events::total", 0.0) for v in router.values())

    read_reqs = sum(v.get("readReqs", 0.0) for v in mem.values())
    write_reqs = sum(v.get("writeReqs", 0.0) for v in mem.values())
    read_bytes = sum(v.get("bytesReadSys", 0.0) for v in mem.values())
    write_bytes = sum(v.get("bytesWrittenSys", 0.0) for v in mem.values())
    rd_retries = sum(v.get("numRdRetry", 0.0) for v in mem.values())
    wr_retries = sum(v.get("numWrRetry", 0.0) for v in mem.values())
    rd_queues = [v.get("avgRdQLen", 0.0) for v in mem.values()]
    bus_utils = [v.get("dram.busUtil", 0.0) for v in mem.values()]
    lat_weight = sum(v.get("readBursts", 0.0) + v.get("writeBursts", 0.0)
                     for v in mem.values())
    weighted_lat = sum(v.get("dram.avgMemAccLat", 0.0) *
                       (v.get("readBursts", 0.0) + v.get("writeBursts", 0.0))
                       for v in mem.values())

    flits = scalar.get("system.ruby.network.flits_injected::total", 0.0)
    link_flits = scalar.get("system.ruby.network.link_flits::total", 0.0)
    pkt_lat = scalar.get("system.ruby.network.average_packet_latency", 0.0)
    pkt_queue = scalar.get(
        "system.ruby.network.average_packet_queueing_latency", 0.0)

    return {
        "workload": workload, "cpu_model": cpu_model,
        "sve_vl_bits": sve_vl_bits, "tile_n": tile_n,
        "M": m, "K": k, "N": n,
        "iterations": iterations, "section": section,
        "phase": "kernel" if is_kernel else "teardown",
        "iteration": zero_panel // n_tiles if is_kernel else "",
        "panel": zero_panel % n_tiles if is_kernel else "",
        "n_begin": (zero_panel % n_tiles) * tile_n if is_kernel else "",
        "sim_ticks": sim_ticks, "elapsed_cycles": elapsed_cycles,
        "committed_insts": committed,
        "aggregate_ipc": ratio(committed, elapsed_cycles),
        "mean_core_ipc": ratio(sum(core_ipc), len(core_ipc)),
        "max_core_cycles": max_cycles, "min_core_cycles": min_cycles,
        "core_cycle_imbalance": ratio(max_cycles - min_cycles, max_cycles),
        "idle_cycles_total": idle,
        "dcache_hits": d_hits, "dcache_misses": d_misses,
        "dcache_accesses": d_accesses,
        "dcache_miss_rate": ratio(d_misses, d_accesses),
        "dcache_mpki": ratio(d_misses * 1000.0, committed),
        "l0_replacements": data["l0_replacements"],
        "iq_full_events": iq_full, "lsq_full_events": lsq_full,
        "cache_blocked_events": cache_blocked,
        "spm_lsq_full_events": spm_lsq_full,
        "spm_cache_blocked_events": spm_cache_blocked,
        "spmcp_issued": sum_suffix(".m_spmcpRequestsIssued"),
        "spmcp_completed": sum_suffix(".m_spmcpRequestsCompleted"),
        "spmcp_busy_core_cycles": sum_suffix(".m_spmcpBusyCycles"),
        "spmcp_outstanding_cycle_area": sum_suffix(
            ".m_spmcpOutstandingCycleArea"),
        "spmcp_buffer_full_rejects": sum_suffix(
            ".m_spmcpBufferFullRejects"),
        "spmcp_blocked_line_rejects": sum_suffix(
            ".m_spmcpBlockedLineRejects"),
        "spmcp_mean_latency_cycles": scalar.get(
            "system.ruby.RequestType.SPMCP_fetch.latency_hist_seqr::mean",
            0.0),
        "spm_load_requests": scalar.get(
            "system.ruby.L1Cache_Controller.SPMLD::total",
            scalar.get("system.ruby.L1Cache_Controller.SPMLD", 0.0)),
        "spm_load_mean_latency_cycles": scalar.get(
            "system.ruby.RequestType.SPMLD.latency_hist_seqr::mean", 0.0),
        "flits_injected": flits,
        "flits_received": scalar.get(
            "system.ruby.network.flits_received::total", 0.0),
        "packets_injected": scalar.get(
            "system.ruby.network.packets_injected::total", 0.0),
        "link_flits": link_flits,
        "flit_amplification": ratio(link_flits, flits),
        "average_hops": scalar.get("system.ruby.network.average_hops", 0.0),
        "packet_latency_ticks": pkt_lat,
        "packet_network_latency_ticks": scalar.get(
            "system.ruby.network.average_packet_network_latency", 0.0),
        "packet_queue_latency_ticks": pkt_queue,
        "packet_queue_fraction": ratio(pkt_queue, pkt_lat),
        "router_requests": requests, "router_grants": grants,
        "router_no_credit": no_credit, "router_no_free_vc": no_vc,
        "router_grant_efficiency": ratio(grants, requests),
        "router_credit_block_rate": ratio(no_credit, requests),
        "router_vc_block_rate": ratio(no_vc, ready),
        "dram_read_reqs": read_reqs, "dram_write_reqs": write_reqs,
        "dram_read_bytes": read_bytes, "dram_write_bytes": write_bytes,
        "dram_avg_read_queue": ratio(sum(rd_queues), len(rd_queues)),
        "dram_max_read_queue": max(rd_queues, default=0.0),
        "dram_avg_mem_latency_ticks": ratio(weighted_lat, lat_weight),
        "dram_mean_bus_util_pct": ratio(sum(bus_utils), len(bus_utils)),
        "dram_max_bus_util_pct": max(bus_utils, default=0.0),
        "dram_read_retries": rd_retries, "dram_write_retries": wr_retries,
    }


def parse_file(
        path, workload, cpu_model, sve_vl_bits, m, k, n, iterations):
    rows = []
    data = None
    section = 0
    with path.open("r", errors="replace") as stats:
        for line in stats:
            if line.startswith("---------- Begin Simulation Statistics"):
                section += 1
                data = blank_section()
                continue
            if data is None:
                continue
            if line.startswith("---------- End Simulation Statistics"):
                rows.append(finish_section(
                    data, workload, cpu_model, sve_vl_bits,
                    m, k, n, iterations, section))
                data = None
                continue
            words = line.split()
            value = parse_value(words)
            if value is None:
                continue
            name = words[0]
            data["scalar"][name] = value
            match = CPU_RE.fullmatch(name)
            if match:
                data["cpu"].setdefault(int(match.group(1) or 0), {})[
                    match.group(2)] = value
                continue
            match = MEM_RE.fullmatch(name)
            if match:
                data["mem"].setdefault(int(match.group(1) or 0), {})[
                    match.group(2)] = value
                continue
            match = ROUTER_RE.fullmatch(name)
            if match:
                data["router"].setdefault(int(match.group(1)), {})[
                    match.group(2)] = value
                continue
            if re.fullmatch(
                    r"system\.ruby\.l0_cntrl[0-7]\.Dcache\.m_demand_hits",
                    name):
                data["dcache_hits"] += value
            elif re.fullmatch(
                    r"system\.ruby\.l0_cntrl[0-7]\.Dcache\.m_demand_misses",
                    name):
                data["dcache_misses"] += value
            elif name == "system.ruby.L0Cache_Controller.S.L0_Replacement::total":
                data["l0_replacements"] = value
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path,
                        default=Path("results/acl_sve_gemm"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/acl_sve_gemm/per_epoch_bottlenecks.csv"))
    args = parser.parse_args()

    rows = []
    for directory in sorted(args.results.glob("mesi3_*_garnet_*c_m*_n*_k*_i*")):
        match = DIR_RE.fullmatch(directory.name)
        if not match or not (directory / "stats.txt").is_file():
            continue
        cpu_model = {
            "o3": "DerivO3CPU",
            "neoversev2": "NeoverseV2",
            "neoversev2_fdp0": "NeoverseV2-noFDP",
            "neoversev2_fdp1": "NeoverseV2-FDP",
            "derivo3": "DerivO3CPU",
            "derivo3paper": "DerivO3CPU-PaperCore",
            "derivo3paperpf": "DerivO3CPU-PaperCore-PaperCache-RubyPF",
        }[match.group("cpu")]
        if match.group("dtype"):
            cpu_model += "-FP16"
        if match.group("mode") == "spm_":
            cpu_model += "-CacheFlexSPM-2L1S"
        elif match.group("mode") == "spmsnap_":
            cpu_model += "-CacheFlexSPMSnap-2L1S"
        elif match.group("mode") == "spmproto_":
            cpu_model += "-SPMProtocol-NoSPM"
        cpu_model += f"-{match.group('cores')}C"
        # Pre-VL-tag result directories were all produced by the original
        # fixed-width 512-bit driver.
        sve_vl_bits = int(match.group("vl") or 512)
        m = int(match.group("m"))
        n = int(match.group("n"))
        k = int(match.group("k"))
        iterations = int(match.group("iterations"))
        workload = WORKLOADS.get((m, k, n))
        if workload is None:
            continue
        rows.extend(parse_file(
            directory / "stats.txt", workload, cpu_model,
            sve_vl_bits, m, k, n, iterations))

    rows.sort(key=lambda row: (
        row["cpu_model"], int(row["workload"][1:]), row["section"]))
    if not rows:
        raise SystemExit("no W1-W7 statistics found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
