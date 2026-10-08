"""Extract first-dump ROI traffic and overlapping CPU diagnostics as JSON."""
import json
from pathlib import Path
import re
import sys

def analyze(path):
    text = (path / 'stats.txt').read_text().split('End Simulation Statistics')[0]
    s = {}
    for line in text.splitlines():
        w = line.split()
        if len(w) > 1:
            try:
                s[w[0]] = float(w[1])
            except ValueError:
                pass
    def sum_re(pattern):
        return sum(v for k, v in s.items() if re.search(pattern, k))
    def vector(name):
        match = re.search(r'^system.ruby.network\.' + name + r'\s+\|(.+)$', text, re.M)
        return [float(x.split()[0]) for x in match[1].split('|')] if match else []
    def weighted_load(unit):
        counts = [s.get(f'system.cpu{i}.{unit}.loadToUse::samples', 0) for i in range(8)]
        return sum(c * s.get(f'system.cpu{i}.{unit}.loadToUse::mean', 0)
                   for i, c in enumerate(counts)) / max(1, sum(counts))
    r = dict(path=str(path), roi_us=s['simTicks']/1e6,
             dram_read_bytes=sum_re(r'\.dram.bytesRead::total$'),
             dram_write_bytes=sum_re(r'\.dram.bytesWritten::total$'),
             l1d_misses=sum_re(r'l0_cntrl\d+.Dcache.m_demand_misses$'),
             private_l2_misses=sum_re(r'l1_cntrl\d+.cache.m_demand_misses$'),
             llc_misses=sum_re(r'l2_cntrl\d+.L2cache.m_demand_misses$'),
             raw_wait_cycles=sum_re(r'\.spmLsq0.spmRawWaitCycles$'),
             raw_blocks=sum_re(r'\.spmLsq0.spmRawBlockEvents$'),
             spm_port_rejects=sum_re(r'\.spmLsq0.spmPortBlockedEvents$'),
             spm_responses=sum_re(r'\.spmLsq0.spmLoadResponses$'),
             regular_load_mean_cycles=weighted_load('regularLsq0'),
             spm_load_mean_cycles=weighted_load('spmLsq0'),
             packets_by_vnet=vector('packets_injected'),
             flits_by_vnet=vector('flits_injected'),
             peak_link_utilization=max(vector('link_utilization'), default=0))
    for metric in ('packets_injected::total', 'flits_injected::total',
                   'average_packet_network_latency', 'average_packet_queueing_latency', 'average_hops'):
        r[metric] = s.get('system.ruby.network.' + metric, 0)
    r['spm_service_mean_cycles'] = sum_re(r'\.spmLsq0.spmLoadServiceCycles$') / max(1, r['spm_responses'])
    for suffix in ('ROBFullEvents', 'SQFullEvents', 'iqFullEvents', 'lsqFullEvents',
                   'spmLsqFullEvents', 'm_spmcpRequestsIssued', 'm_spmcpRequestsCompleted',
                   'm_spmcpBusyCycles', 'numCycles'):
        r[suffix] = sum_re(r'\.' + suffix + '$')
    return r

if __name__ == '__main__':
    print(json.dumps([analyze(Path(x)) for x in sys.argv[1:]], indent=2))
