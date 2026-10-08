"""Validate paired scalar FlashAttention runs and print ROI CSV to stdout."""
import configparser
import csv
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])

def read_run(path):
    log = (path / 'run.log').read_text()
    assert 'FLASH_ROI begin' in log and 'FLASH_ROI end' in log, path
    assert 'exiting with last active thread context' in log, path
    assert not re.search(r'panic:|fatal:', log), path
    result = dict(re.findall(r'(\w+)=([^\s]+)', log.split('FLASH_RESULT ')[-1].splitlines()[0]))
    assert result['nonfinite'] == '0', path
    stats = {}
    for line in (path / 'stats.txt').read_text().split('End Simulation Statistics')[0].splitlines():
        fields = line.split()
        if len(fields) >= 2:
            try:
                stats[fields[0]] = float(fields[1])
            except ValueError:
                pass
    config = configparser.ConfigParser(interpolation=None, strict=False)
    config.read(path / 'config.ini')
    return result, stats, config

def total(stats, suffix):
    return sum(v for k, v in stats.items() if k.endswith(suffix))

writer = csv.writer(sys.stdout)
writer.writerow(['T', 'cache_us', 'spm_us', 'speedup', 'spm_copies',
                 'cache_packets', 'spm_packets', 'cache_dram_read_bytes',
                 'spm_dram_read_bytes', 'matching_hash'])
for tokens in (128, 256, 512, 1024):
    a, sa, ca = read_run(root / f'cache_pretouch_t{tokens}')
    b, sb, cb = read_run(root / f'spm_konly_overlap_t{tokens}')
    for key in ('T', 'Hq', 'Hkv', 'd', 'cores', 'Br', 'Bc', 'hash'):
        assert a[key] == b[key], (tokens, key)
    assert b['variant'] == 'spm_private'
    for section in ca.sections():
        if re.fullmatch(r'system.cpu\d+', section):
            for key in ('numROBEntries', 'numIQEntries', 'LQEntries', 'SQEntries',
                        'fetchWidth', 'decodeWidth', 'renameWidth', 'commitWidth',
                        'issueWidth', 'dispatchWidth', 'wbWidth', 'cacheLoadPorts',
                        'cacheStorePorts', 'spmLoadPorts', 'spmStorePorts',
                        'numPhysIntRegs', 'numPhysVecRegs'):
                assert ca[section][key] == cb[section][key], (section, key)
        for key in ('size', 'assoc', 'clock', 'sve_vl_se'):
            if key in ca[section]:
                assert ca[section][key] == cb[section][key], (section, key)
    issued = total(sb, '.m_spmcpRequestsIssued')
    assert issued == total(sb, '.m_spmcpRequestsCompleted') > 0
    assert total(sb, '.m_spmcpClaimRequestsIssued') == 0
    assert total(sa, '.m_spmcpRequestsIssued') == 0
    writer.writerow([tokens, sa['simTicks']/1e6, sb['simTicks']/1e6,
                     sa['simTicks']/sb['simTicks'], int(issued),
                     total(sa, 'network.packets_injected::total'),
                     total(sb, 'network.packets_injected::total'),
                     total(sa, '.dram.bytesRead::total'),
                     total(sb, '.dram.bytesRead::total'), a['hash']])
