"""Require functional completion and observer-off/on timing equivalence."""
import configparser
import json
from pathlib import Path
import re
import sys
from analyze_traffic import analyze


def read(path, expected_vl=4):
    text = (path/'run.log').read_text()
    assert all(x in text for x in ('FLASH_ROI begin', 'FLASH_ROI end',
                                  'exiting with last active thread context'))
    assert not re.search(r'panic:|fatal:', text)
    row = dict(re.findall(r'(\w+)=([^\s]+)', text.split('FLASH_RESULT ')[-1].splitlines()[0]))
    assert row['nonfinite'] == '0'
    cfg = configparser.ConfigParser(interpolation=None, strict=False)
    cfg.read(path/'config.ini')
    for section in cfg.sections():
        for key, value in list(cfg[section].items()):
            cfg[section][key] = value.replace(str(path.resolve()), '<RESULT>')
    assert cfg[cfg['system.cpu0']['clk_domain']]['clock'] == '400'
    assert cfg['system.cpu0.isa']['sve_vl_se'] == str(expected_vl)
    return row, analyze(path), cfg


def validate(root, tokens, expected_vl=4):
    rows = {}
    for kind in ('cache', 'spm'):
        a, sa, ca = read(root/f'{kind}_t{tokens}_p0', expected_vl)
        b, sb, cb = read(root/f'{kind}_t{tokens}_p1', expected_vl)
        assert a == b, (kind, 'output mismatch')
        for key in sa.keys() - {'path'}:
            assert sa[key] == sb[key], (kind, key, sa[key], sb[key])
        for section in ca.sections():
            for key, val in ca[section].items():
                if key not in ('cacheflexpcprofile', 'cacheflex_message_profile'):
                    assert val == cb[section][key], (section, key)
        rows[kind] = dict(output=b, counters=sb)
    assert rows['cache']['output']['hash'] == rows['spm']['output']['hash']
    assert rows['cache']['counters']['m_spmcpRequestsIssued'] == 0
    spm = rows['spm']['counters']
    assert spm['m_spmcpRequestsIssued'] == spm['m_spmcpRequestsCompleted'] > 0
    return rows


if __name__ == '__main__':
    root = Path(sys.argv[1])
    print(json.dumps({t: validate(root, t) for t in map(int, sys.argv[2:] or ['128', '1024'])}, indent=2))
