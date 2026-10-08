"""Validate native VL16 scalar FA pairs; no assumptions about VL4 hashes."""
import json
from pathlib import Path
import sys
from validate_attribution import read, validate

def check(root, t):
    result = validate(root, t, expected_vl=16)
    a, sa, ca = read(root/f'cache_t{t}_p1', expected_vl=16)
    b, sb, cb = read(root/f'spm_t{t}_p1', expected_vl=16)
    assert set(ca.sections()) == set(cb.sections())
    for section in ca.sections():
        assert set(ca[section]) == set(cb[section])
        for key, x in ca[section].items():
            y = cb[section][key]
            if x == y or (section == 'system.cpu0.workload' and key in ('cmd', 'executable')):
                continue
            assert key == 'num_spm_ways' and x == '0' and y == '4', (section,key,x,y)
    for cfg in (ca, cb):
        for core in range(8):
            assert cfg[f'system.cpu{core}.isa']['sve_vl_se'] == '16'
            assert cfg[f'system.ruby.l1_cntrl{core}.cache.replacement_policy']['type'] == 'LRURP'
    for key in ('T', 'd', 'Hq', 'Hkv', 'cores', 'Br', 'Bc', 'hash'):
        assert a[key] == b[key], key
    assert a['Bc'] == '128'
    assert sb['m_spmcpRequestsIssued'] == sb['m_spmcpRequestsCompleted'] == 8*(t//128)*64*4
    result['speedup'] = sa['roi_us']/sb['roi_us']
    return result

if __name__ == '__main__':
    print(json.dumps({t:check(Path(sys.argv[1]),t) for t in map(int,sys.argv[2:] or ['128','1024'])},indent=2))
