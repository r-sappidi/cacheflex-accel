"""Strict hardware matching plus observer invariance for the clean FA pair."""
import json
from pathlib import Path
import sys
from validate_attribution import read, validate


def compare_configs(a, b, allow_spm):
    assert set(a.sections()) == set(b.sections())
    differences = []
    for section in a.sections():
        assert set(a[section]) == set(b[section]), section
        for key in a[section]:
            x, y = a[section][key], b[section][key]
            if x == y:
                continue
            if section == 'system.cpu0.workload' and key in ('cmd', 'executable'):
                continue
            assert allow_spm and key == 'num_spm_ways' and x == '0' and y == '3', (section, key, x, y)
            differences.append((section, key, x, y))
    for cfg in (a, b):
        for core in range(8):
            assert cfg[f'system.ruby.l1_cntrl{core}.cache.replacement_policy']['type'] == 'LRURP'
    return differences


if __name__ == '__main__':
    root = Path(sys.argv[1])
    results = {}
    for t in map(int, sys.argv[2:] or ['128', '1024']):
        pair = validate(root, t)
        a, sa, ca = read(root/f'cache_t{t}_p1')
        b, sb, cb = read(root/f'spm_t{t}_p1')
        for key in ('T', 'd', 'Hq', 'Hkv', 'cores', 'Br', 'Bc', 'hash'):
            assert a[key] == b[key], key
        pair['allowed_config_differences'] = compare_configs(ca, cb, True)
        pair['speedup'] = sa['roi_us']/sb['roi_us']
        if t == 1024:
            g, sg, cg = read(root/'spm_group4_t1024_p1')
            compare_configs(cb, cg, False)
            assert g['hash'] == a['hash']
            assert sg['m_spmcpRequestsIssued'] == sg['m_spmcpRequestsCompleted'] == 67584
            pair['group4'] = dict(output=g, counters=sg, speedup=sa['roi_us']/sg['roi_us'])
        results[t] = pair
    print(json.dumps(results, indent=2))
