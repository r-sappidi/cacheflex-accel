"""Isolated single-head, query-parallel port of the corrected VL16 driver."""
from pathlib import Path
import json
import subprocess
import sys
from build_pv_improvements import build_variant


def transform(source):
    replacements = {
        '(size_t)Cores*t*dim': '(size_t)t*dim',
        '(size_t)Cores*T*d': '(size_t)T*d',
        '(size_t)8*T*d': '(size_t)T*d',
        'n_ib=(T+Br-1)/Br': 'n_ib=(T/Cores+Br-1)/Br',
        '(size_t)c.tid*T*d': '(size_t)c.tid*(T/Cores)*d',
        'Hq=8 Hkv=1': 'Hq=1 Hkv=1',
    }
    for old, new in replacements.items():
        assert old in source, old
        source = source.replace(old, new)
    anchor = 'if(d!=64 ||'
    assert source.count(anchor) == 1
    source = source.replace(anchor,
        'if(T<=0 || Br<=0 || T%(Cores*Br)!=0) return 2;\n '+anchor)
    # Full query blocks per worker: no partial/empty partitions. K/V still
    # traverse the full T, while each worker owns exactly T/8 output rows.
    return source


def build(out, config):
    assert config['placement'] in ('cache', 'v')
    assert not config['overlap'] and not config['offset']
    build_variant(out, config)
    path = out/'driver.cpp'
    path.write_text(transform(path.read_text()))
    subprocess.run(['bash', str(out/'build_command.sh')], check=True)
    (out/'single_head.json').write_text(json.dumps(dict(
        heads=1, cores=8, mapping='contiguous query blocks',
        constraint='T divisible by 8*Br', config=config), indent=2))


if __name__ == '__main__':
    build(Path(sys.argv[1]).resolve(), json.loads(Path(sys.argv[2]).read_text()))
