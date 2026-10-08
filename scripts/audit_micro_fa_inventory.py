#!/usr/bin/env python3
"""Read-only static inventory; does not claim execution or arithmetic proof."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / 'results/cacheflex_flash_attention/micro_upstream_20260918'
REBUTTAL = ROOT / 'results/cacheflex_flash_attention/micro_rebuttal_audit_20260921'


def tracked(repo):
    return subprocess.check_output(['git', '-C', str(repo), 'ls-files'], text=True).splitlines()


def main():
    rows = []
    for name in tracked(MASTER):
        if not (name.startswith(('experiments/', 'kernels/'))
                and name.endswith('.cpp') and 'flash' in Path(name).name):
            continue
        data = (MASTER / name).read_bytes()
        lines = data.decode().splitlines()
        rows.append({
            'path': name,
            'sha256': hashlib.sha256(data).hexdigest(),
            'same_on_rebuttal': (REBUTTAL / name).read_bytes() == data,
            'widen_lines': [i for i, s in enumerate(lines, 1) if 'svcvt_f32_f16' in s],
            'narrow_lines': [i for i, s in enumerate(lines, 1) if 'svcvt_f16_f32' in s],
            'zip_unzip_intrinsic_lines': [i for i, s in enumerate(lines, 1)
                                          if re.search(r'sv(zip|uzp)', s)],
            'validation_keyword_lines': [i for i, s in enumerate(lines, 1)
                                         if re.search(r'allclose|isclose|tolerance|mismatch|assert\(', s, re.I)],
        })
    rebuttal_extra = [s for s in tracked(REBUTTAL)
                      if s.endswith('.cpp') and 'flash' in Path(s).name
                      and s not in {r['path'] for r in rows}]
    print(json.dumps({'scope': 'Current master and rebuttal tips, not all historical commits',
                      'static_only': True, 'count': len(rows), 'sources': rows,
                      'additional_rebuttal_flash_cpp': rebuttal_extra}, indent=2))


if __name__ == '__main__':
    main()
