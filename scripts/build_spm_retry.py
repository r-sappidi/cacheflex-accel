"""Isolated bounded silent-retry repair; retain original failed placement build."""
from pathlib import Path
import os
import build_finite_micro as build

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/cacheflex_flash_attention/spm_set_spread_20260927'

if __name__ == '__main__':
    build.BASE = BASE
    build.BUILD = BASE/os.environ.get('CACHEFLEX_RETRY_BUILD', 'build_retry_v3')
    build.OLD = BASE/'build'
    build.main()
