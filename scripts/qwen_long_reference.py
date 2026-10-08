"""Bounded-memory, single-thread full Qwen oracle for L3-10."""
import argparse
import os
import time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from build_access_timing import save, sha
from qwen_l3_reference import reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', type=int, required=True)
    parser.add_argument('--bc', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.t in (16384, 32768) and args.bc in (256, 512)
    assert not args.out.exists(), 'Do not overwrite reference evidence'
    start = time.time()

    def progress(heads, total):
        save(args.out.with_suffix('.progress.json'), dict(
            pid=os.getpid(), completed_heads=heads, total_heads=total,
            elapsed_seconds=time.time()-start, updated=time.time()))

    progress(0, 14)
    result = reference(args.t, args.bc, hq=14, hkv=2, causal=True,
                       chunk=64, progress=progress)
    assert result.shape == (14, args.t, 64) and np.isfinite(result).all()
    temporary = args.out.with_suffix('.partial')
    with temporary.open('wb') as f:
        np.save(f, result)
    temporary.replace(args.out)
    save(args.out.with_suffix('.json'), dict(
        t=args.t, bc=args.bc, hq=14, hkv=2, causal=True, chunk=64,
        elements=result.size, sha256=sha(args.out),
        host_seconds=time.time()-start, full_reference=True))


if __name__ == '__main__':
    main()
