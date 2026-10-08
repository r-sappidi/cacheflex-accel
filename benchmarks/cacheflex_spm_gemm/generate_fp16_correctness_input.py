#!/usr/bin/env python3
"""Generate deterministic, exactly representable FP16 GEMM inputs/reference."""

import struct
import sys


def a_value(row: int, kk: int) -> float:
    magnitude = 1.0 if ((row + 3 * kk) & 2) else 0.5
    return -magnitude if ((5 * row + kk) & 1) else magnitude


def b_value(kk: int, col: int) -> float:
    magnitude = 0.5 if ((kk + 7 * col) & 2) else 0.25
    return -magnitude if ((3 * kk + col) & 1) else magnitude


def main() -> int:
    if len(sys.argv) != 5:
        print(f"usage: {sys.argv[0]} M K N OUTPUT", file=sys.stderr)
        return 2
    m, k, n = map(int, sys.argv[1:4])
    path = sys.argv[4]
    if min(m, k, n) <= 0:
        raise ValueError("matrix dimensions must be positive")

    # Every product and partial sum is a multiple of 1/8 with magnitude below
    # 128 for the target K=256, so FP16 FMLA incurs no rounding ambiguity.
    expected_by_residue = {
        (row, col): sum(
            a_value(row, kk) * b_value(kk, col) for kk in range(k)
        )
        for row in range(4)
        for col in range(4)
    }
    with open(path, "wb") as output:
        for row in range(m):
            for kk in range(k):
                output.write(struct.pack("<e", a_value(row, kk)))
        for kk in range(k):
            for col in range(n):
                output.write(struct.pack("<e", b_value(kk, col)))
        for row in range(m):
            for col in range(n):
                expected = expected_by_residue[(row & 3, col & 3)]
                output.write(struct.pack("<e", expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
