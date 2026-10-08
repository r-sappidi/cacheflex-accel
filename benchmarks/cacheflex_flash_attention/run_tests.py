"""Run host-only FA unit tests; never launches gem5 or writes result runs."""
import argparse
from pathlib import Path
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', default='test_*.py',
                        help='unittest filename pattern within tests/')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    suite = unittest.defaultTestLoader.discover(
        str(here / 'tests'), pattern=args.pattern)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
