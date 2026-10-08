"""Standalone recorder tests: no gem5 rebuild or running simulation required."""
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest

from decode_ruby_flight import decode

ROOT = Path(__file__).resolve().parents[1]
SOURCE = r'''
#include "mem/ruby/slicc_interface/FlightRecorder.hh"
#include <csignal>
int main(int argc, char **argv) {
    if (argc > 2 && std::string(argv[2]) == "invalid") {
        try { gem5::ruby::FlightRecorder invalid(argv[1], 0); }
        catch (const std::runtime_error &) { return 0; }
        return 1;
    }
    if (argc > 2 && std::string(argv[2]) == "existing") {
        try { gem5::ruby::FlightRecorder existing(argv[1], 3); }
        catch (const std::runtime_error &) { return 0; }
        return 1;
    }
    gem5::ruby::FlightRecorder recorder(argv[1], 3);
    for (int i=1; i<=10; ++i) recorder.append(i*100, "event="+std::to_string(i));
    if (argc > 2) {
        recorder.append(1100, std::string(2000, 'x'));
        raise(SIGKILL);
    }
}
'''


class FlightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT / 'scripts')
        cls.root = Path(cls.temp.name)
        source = cls.root / 'test.cc'
        source.write_text(SOURCE)
        cls.exe = cls.root / 'test'
        subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                        '-I'+str(ROOT/'gem5/src'), str(source), '-o', str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_wrap(self):
        path = self.root / 'wrap.flight.bin'
        subprocess.run([self.exe, path], check=True)
        rows, rejected = decode(path)
        self.assertEqual([r[2] for r in rows], [8, 9, 10])
        self.assertEqual(rejected, 0)
        self.assertEqual(path.stat().st_size, 64 + 3 * 1024)

    def test_reject_invalid_and_existing(self):
        path = self.root / 'invalid.flight.bin'
        subprocess.run([self.exe, path, 'invalid'], check=True)
        self.assertFalse(path.exists())
        path.write_bytes(b'preserve this file')
        subprocess.run([self.exe, path, 'existing'], check=True)
        self.assertEqual(path.read_bytes(), b'preserve this file')

    def test_kill_and_corruption(self):
        path = self.root / 'kill.flight.bin'
        result = subprocess.run([self.exe, path, 'kill'])
        self.assertEqual(result.returncode, -signal.SIGKILL)
        rows, rejected = decode(path)
        self.assertEqual([r[2] for r in rows], [9, 10, 11])
        self.assertEqual(rejected, 0)
        self.assertTrue(rows[-1][3].endswith('[TRUNCATED]'))
        with path.open('r+b') as f:
            f.seek(64 + 24)
            f.write(b'!')
        rows, rejected = decode(path)
        self.assertEqual(rejected, 1)
        self.assertEqual([r[2] for r in rows], [9, 11])
        with path.open('r+b') as f:
            f.seek(64 + 1024 + 1016)
            f.write(bytes(8))
        rows, rejected = decode(path)
        self.assertEqual(rejected, 2)
        self.assertEqual([r[2] for r in rows], [9])


if __name__ == '__main__':
    unittest.main()
