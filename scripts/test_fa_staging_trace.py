import tempfile
import unittest
from pathlib import Path
from run_fa_staging_trace import choose_window, extract_context, trace_lines


class StagingTraceTests(unittest.TestCase):
    def test_binary_context_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'trace';dst=Path(tmp)/'context'
            matching=b'12: system.cpu0: [sn:100] data=\xff\x80\x00\n'
            address=b'13: global: 401e14 payload=\xfe\n'
            src.write_bytes(matching+b'12: system.cpu1: [sn:100]\n'+address)
            report=extract_context(src,dst,100)
            self.assertEqual(dst.read_bytes(),matching+address)
            self.assertEqual(report['non_utf8_lines'],2)
            self.assertEqual(report['selected_lines'],2)
            self.assertIn('\\xff',next(trace_lines(src)))

    def test_excludes_compute_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace'
            p.write_text('\n'.join(f'{t}: system.cpu0.commit: CFBOUND pc={pc} sn={sn}' for t,pc,sn in [
                (10,'0x401db0',1),(20,'0x401e14',2),(40,'0x401e14',3),
                (50,'0x401e94',4),(10000,'0x401db0',5),(10010,'0x401e14',6),
                (10050,'0x401e14',7)]))
            r=choose_window(p,0,20000)
            self.assertEqual(r['count'],2)
            self.assertEqual(r['selected']['sn'],7)
            self.assertEqual(r['selected']['gap'],40)

    def test_no_completed_pairs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace';p.write_text('')
            with self.assertRaises(AssertionError):choose_window(p,0,20000)


if __name__=='__main__':unittest.main()
