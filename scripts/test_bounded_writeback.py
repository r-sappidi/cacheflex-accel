"""Static routing regression and preserved dynamic failure evidence checks."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / 'gem5/src/mem/ruby/protocol'


class WritebackOrderingTest(unittest.TestCase):
    def test_return_input_precedes_new_requests(self):
        source = (PROTO / 'MESI_Three_Level_SPM-L1cache.sm').read_text()
        self.assertLess(source.index('in_port(spmBufferFromL0_in,'),
                        source.index('in_port(messageBufferFromL0_in,'))

    def test_eviction_and_recall_share_ordering_domain(self):
        source = (PROTO / 'MESI_Three_Level_SPM-L0cache.sm').read_text()
        for name in ('g_issuePUTX', 'f_sendDataToL1', 'fi_sendInvAck'):
            action = source.split('action(' + name + ',', 1)[1].split('\n  action(', 1)[0]
            self.assertIn('enqueue(replyNetwork_out,', action)
            self.assertNotIn('enqueue(requestNetwork_out,', action)

    def test_writeback_transitions_preserve_data_and_queue(self):
        source = (PROTO / 'MESI_Three_Level_SPM-L1cache.sm').read_text()
        for states, target in [('{M,E}', 'MM'), ('{M_IL0, E_IL0}', 'MM_IL0'),
                               ('{EX_L0, MX_L0}', 'MX_L0')]:
            header = f'transition({states}, WriteBackEscape, {target})'
            actions = source.split(header, 1)[1].split('}', 1)[0]
            self.assertIn('u_writeDataFromEscape;', actions)
            self.assertIn('k_spmPopL0RequestQueue;', actions)
        self.assertNotRegex(source, r'transition\(M_I, WriteBackEscape')

    def test_original_failure_is_ack_overtaking_eviction(self):
        path = ROOT / 'results/cacheflex_flash_attention/hardware_bounds_20260921/diagnosis/writeback_v3'
        lines = [x for x in (path / 'protocol.log').read_text().splitlines()
                 if '[0x792080, line 0x792080]' in x and 'Resource Stall' not in x]
        joined = '\n'.join(lines)
        self.assertRegex(joined, r'L0_Replacement\s+M>I')
        self.assertRegex(joined, r'L0_AckEscape\s+M_IL0>MM')
        self.assertRegex(joined, r'L1_Replacement\s+MM>M_I')
        self.assertNotIn('WriteBack ', joined)
        self.assertIn('event: WriteBack state: M_I', (path / 'run.log').read_text())

    def test_v4_failure_is_refetch_overtaking_eviction(self):
        path = ROOT / 'results/cacheflex_flash_attention/hardware_bounds_20260921/diagnosis/writeback_v4'
        lines = [x for x in (path / 'protocol.log').read_text().splitlines()
                 if '[0x601f80, line 0x601f80]' in x and '225842000' in x]
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[0], r'Store\s+M>M')
        self.assertRegex(lines[1], r'WriteBackEscape\s+M>MM')
        self.assertIn('event: WriteBackEscape state: MM', (path / 'run.log').read_text())


if __name__ == '__main__':
    unittest.main()
