"""Synthetic identity/zero-address and panel-accounting parser regression."""
import gzip
from pathlib import Path
import tempfile
import unittest
from analyze_ingress_trace import analyze
from analyze_ingress_credits import reconstruct


class TraceTest(unittest.TestCase):
    def test_actual_credit_lifetime_not_poll_count(self):
        pool='system.ruby.l0_cntrl0.writeCredits'
        es=[dict(tick=100+i,obj=pool,kind='POOL_ALLOC',live=str(i+1),
                 addr=hex(64*i),evidence_line=i+1) for i in range(8)]
        es += [dict(tick=t,obj=pool,kind='POOL_FULL',live='8') for t in (110,120,130)]
        es += [dict(tick=150,obj=pool,kind='POOL_RELEASE',live='7',addr='0x0')]
        full,_=reconstruct(es,[],200)
        segments=full[pool]
        self.assertEqual(len(segments),1)
        self.assertEqual((segments[0]['start'],segments[0]['end']),(107,150))
        self.assertEqual(len(segments[0]['owners']),8)
        self.assertEqual(segments[0]['unknown'],0)

    def test_boundary_unknown_releases(self):
        pool='system.ruby.l1_cntrl0.missCredits'
        es=[dict(tick=10,obj=pool,kind='POOL_FULL',live='20'),
            dict(tick=20,obj=pool,kind='POOL_RELEASE',live='19',addr='0x40')]
        full,_=reconstruct(es,[],100)
        self.assertEqual(full[pool][0]['unknown'],20)
        self.assertEqual(full[pool][0]['end'],20)

    def test_panel_and_consumer(self):
        lines=[]
        for i in range(256):
            sn=i+1; dst=(i//2)*64 | ((i%2)<<16); addr=0x100000+i*64;t=10000+i*400
            stem='system.cpu0.iew.lsq.thread0: [SPM] '
            lines.extend([
                f'{t}: {stem}Reserve Shadow (addr): SN={sn} Addr={dst:#x}\n',
                f'{t}: {stem}Store/Cmd Sent to Cache: SN={sn} Addr={addr:#x} (Type: CP2SPM)\n',
                f'{t}: {stem}Force Complete: SN={sn}\n',
                f'{t+400} 0 L0Cache SPMCP_fetch I>I [{addr:#x}, line {addr:#x}] \n',
                f'{t+800} 0 L1Cache SPMCP_fetch I>IX_D [{addr:#x}, line {addr:#x}] \n',
                f'{t+1600} 0 L1Cache Data IX_D>X [{addr:#x}, line {addr:#x}] \n',
                f'{t+2000}: {stem}Resp Recv: SN={sn} Addr={dst:#x} (Waking 0 blocked loads)\n'])
        lines.append('200000: system.cpu0.iew.lsq.thread0: [SPM] Load Issue to Cache: SN=300 Addr=0\n')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with gzip.open(p/'trace.gz','wt') as f:f.writelines(lines)
            r=analyze(p);panel=r['panels'][0]
            self.assertTrue(panel['complete'])
            self.assertEqual(len(panel['copies']),256)
            self.assertEqual(panel['last_install'],113600)
            self.assertEqual(panel['last_ack'],114000)
            self.assertEqual(panel['first_consumer_issue'],200000)
            self.assertEqual(panel['all_copies_ready_before_first_consumer_cycles'],215)
            self.assertEqual(r['raw_blocked_loads'],0)


if __name__=='__main__':unittest.main()
