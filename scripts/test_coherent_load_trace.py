import copy
import unittest
from pathlib import Path
from analyze_coherent_load_trace import assemble
from run_coherent_load_trace import command

def sample():
    key=(0,1)
    n=dict(core=0,sn=1,pc=0x4020c0,dispatch=0,regs_ready=400,ready=800,
           issues=[1200],done=8000,commit=8400,sources=[])
    events=[]
    def e(kind,t,**kw):events.append(dict(event=kind,tick=t,pc=n['pc'],**kw))
    e('ADDRESS',1600,bytes=128);e('TRANSLATED_READ',2000);e('ELIGIBLE',2400)
    e('FRAG_RESULT',2800,id=10,pa=0x1000,bytes=64,accepted=1)
    e('FRAG_RESULT',4000,id=20,pa=0x1040,bytes=64,accepted=1)
    # First-issued fragment comes back LAST.
    e('FRAG_RESPONSE',7200,id=10,bytes=64,squashed=0)
    e('FRAG_RESPONSE',6400,id=20,bytes=64,squashed=0)
    return {key:n},{key:events},{},{}

class LoadTraceTests(unittest.TestCase):
    def test_debug_reuses_resource_check_result(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/'gem5/src/mem/ruby/system/Sequencer.cc').read_text()
        observation=source.split('CFL_PKT("SEQ_REJECT", pkt, "reason=miss_or_writeback_pool',1)[1].split(';',1)[0]
        self.assertNotIn('->full()',observation)
        self.assertIn('micro_misses_full, micro_wb_full',observation)
    def test_tail_and_additive_timeline(self):
        rows,ex,_,_=assemble(*sample(),'spm')
        self.assertFalse(ex);r=rows[0]
        self.assertEqual(r['last_fragment']['packet'],'0xa')
        self.assertEqual(sum(r['intervals'].values()),r['cycles'])
        self.assertEqual(r['intervals']['remaining_memory_response'],8)
        self.assertEqual(r['last_fragment']['service_cycles'],11)
    def test_missing_fragment_not_silently_averaged(self):
        n,l,s,q=sample();l[0,1].pop()
        rows,ex,_,_=assemble(n,l,s,q,'spm')
        self.assertFalse(rows);self.assertEqual(ex['incomplete_fragment_set'],1)
    def test_uncommitted_excluded(self):
        n,l,s,q=sample();del n[0,1]['commit']
        rows,ex,_,_=assemble(n,l,s,q,'spm')
        self.assertFalse(rows);self.assertEqual(ex['boundary_uncommitted_or_unknown_readiness'],1)
    def test_initially_ready_memdep_insert_and_normal_retries(self):
        n,l,s,q=sample();node=n[0,1]
        del node['regs_ready'];node['sources']=[dict(ready=1,reg=9)]
        node['issues']=[1200,2000,2800]
        rows,ex,_,_=assemble(n,l,s,q,'spm')
        self.assertFalse(ex);self.assertEqual(rows[0]['issue_attempts'],3)
        self.assertEqual(rows[0]['intervals']['register_dependencies'],0)
    def test_unobserved_wakeup_not_assumed(self):
        n,l,s,q=sample();node=n[0,1]
        del node['regs_ready'];node['sources']=[dict(ready=0,reg=9)]
        rows,ex,_,_=assemble(n,l,s,q,'spm')
        self.assertFalse(rows)
    def test_ingress_identity_not_address(self):
        n,l,s,q=sample()
        s[0,10]=[dict(event='SEQ_ENQUEUE',tick=2800,msg=99,ready=3200),
                 dict(event='SEQ_DONE',tick=6800,source='L1Cache',external=1)]
        q[99]=[dict(event='QUEUE_DEQUEUE',tick=3600,final=1,name='mandatory')]
        q[88]=[dict(event='QUEUE_DEQUEUE',tick=6000,final=1,name='unrelated')]
        r=assemble(n,l,s,q,'spm')[0][0]
        f=r['last_fragment'];self.assertEqual(f['ingress_ready_to_dequeue_cycles'],1)
        self.assertEqual(f['callback_to_cpu_cycles'],1)
    def test_command_strips_reference_debug(self):
        for v in ('cache','spm'):
            cmd=command(v,512,Path('/tmp/not-run'))
            self.assertFalse(any(x.startswith('--debug') for x in cmd))
            self.assertTrue(cmd[2].endswith('/micro_timing_se.py'))
            traced=command(v,4096,Path('/tmp/not-run'),(100,200),True)
            self.assertEqual(sum(x.startswith('--debug-flags=') for x in traced),1)

if __name__=='__main__':unittest.main()
