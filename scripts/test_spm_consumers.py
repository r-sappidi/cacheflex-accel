import unittest
from analyze_spm_consumers import parse_lines,consumer_rows
from analyze_spm_fma_consumers import fma_rows, root_producer

def sample(load_done=9,other_done=20,late_dispatch=None,wait=False):
    logs=[]
    def emit(t,event,fields):logs.append(f'{t*400}: system.cpu0.iq: CFD {event} {fields}\n')
    emit(0,'NODE','sn=1 pc=0x100 upc=0 op=1 load=1 store=0 asm=spm.ld1h z0.h')
    emit(0,'DST','sn=1 idx=0 reg=20')
    emit(0,'READY','sn=1 mem=1')
    emit(0,'ISSUE','sn=1 pc=0x100 op=1 unit=1 latency=1')
    emit(1,'NODE','sn=2 pc=0x104 upc=0 op=2 load=0 store=0 asm=fmla z1.h, z2.h, z3.h')
    emit(1,'DST','sn=2 idx=0 reg=21')
    emit(1,'ISSUE','sn=2 pc=0x104 op=2 unit=2 latency=3')
    dispatch=late_dispatch if late_dispatch else 3
    emit(dispatch,'NODE','sn=3 pc=0x108 upc=0 op=2 load=0 store=0 asm=fmla z1.h, z0.h, z2.h')
    emit(dispatch,'SRC',f'sn=3 idx=0 reg=20 ready={int(dispatch>=load_done)}')
    emit(dispatch,'SRC',f'sn=3 idx=1 reg=21 ready={int(dispatch>=other_done)}')
    emit(load_done,'DONE','sn=1 pc=0x100 op=1 load=1 spm=1 va=0x0')
    emit(load_done+1,'COMMIT','sn=1 pc=0x100 upc=0')
    emit(other_done,'DONE','sn=2 pc=0x104 op=2 load=0 spm=0 va=0x0')
    ready=max(dispatch,load_done,other_done)
    emit(ready,'READY','sn=3 mem=0')
    if wait:emit(ready,'WAIT','sn=3 op=2 reason=fu_unavailable issued=2')
    emit(ready+1,'ISSUE','sn=3 pc=0x108 op=2 unit=2 latency=3')
    emit(ready+4,'DONE','sn=3 pc=0x108 op=2 load=0 spm=0 va=0x0')
    emit(ready+5,'COMMIT','sn=3 pc=0x108 upc=0')
    return sorted(logs,key=lambda x:int(x.split(':',1)[0]))

class ConsumerTests(unittest.TestCase):
    def rows(self,**kw):
        n,_,_=parse_lines(sample(**kw));return consumer_rows(n)[0]
    def test_late_fma_masks_earlier_spm(self):
        slow=self.rows()[0];fast=self.rows(load_done=5)[0]
        self.assertEqual(slow['load_latency_cycles'],9)
        self.assertEqual(fast['load_latency_cycles'],5)
        self.assertEqual(slow['issue'],fast['issue'])
        self.assertEqual(slow['reason'],'other_operand:FMA_accumulator_or_compute')
        self.assertEqual(fast['data_to_ready_cycles']-slow['data_to_ready_cycles'],4)
    def test_actual_ready_queue_stall(self):
        r=self.rows(other_done=4,wait=True)[0]
        self.assertEqual(r['reason'],'SPM_is_last_or_tied_ready_operand')
        self.assertEqual(r['ready_queue_wait_cycles'],{'fu_unavailable':1})
    def test_late_consumer_keeps_ready_source_provenance(self):
        r=self.rows(other_done=4,late_dispatch=15)[0]
        self.assertEqual(r['reason'],'consumer_dispatched_after_SPM_ready')
    def test_unknown_source_not_misclassified(self):
        lines=sample();lines.insert(0,'1200: system.cpu0.iq: CFD SRC sn=3 idx=2 reg=999 ready=0\n')
        n,_,_=parse_lines(lines);r=consumer_rows(n)[0][0]
        self.assertEqual(r['reason'],'incomplete_source_provenance')
    def test_uncommitted_consumer_excluded(self):
        n,_,_=parse_lines([x for x in sample() if 'COMMIT sn=3' not in x])
        r,e=consumer_rows(n);self.assertEqual(r,[]);self.assertEqual(e['no_complete_consumer'],1)
    def test_first_fma_not_accumulator_descendants(self):
        n,_,_=parse_lines(sample())
        rows,excluded=fma_rows(n)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['consumer']['sn'],3)
        self.assertEqual(rows[0]['reason'],'other_operand:FMA_accumulator_or_compute')
    def test_lane_assembly_reaches_coherent_root(self):
        n,_,_=parse_lines(sample())
        producer=n[(0,2)]
        producer['asm']='splice z1, p0, z2, z3'
        producer['sources']=[dict(producer=4),dict(producer=4)]
        n[(0,4)]=dict(core=0,sn=4,asm='ld1rqh z2, [x0]',load=1,
                      done=18*400,issues=[2*400],sources=[])
        kind,chain=root_producer(producer,n)
        self.assertEqual(kind,'coherent_load')
        self.assertEqual([p['sn'] for p in chain],[2,4])
        del n[(0,4)]
        self.assertEqual(root_producer(producer,n)[0],'unknown_boundary')
    def test_transparent_splice_not_reported_as_final_use(self):
        n,_,_=parse_lines(sample(other_done=4))
        c=n[(0,3)]
        n[(0,4)]=dict(core=0,sn=4,asm='splice z0, p0, z0, z0',
                      dispatch=3*400,ready=9*400,done=11*400,commit=14*400,
                      issues=[9*400],waits=[],sources=[dict(producer=1)])
        c['sources'][0]['producer']=4
        c['ready']=11*400;c['issues']=[11*400];c['done']=15*400;c['commit']=16*400
        rows,_=fma_rows(n)
        self.assertEqual([p['sn'] for p in rows[0]['operand_path']],[1,4])
        self.assertEqual(rows[0]['consumer']['sn'],3)

if __name__=='__main__':unittest.main()
