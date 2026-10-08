import unittest,json,tempfile
from pathlib import Path
from analyze import parse
from run import config_equal,plan
class Diagnostics(unittest.TestCase):
 def fixture(self):
  specs=[(0,'NI_PACKET','packet=1 src=8 dst=0 vnet=1 count=1'),(0,'NI_CLASS','packet=1 kind=response type=DATA spm_dst=0x60000000 born=0 source_ready=0'),(1,'NI_STATE','credit=0 older=0'),(2,'NI_STATE','credit=1 older=0'),(3,'NI_STATE','credit=1 older=0'),(3,'NI_SEND',''),(5,'ROUTER_ARRIVE','router=1'),(5,'ROUTER_READY','router=1 ready=6'),(7,'SA_NO_CREDIT','router=1 out=0'),(9,'SA_GRANT','router=1'),(12,'NI_ARRIVE',''),(12,'NI_EJECT_STALL','quota=0 direct=0'),(15,'NI_COMPLETE','')]
  return [f'{t}: ni: CFT {e} msg=0x1 addr=0x800 '+(d if e in ('NI_PACKET','NI_CLASS') else 'packet=1 flit=0 count=1 '+d)+'\n' for t,e,d in specs]
 def test_intervals_and_classification(self):
  r=parse(self.fixture());p=r['packets'][0]
  self.assertEqual(p['group'],'copy_DATA');self.assertEqual(p['parts_ticks'],dict(source_message_age=0,source_ready_wait=0,ni_tail_wait=3,router_pipeline=1,router_wait=3,crossbar_and_links=5,destination_wait=3))
  self.assertEqual(p['ni_samples'],dict(no_credit=1,eligible_not_selected=1,selected=1))
 def test_censored_not_invented(self):
  r=parse(self.fixture()[:-1]);self.assertEqual(r['complete_packets'],0);self.assertEqual(r['censored_packets'],1)
 def test_coherent_distinct(self):
  r=parse([x.replace('spm_dst=0x60000000','spm_dst=0') for x in self.fixture()]);self.assertIn('coherent_DATA',r['groups'])
 def test_invalid_negative_interval_rejected(self):
  f=self.fixture();f=[x.replace('ready=6','ready=10') for x in f]
  with self.assertRaises(AssertionError):parse(f)
 def test_config_negative(self):
  with tempfile.TemporaryDirectory() as t:
   a=Path(t)/'a';b=Path(t)/'b';a.mkdir();b.mkdir()
   (a/'config.json').write_text(json.dumps(dict(capacity=32,out=str(a))))
   (b/'config.json').write_text(json.dumps(dict(capacity=32,out=str(b))));config_equal(a,b)
   (b/'config.json').write_text(json.dumps(dict(capacity=64,out=str(b))))
   with self.assertRaises(AssertionError):config_equal(a,b)
 def test_plan_reuses_full_controls(self):
  p=plan();self.assertEqual(len(p),10)
  for i in (2,4,6,8):
   a,b=p[i:i+2];self.assertEqual(a['source'],b['source']);self.assertEqual(a['stop'],b['stop']);self.assertFalse(a['trace']);self.assertTrue(b['trace'])
if __name__=='__main__':unittest.main()
