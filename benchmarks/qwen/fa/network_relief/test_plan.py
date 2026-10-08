import os,unittest,json,tempfile
from pathlib import Path
os.environ.update(FA43_ARM='cache_buf8',FA43_CPU_OFFSET='0')
import plan,run
class Checks(unittest.TestCase):
 def test_only_buffer_and_placement(self):
  for spm in (False,True):
   p=plan.parameters('buf16',spm)
   self.assertEqual(len(p),9);self.assertEqual(p.pop('system.ruby.network.buffers_per_data_vc'),16)
   self.assertTrue(all(v==(6 if spm else 0) for v in p.values()))
 def test_control_provenance(self):
  for arm in ('cache_buf8','spm_buf8','cache_buf16','spm_buf16'):
   os.environ['FA43_ARM']=arm
   for c in plan.cases():
    p=plan.control_path(c);a=run.read(p/'accepted.json')
    self.assertEqual(run.sha(p/'output.fp16'),a['output_sha256']);self.assertEqual(a['case']['t'],c['t']);self.assertEqual(a['case']['spm'],c['spm'])
 def test_strict_config_and_negative(self):
  os.environ['FA43_ARM']='spm_buf8';c=plan.cases()[0]
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);cmd=run.command(c,p);d=run.read(run.PARENT/'config.json')
   def normalize(x):
    if isinstance(x,dict):return {k:normalize(v) for k,v in x.items()}
    if isinstance(x,list):return [normalize(v) for v in x]
    return x.replace(str(run.PARENT),str(p)) if isinstance(x,str) else x
   d=normalize(d);d['system']['ruby']['network']['buffers_per_data_vc']=8
   w=d['system']['cpu'][0]['workload'][0];w['executable']=str(run.G/'guest.export-safe');w['cmd']=['kv-fa',*map(str,plan.argv(c))]
   run.save(p/'config.json',d);run.config_check(p,c,cmd)
   d['system']['ruby']['network']['buffers_per_data_vc']=16;run.save(p/'config.json',d)
   with self.assertRaises(AssertionError):run.config_check(p,c,cmd)
if __name__=='__main__':unittest.main()
