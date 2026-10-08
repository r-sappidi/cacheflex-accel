import unittest,sys,copy,json,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from plan import cases,parameters,work,argv
class Tests(unittest.TestCase):
 def test_pairs(self):
  c=cases();self.assertEqual(len(c),25);self.assertEqual(len({x['name'] for x in c}),25)
  for t in (256,4096):
   for v in ('control','stage','packed','l1','l2','both'):
    self.assertEqual({x['spm'] for x in c if x['t']==t and x['variant']==v and x['kind']=='fa'},{False,True})
 def test_staging_addresses(self):
  for rows in (128,256,512,1024):
   old=[((((r%1024)+((r//1024)*2<<10))<<6)|(w<<16)) for r in range(rows) for w in range(2)]
   new=[(r<<6)|(w<<16) for r in range(rows) for w in range(2)]
   self.assertEqual(old,new)
 def test_packed_mapping(self):
  # Unique integer bit patterns independently specify column-major eight-row blocks.
  for bc in (128,256,512):
   original=[[r*bc+k for k in range(bc)] for r in range(16)]
   packed=[original[mb*8+r][k] for mb in range(2) for k in range(bc) for r in range(8)]
   direct=[None]*(16*bc)
   for r in range(16):
    for x in range(0,bc,64):
     for lane in range(64):direct[(r//8)*bc*8+(x+lane)*8+r%8]=original[r][x+lane]
   self.assertEqual(packed,direct)
 def test_work(self):
  self.assertEqual(work(4096)[2],974848);self.assertEqual(work(256)[2],7168)
  self.assertEqual(work(4096)[0],[28]*8)
  self.assertEqual(work(256)[0],[2,2,2,2,2,2,1,1])
  self.assertEqual(work(256)[1],[512,512,512,512,512,512,256,256])
  for t in (256,512,1024,2048,4096):
   expected=[0]*8;updates=[0]*8;groups=t//256
   for task in range(14*groups):
    head=task//groups;group=(task%groups+head)%groups
    expected[task%8]+=1;updates[task%8]+=256*(group+1)
   self.assertEqual(work(t)[:2],(expected,updates))
 def test_parameter_scope(self):
  a=parameters('both',True);self.assertEqual(len(a),24)
  self.assertEqual(sum(v==16 for v in a.values()),8);self.assertEqual(sum(v==40 for v in a.values()),8)
  self.assertEqual(parameters('packed',True),parameters('stage',True))
 def test_config_rejections(self):
  import run
  c=cases()[1];path=run.OUT/'runs'/c['name'];cfg=copy.deepcopy(run.read(run.PARENT/'config.json'))
  cfg=json.loads(json.dumps(cfg).replace(str(run.PARENT),str(path)))
  cfg['system']['cpu'][0]['workload'][0]['executable']=str(run.G/'guest.export-safe')
  cfg['system']['cpu'][0]['workload'][0]['cmd']=['kv-fa',*map(str,argv(c))]
  for i in range(8):cfg['system']['ruby'][f'l1_cntrl{i}']['cache']['num_spm_ways']=0
  from unittest.mock import patch
  actual_read=run.read
  def check(changed):
   with patch.object(run,'read',side_effect=lambda p:changed if Path(p)==path/'config.json' else actual_read(p)):
    return run.config_check(path,c,run.command(c,path))
  check(cfg)
  for mutate in ('capacity','argv','guest','timing'):
   bad=copy.deepcopy(cfg)
   if mutate=='capacity':bad['system']['ruby']['l1_cntrl0']['cache']['size']*=2
   if mutate=='argv':bad['system']['cpu'][0]['workload'][0]['cmd'][1]='512'
   if mutate=='guest':bad['system']['cpu'][0]['workload'][0]['executable']='wrong'
   if mutate=='timing':bad['system']['ruby']['l0_cntrl0']['dataMissCredits']['capacity']=64
   with self.assertRaises(AssertionError):check(bad)
if __name__=='__main__':unittest.main()
