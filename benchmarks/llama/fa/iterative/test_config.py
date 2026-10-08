import copy,json,tempfile,unittest
from pathlib import Path
import run
from plan import case,argv,parameters
class ConfigTests(unittest.TestCase):
 def test_only_declared_config_changes(self):
  original=run.read(run.PARENT/'config.json');c=case(False,t=256)
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)
   def rewrite(v):
    if isinstance(v,dict):return {k:rewrite(x) for k,x in v.items()}
    if isinstance(v,list):return [rewrite(x) for x in v]
    if isinstance(v,str):return v.replace(str(run.PARENT),str(p))
    return v
   cfg=rewrite(original)
   cfg['system']['cpu'][0]['workload'][0]['executable']=str(run.G/'guest.export-safe')
   cfg['system']['cpu'][0]['workload'][0]['cmd']=['kv-fa',*map(str,argv(c))]
   for i in range(8):cfg['system']['ruby'][f'l1_cntrl{i}']['cache']['num_spm_ways']=0
   (p/'config.json').write_text(json.dumps(cfg));cmd=run.command(c,p);run.config_check(p,c,cmd)
   for typ in ('capacity','argv','guest'):
    bad=copy.deepcopy(cfg)
    if typ=='capacity':bad['system']['ruby']['l0_cntrl0']['dataMissCredits']['capacity']+=1
    if typ=='argv':bad['system']['cpu'][0]['workload'][0]['cmd'][11]='14'
    if typ=='guest':bad['system']['cpu'][0]['workload'][0]['executable']='wrong'
    (p/'config.json').write_text(json.dumps(bad))
    with self.assertRaises(AssertionError):run.config_check(p,c,cmd)
if __name__=='__main__':unittest.main()
