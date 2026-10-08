"""Opt-in existing ideal-copy wiring, all ordinary configuration unchanged."""
import os,runpy,sys
from pathlib import Path
P=Path('/data2/rsappidi/cacheflex_mc/results/cacheflex_l3_spm/direct_link_20260928/build_v4/configs/deprecated/example/l3_direct_reduction_se.py')
sys.path.insert(0,str(P.parents[2]))
from ruby import MESI_Three_Level_SPM as protocol
old_create=protocol.create_system
def create(*args,**kwargs):
 result=old_create(*args,**kwargs)
 if os.environ.get('CACHEFLEX_ISOLATE_COPY_ORACLE')=='1':
  ruby=args[5];cpus=args[6]
  for i in range(len(cpus)):
   l0=getattr(ruby,f'l0_cntrl{i}');l1=getattr(ruby,f'l1_cntrl{i}')
   l0.sequencer.cacheflex_ideal_staging_cache=l1.cache
   l0.sequencer.cacheflex_ideal_staging_controller=l1
 return result
protocol.create_system=create
runpy.run_path(str(P),run_name='__m5_main__')
