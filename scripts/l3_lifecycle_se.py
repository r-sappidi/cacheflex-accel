"""Explicit first-update initialization and translated near-L3 finalization."""
from pathlib import Path
import runpy
from m5.util import addToPath
addToPath('../..')
from ruby import MESI_Three_Level_SPM as protocol
old_options=protocol.define_options
old_create=protocol.create_system
def define_options(parser):
    old_options(parser)
    parser.add_argument('--cacheflex-l3-initialize',action='store_true')
    parser.add_argument('--cacheflex-l3-finalize',action='store_true')
def create_system(options,full_system,system,dma_ports,bootmem,ruby_system,cpus):
    result=old_create(options,full_system,system,dma_ports,bootmem,ruby_system,cpus)
    if options.cacheflex_l3_initialize or options.cacheflex_l3_finalize:
        assert options.cacheflex_l3_reduction and options.cacheflex_l3_spm
    for i in range(options.num_l2caches):
        llc=getattr(ruby_system,f'l2_cntrl{i}')
        llc.l3_reduction_initialize=options.cacheflex_l3_initialize
        llc.l3_reduction_finalize=options.cacheflex_l3_finalize
        # 32 FP32 lanes: 16-lane multiply3cy +1 beat, then narrow3cy +1 beat.
        llc.l3_finalize_arithmetic=8
    return result
protocol.define_options=define_options
protocol.create_system=create_system
runpy.run_path(str(Path(__file__).with_name('l3_reduction_pipeline_se.py')),run_name='__m5_main__')
