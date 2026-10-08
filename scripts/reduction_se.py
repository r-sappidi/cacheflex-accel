"""Layer only the experimental reduction device onto frozen finite MICRO config."""
from pathlib import Path
import runpy
from m5.util import addToPath
addToPath('../..')
from ruby import MESI_Three_Level_SPM as protocol
original_options=protocol.define_options
original_create=protocol.create_system
def define_options(parser):
    original_options(parser)
    parser.add_argument('--cacheflex-reduction',action='store_true')
    parser.add_argument('--cacheflex-reduction-sync',action='store_true')
def create_system(options,full_system,system,dma_ports,bootmem,ruby_system,cpus):
    result=original_create(options,full_system,system,dma_ports,bootmem,ruby_system,cpus)
    if options.cacheflex_reduction:
        for i,cpu in enumerate(cpus):
            seq=getattr(ruby_system,f'l0_cntrl{i}').sequencer
            seq.cacheflex_reduction=True
            seq.cacheflex_reduction_async=not options.cacheflex_reduction_sync
            seq.cacheflex_reduction_cache=getattr(ruby_system,f'l1_cntrl{i}').cache
            # A command's payload is a contribution, NOT the new slot value.
            cpu.cacheFlexSpmStoreForwarding=False
    return result
protocol.define_options=define_options
protocol.create_system=create_system
runpy.run_path(str(Path(__file__).with_name('micro_timing_se.py')),run_name='__m5_main__')
