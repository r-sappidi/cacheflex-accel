"""Bounded bank-local scale/add extension on the frozen shared-L3 configuration."""
from pathlib import Path
import runpy
from m5.util import addToPath
from m5.objects import MessageBuffer
addToPath('../..')
from ruby import MESI_Three_Level_SPM as protocol
old_options=protocol.define_options
old_create=protocol.create_system
def define_options(parser):
    old_options(parser)
    parser.add_argument('--cacheflex-l3-reduction',action='store_true')
    parser.add_argument('--cacheflex-l3-reduction-sync',action='store_true')
    parser.add_argument('--cacheflex-l3-reduction-depth',type=int,default=4)
    parser.add_argument('--cacheflex-l3-reduction-arithmetic',type=int,default=9)
def create_system(options,full_system,system,dma_ports,bootmem,ruby_system,cpus):
    result=old_create(options,full_system,system,dma_ports,bootmem,ruby_system,cpus)
    assert options.cacheflex_l3_reduction_depth>0
    assert options.cacheflex_l3_reduction_arithmetic>0
    if options.cacheflex_l3_reduction:
        assert options.cacheflex_l3_spm and options.cacheflex_l3_service_port
        for i,cpu in enumerate(cpus):
            seq=getattr(ruby_system,f'l0_cntrl{i}').sequencer
            seq.cacheflex_reduction=True
            seq.cacheflex_l3_reduction=True
            seq.cacheflex_reduction_async=not options.cacheflex_l3_reduction_sync
            seq.cacheflex_reduction_cache=getattr(ruby_system,f'l1_cntrl{i}').cache
            seq.cacheflex_reduction_source_entries=options.cacheflex_l3_reduction_depth
            seq.cacheflex_reduction_sink_entries=options.cacheflex_l3_reduction_depth
            cpu.cacheFlexSpmStoreForwarding=False
    for i in range(options.num_l2caches):
        llc=getattr(ruby_system,f'l2_cntrl{i}')
        llc.l3_reduction_enabled=options.cacheflex_l3_reduction
        llc.l3_reduction_entries=options.cacheflex_l3_reduction_depth
        llc.l3_reduction_arithmetic=options.cacheflex_l3_reduction_arithmetic
        llc.l3ReductionQueue=MessageBuffer(ordered=True,buffer_size=options.cacheflex_l3_reduction_depth)
    return result
protocol.define_options=define_options
protocol.create_system=create_system
runpy.run_path(str(Path(__file__).with_name('l3_spm_se.py')),run_name='__m5_main__')
