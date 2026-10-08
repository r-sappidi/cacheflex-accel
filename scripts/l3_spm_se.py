"""Opt-in shared LLC slots, on the frozen finite-resource MICRO translation."""
from pathlib import Path
import runpy
from m5.util import addToPath
addToPath('../..')
from ruby import MESI_Three_Level_SPM as protocol
old_options=protocol.define_options
old_create=protocol.create_system
def define_options(parser):
    old_options(parser)
    parser.add_argument('--cacheflex-l3-spm',action='store_true')
    parser.add_argument('--cacheflex-l3-service-port',action='store_true')
    parser.add_argument('--cacheflex-l3-ways',type=int,default=8)
def create_system(options,full_system,system,dma_ports,bootmem,ruby_system,cpus):
    result=old_create(options,full_system,system,dma_ports,bootmem,ruby_system,cpus)
    if options.cacheflex_l3_spm:
        assert options.num_l2caches<=8 and options.num_clusters==1
        assert 1<=options.cacheflex_l3_ways<=min(8,options.l2_assoc-2)
        for i in range(len(cpus)):
            getattr(ruby_system,f'l1_cntrl{i}').l3_spm_enabled=True
    for i in range(options.num_l2caches):
        llc=getattr(ruby_system,f'l2_cntrl{i}')
        llc.l3_spm_enabled=options.cacheflex_l3_spm
        llc.l3_shared_service_port=options.cacheflex_l3_service_port
        # Match the current MICRO-translated LLC service delay initially.
        # Do not assume a tag-bypass latency benefit without calibration.
        llc.l3_spm_latency=35 if options.cacheflex_micro_timing else llc.l2_response_latency
        if options.cacheflex_l3_spm:
            llc.L2cache.num_spm_ways=options.cacheflex_l3_ways
    return result
protocol.define_options=define_options
protocol.create_system=create_system
runpy.run_path(str(Path(__file__).with_name('micro_timing_se.py')),run_name='__m5_main__')
