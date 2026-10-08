"""Local one-credit direct reduction transport; existing L3 model unchanged."""
from pathlib import Path
import runpy
from m5.util import addToPath
addToPath('../..')
from ruby import MESI_Three_Level_SPM as protocol
old_options=protocol.define_options
old_create=protocol.create_system
def define_options(parser):
    old_options(parser)
    parser.add_argument('--cacheflex-direct-reduction',action='store_true')
    parser.add_argument('--cacheflex-direct-bytes',type=int,default=16)
    parser.add_argument('--cacheflex-direct-latency',type=int,default=2)
def create_system(options,full_system,system,dma_ports,bootmem,ruby_system,cpus):
    result=old_create(options,full_system,system,dma_ports,bootmem,ruby_system,cpus)
    if options.cacheflex_direct_reduction:
        assert options.network=='garnet' and options.num_l2caches==options.num_cpus
        assert options.cacheflex_direct_bytes in (8,16,32) and options.cacheflex_direct_latency>=1
    ruby_system.network.cacheflex_direct_reduction=options.cacheflex_direct_reduction
    ruby_system.network.cacheflex_direct_bytes=options.cacheflex_direct_bytes
    ruby_system.network.cacheflex_direct_latency=options.cacheflex_direct_latency
    return result
protocol.define_options=define_options
protocol.create_system=create_system
runpy.run_path(str(Path(__file__).with_name('l3_lifecycle_se.py')),run_name='__m5_main__')
