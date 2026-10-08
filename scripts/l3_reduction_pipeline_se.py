"""Orthogonal bounded bank-pipeline and typed FP16-contribution experiments."""
from pathlib import Path
import runpy
from m5.util import addToPath
addToPath('../..')
from ruby import MESI_Three_Level_SPM as protocol
old_options=protocol.define_options
old_create=protocol.create_system

def define_options(parser):
    old_options(parser)
    parser.add_argument('--cacheflex-l3-reduction-pipeline',action='store_true')
    parser.add_argument('--cacheflex-l3-reduction-fp16',action='store_true')
    parser.add_argument('--cacheflex-l3-reduction-interval',type=int,default=4)
    parser.add_argument('--cacheflex-l3-reduction-convert',type=int,default=6)

def create_system(options,full_system,system,dma_ports,bootmem,ruby_system,cpus):
    result=old_create(options,full_system,system,dma_ports,bootmem,ruby_system,cpus)
    # Fixed 16-lane, fully pipelined FP32 multiply/add and conversion units.
    # 64 lanes/16 per cycle => initiation interval >=4. Conversion stage:
    # 3-cycle latency plus 3 additional vector beats =>6 before arithmetic.
    assert options.cacheflex_l3_reduction_interval>=4
    assert options.cacheflex_l3_reduction_convert>=6
    for i in range(options.num_l2caches):
        llc=getattr(ruby_system,f'l2_cntrl{i}')
        llc.l3_reduction_pipeline=options.cacheflex_l3_reduction_pipeline
        llc.l3_reduction_interval=options.cacheflex_l3_reduction_interval
        llc.l3_reduction_fp16=options.cacheflex_l3_reduction_fp16
        llc.l3_reduction_convert=options.cacheflex_l3_reduction_convert
    return result

protocol.define_options=define_options
protocol.create_system=create_system
runpy.run_path(str(Path(__file__).with_name('l3_reduction_retry_se.py')),run_name='__m5_main__')
