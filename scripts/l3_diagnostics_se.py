"""Opt-in reduction timing sensitivities; zero overrides preserve old timings."""
from pathlib import Path
import runpy
from m5.util import addToPath
addToPath('../..')
from ruby import MESI_Three_Level_SPM as protocol
old_options = protocol.define_options
old_create = protocol.create_system

def define_options(parser):
    old_options(parser)
    parser.add_argument('--diag-reduction-read', type=int, default=0)
    parser.add_argument('--diag-reduction-reply', type=int, default=0)

def create_system(options, full_system, system, dma_ports, bootmem, ruby_system, cpus):
    result = old_create(options, full_system, system, dma_ports, bootmem, ruby_system, cpus)
    assert options.diag_reduction_read >= 0 and options.diag_reduction_reply >= 0
    for i in range(options.num_l2caches):
        llc = getattr(ruby_system, f'l2_cntrl{i}')
        llc.l3_reduction_read_delay = options.diag_reduction_read
        llc.l3_reduction_reply_delay = options.diag_reduction_reply
    return result

protocol.define_options = define_options
protocol.create_system = create_system
runpy.run_path(str(Path(__file__).with_name('l3_direct_reduction_se.py')), run_name='__m5_main__')
