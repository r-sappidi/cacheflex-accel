"""SE config adapter for the finite-resource protocol, copied beside se.py.

Only the new recall-response connection is configured here. Existing config
trees used by active experiments remain untouched.
"""
from pathlib import Path
import runpy
from m5.util import addToPath
addToPath('../..')
from ruby import MESI_Three_Level_SPM as protocol

original_options = protocol.define_options
original_create = protocol.create_system

def define_options(parser):
    original_options(parser)
    parser.add_argument('--cacheflex-finite-response-channel', action='store_true')

def create_system(options, full_system, system, dma_ports, bootmem, ruby_system, cpus):
    result = original_create(options, full_system, system, dma_ports, bootmem, ruby_system, cpus)
    for i in range(len(cpus)):
        l0 = getattr(ruby_system, f'l0_cntrl{i}')
        l1 = getattr(ruby_system, f'l1_cntrl{i}')
        # Legacy route shares exactly the original FIFO. Finite mode uses the
        # formerly unproduced direct input as an independent recall return lane.
        l0.replyToL1 = l1.spmBufferFromL0 if options.cacheflex_finite_response_channel else l0.bufferToL1
    return result

protocol.define_options = define_options
protocol.create_system = create_system
runpy.run_path(str(Path(__file__).with_name('se.py')), run_name='__m5_main__')
