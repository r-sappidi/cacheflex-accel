"""Diagnostic oracle staging; no changes unless explicitly enabled.

Installed next to the frozen micro_timing_se.py and bounded_se.py adapters.
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
    parser.add_argument('--cacheflex-ideal-staging', action='store_true',
                        help='Oracle diagnostic, NOT realizable hardware')

def create_system(options, full_system, system, dma_ports, bootmem, ruby_system, cpus):
    result = original_create(options, full_system, system, dma_ports, bootmem, ruby_system, cpus)
    if options.cacheflex_ideal_staging:
        assert not full_system, 'Oracle only validated for explicit SE diagnostics'
        for i in range(len(cpus)):
            l0 = getattr(ruby_system, f'l0_cntrl{i}')
            l1 = getattr(ruby_system, f'l1_cntrl{i}')
            l0.sequencer.cacheflex_ideal_staging_cache = l1.cache
            l0.sequencer.cacheflex_ideal_staging_controller = l1
    return result

protocol.define_options = define_options
protocol.create_system = create_system
runpy.run_path(str(Path(__file__).with_name('micro_timing_se.py')), run_name='__m5_main__')
