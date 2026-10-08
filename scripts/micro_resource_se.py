"""Opt-in MICRO-derived credit classes; loaded beside a frozen se.py.

Ruby coherence and Garnet remain Ruby. No assertion of cycle equivalence with
Classic. Disabled profile is used for legacy timing equivalence gates.
"""
from pathlib import Path
import runpy
from m5.util import addToPath
from m5.objects import MicroResourcePool, MessageBuffer
addToPath('../..')
from ruby import MESI_Three_Level_SPM as protocol

original_options = protocol.define_options
original_create = protocol.create_system

def define_options(parser):
    original_options(parser)
    parser.add_argument('--cacheflex-micro-resources', action='store_true')
    parser.add_argument('--cacheflex-micro-small', action='store_true')
    parser.add_argument('--cacheflex-finite-response-channel', action='store_true')

def create_system(options, full_system, system, dma_ports, bootmem, ruby_system, cpus):
    result = original_create(options, full_system, system, dma_ports, bootmem, ruby_system, cpus)
    enabled = options.cacheflex_micro_resources
    def pool(n):
        return MicroResourcePool(capacity=(2 if options.cacheflex_micro_small else n) if enabled else 0)
    for i in range(len(cpus)):
        l0 = getattr(ruby_system, f'l0_cntrl{i}')
        l1 = getattr(ruby_system, f'l1_cntrl{i}')
        l0.dataMissCredits = pool(8)
        l0.instMissCredits = pool(8)
        l0.writeCredits = pool(8)
        l1.missCredits = pool(20)
        l1.writeCredits = pool(8)
        l1.l0WriteCredits = l0.writeCredits
        l0.downstreamMissCredits = l1.missCredits
        l0.downstreamWriteCredits = l1.writeCredits
        l0.microSpmQueue = MessageBuffer(ordered=False)
        l0.microSpmToL1 = l1.spmBufferFromL0
        # Ordered evictions/recall replies share their original local FIFO;
        # only pure SPM accesses use the separate, otherwise unused input.
        l0.replyToL1 = l0.bufferToL1
        if enabled:
            l0.sequencer.cacheflex_spm_ingress = l0.microSpmQueue
            l0.sequencer.cacheflex_micro_data_misses = l0.dataMissCredits
            l0.sequencer.cacheflex_micro_inst_misses = l0.instMissCredits
            l0.sequencer.cacheflex_micro_writebacks = l0.writeCredits
            l0.sequencer.cacheflex_micro_targets = 16
    ruby_system.microLlcMissCredits = pool(32)
    ruby_system.microLlcWriteCredits = pool(16)
    for i in range(options.num_l2caches):
        llc = getattr(ruby_system, f'l2_cntrl{i}')
        # Preserve MICRO's pooled capacity, rather than imposing an extra
        # static partition across Ruby's address-interleaved LLC banks.
        llc.missCredits = ruby_system.microLlcMissCredits
        llc.writeCredits = ruby_system.microLlcWriteCredits
    return result

protocol.define_options = define_options
protocol.create_system = create_system
runpy.run_path(str(Path(__file__).with_name('se.py')), run_name='__m5_main__')
