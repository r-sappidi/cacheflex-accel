"""Finite MICRO-derived resources, with distinct recall and pure-SPM lanes.

Copied as bounded_se.py next to the frozen micro_timing_se.py and se.py.
The hardware queue/TBE sizes are explicit design assumptions, not silicon fits.
"""
from pathlib import Path
import runpy
from m5.util import addToPath
from m5.objects import MicroResourcePool, MessageBuffer
addToPath('../..')
from ruby import MESI_Three_Level_SPM as protocol

original_options=protocol.define_options
original_create=protocol.create_system

def define_options(parser):
    original_options(parser)
    parser.add_argument('--cacheflex-micro-resources',action='store_true')
    parser.add_argument('--cacheflex-micro-small',action='store_true')
    parser.add_argument('--cacheflex-finite-response-channel',action='store_true')
    parser.add_argument('--cacheflex-finite-resources',action='store_true')

def create_system(options,full_system,system,dma_ports,bootmem,ruby_system,cpus):
    result=original_create(options,full_system,system,dma_ports,bootmem,ruby_system,cpus)
    finite=options.cacheflex_finite_resources
    small=options.cacheflex_micro_small
    def pool(n):return MicroResourcePool(capacity=(2 if small else n) if options.cacheflex_micro_resources else 0)
    for i,cpu in enumerate(cpus):
        l0=getattr(ruby_system,f'l0_cntrl{i}');l1=getattr(ruby_system,f'l1_cntrl{i}')
        l0.dataMissCredits=pool(8);l0.instMissCredits=pool(8);l0.writeCredits=pool(8)
        l1.missCredits=pool(20);l1.writeCredits=pool(8);l1.l0WriteCredits=l0.writeCredits
        l0.downstreamMissCredits=l1.missCredits;l0.downstreamWriteCredits=l1.writeCredits
        l0.microSpmQueue=MessageBuffer(ordered=False)
        l0.microSpmToL1=l1.spmBufferFromL0
        l1.returnBufferFromL0=MessageBuffer(ordered=True,clamp_fifo_arrival=True)
        l0.replyToL1=l1.returnBufferFromL0 if finite else l0.bufferToL1
        if options.cacheflex_micro_resources:
            l0.sequencer.cacheflex_spm_ingress=l0.microSpmQueue
            l0.sequencer.cacheflex_micro_data_misses=l0.dataMissCredits
            l0.sequencer.cacheflex_micro_inst_misses=l0.instMissCredits
            l0.sequencer.cacheflex_micro_writebacks=l0.writeCredits
            l0.sequencer.cacheflex_micro_targets=16
        if finite:
            l0.sequencer.cacheflex_finite_ingress=True
            l0.sequencer.cacheflex_max_line_targets=4 if small else 16
            cpu.cacheFlexBoundedTracking=True
    ruby_system.microLlcMissCredits=pool(32);ruby_system.microLlcWriteCredits=pool(16)
    for i in range(options.num_l2caches):
        llc=getattr(ruby_system,f'l2_cntrl{i}')
        llc.missCredits=ruby_system.microLlcMissCredits
        llc.writeCredits=ruby_system.microLlcWriteCredits
    if finite:
        ruby_system.network.cacheflex_request_vc_partition=True
        for obj in ruby_system.descendants():
            if isinstance(obj,MessageBuffer):obj.buffer_size=4 if small else 16
            if 'number_of_TBEs' in obj._params:
                name=obj.path()
                obj.number_of_TBEs=8 if small else (16 if '.l0_cntrl' in name else 32 if '.l1_cntrl' in name else 64)
        for i in range(len(cpus)):
            l0=getattr(ruby_system,f'l0_cntrl{i}');l1=getattr(ruby_system,f'l1_cntrl{i}')
            # 32 local ingress slots: 12 demand/bridge +4 recall/eviction +16
            # pure SPM. Return ordering cannot be blocked by either request FIFO.
            l0.bufferToL1.buffer_size=2 if small else 12
            l1.returnBufferFromL0.buffer_size=2 if small else 4
    return result

protocol.define_options=define_options
protocol.create_system=create_system
runpy.run_path(str(Path(__file__).with_name('se.py')),run_name='__m5_main__')
