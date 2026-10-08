"""Opt-in MICRO timing translation; run beside frozen bounded_se.py.

Cache/service clocks follow MICRO's CPU clock. Garnet topology, bandwidth,
router/NI clocks and all resource/credit policies remain unchanged. This does
not equate the multicore mesh to Classic's single-core crossbars.
"""
from pathlib import Path
import runpy
from m5.util import addToPath
addToPath('../..')
from ruby import MESI_Three_Level_SPM as protocol

original_options=protocol.define_options
original_create=protocol.create_system

def define_options(parser):
    original_options(parser)
    parser.add_argument('--cacheflex-micro-timing',action='store_true')

def create_system(options,full_system,system,dma_ports,bootmem,ruby_system,cpus):
    result=original_create(options,full_system,system,dma_ports,bootmem,ruby_system,cpus)
    if not options.cacheflex_micro_timing:return result
    for i in range(len(cpus)):
        l0=getattr(ruby_system,f'l0_cntrl{i}')
        l1=getattr(ruby_system,f'l1_cntrl{i}')
        l0.clk_domain=system.cpu_clk_domain
        l1.clk_domain=system.cpu_clk_domain
        # Distinct request/data service delays feed the same ordered local
        # link. Preserve its FIFO departure order instead of allowing a
        # shorter-latency later message to overtake (or asserting on it).
        l0.bufferToL1.clamp_fifo_arrival=True
        l0.bufferFromL1.clamp_fifo_arrival=True
        # The CPU->Ruby adapter already contributes one cycle relative to
        # Classic. One additional ingress cycle realizes MICRO's L1 tag/data2
        # at the CPU boundary (verified by same-ELF dependent-load probes).
        # Local bus frontend+forward1+1. L2 hit8,
        # local return1, L1 refill response2 (do not charge another lookup).
        l0.mandatory_queue_latency=1
        l0.request_latency=2
        l1.l1_response_latency=11
        l1.l1_request_latency=8
        # Direct SPM latency8 total: CPU adapter1 + ingress1 + forward1 + service5.
        # These adapter stages are included, not added to an8-cycle target.
        l0.spm_load_request_latency=1
        l0.spm_request_latency=1
        l1.spm_access_response_latency=5
        # MICRO L1 bridge forward: tag2+writebuffer forward2+bus2.
        l0.spm_copy_request_latency=4
        l1.spm_response_latency=8
    for i in range(options.num_l2caches):
        llc=getattr(ruby_system,f'l2_cntrl{i}')
        llc.clk_domain=system.cpu_clk_domain
        llc.l2_request_latency=35
        llc.l2_response_latency=35
    return result

protocol.define_options=define_options
protocol.create_system=create_system
runpy.run_path(str(Path(__file__).with_name('bounded_se.py')),run_name='__m5_main__')
