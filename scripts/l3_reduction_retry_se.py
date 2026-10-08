"""Reduction-only retry isolation; no changes to queue/service/issue capacities."""
from pathlib import Path
import runpy
from m5.util import addToPath
from m5.objects import RubySequencer
addToPath('../..')
from ruby import MESI_Three_Level_SPM as protocol
old_options=protocol.define_options
old_create=protocol.create_system
old_connect=RubySequencer.connectCpuPorts

def define_options(parser):
    old_options(parser)
    parser.add_argument('--cacheflex-reduction-retry-isolation',action='store_true')

def create_system(options,full_system,system,dma_ports,bootmem,ruby_system,cpus):
    result=old_create(options,full_system,system,dma_ports,bootmem,ruby_system,cpus)
    for cpu in cpus:
        cpu.cacheFlexIndependentReduction=options.cacheflex_reduction_retry_isolation
    return result

def connect(self,cpu):
    old_connect(self,cpu)
    if bool(cpu.cacheFlexIndependentReduction):
        # Append, preserving all original interface indices. This is another
        # handshake into the SAME sequencer and bounded reduction source queue.
        cpu.reduction_port=self.in_ports

protocol.define_options=define_options
protocol.create_system=create_system
RubySequencer.connectCpuPorts=connect
runpy.run_path(str(Path(__file__).with_name('l3_reduction_se.py')),run_name='__m5_main__')
