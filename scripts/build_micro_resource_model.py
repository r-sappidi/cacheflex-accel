"""Isolated build for the source-backed MICRO/Ruby resource translation."""
import build_hardware_bounds as build

build.OUT = build.ROOT / 'results/cacheflex_flash_attention/micro_resources_20260921'
build.ADAPTER = build.ROOT / 'scripts/micro_resource_se.py'
build.FILES += ['cpu/o3/lsq.cc', 'mem/slicc/symbols/StateMachine.py', 'mem/ruby/SConscript', 'mem/ruby/structures/SConscript',
    'mem/ruby/structures/MicroResourcePool.hh', 'mem/ruby/structures/MicroResourcePool.cc',
    'mem/ruby/structures/MicroResourcePool.py', 'mem/ruby/protocol/RubySlicc_Types.sm',
    'mem/ruby/protocol/MESI_Three_Level_SPM-L2cache.sm']
if __name__ == '__main__':
    build.main()
