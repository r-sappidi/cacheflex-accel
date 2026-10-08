import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import build_micro_head_scaling as builder
import run_micro_head_scaling as runner


class HeadScalingTests(unittest.TestCase):
    def test_whole_kv_group_ownership(self):
        for cores in (1,4,8):
            assigned=[list(range(i*(32//cores),(i+1)*(32//cores))) for i in range(cores)]
            self.assertEqual(sum(assigned,[]),list(range(32)))
            owners={h//4:i for i,heads in enumerate(assigned) for h in heads}
            for i,heads in enumerate(assigned):
                self.assertTrue(all(owners[h//4]==i for h in heads))
                self.assertEqual(heads[0]%4,0)

    def test_inner_head_body_unchanged(self):
        for arch in ('cache','spm'):
            source=(builder.OLD/f'{arch}_vl16/driver.cpp').read_text()
            generated,audit=builder.transform(source,arch)
            before=source.index('        for (int hq = 0; hq < H_q; ++hq)')
            marker='\n        double us = us_since(t0);' if arch=='spm' else '        ROI_END();'
            body=source[before:source.index(marker,before)]
            changed=body.replace('for (int hq = 0; hq < H_q; ++hq)',
                                 'for (int hq = head_begin; hq < head_end; ++hq)')
            self.assertIn(changed,generated)
            self.assertEqual(generated.count('m5_reset_stats(0,0)'),1)
            self.assertIn('H_kv%cores',generated)
            self.assertIn('pthread_barrier_wait(&barrier);',generated)
            self.assertTrue(audit['original_head_body_sha256'])

    def test_command_core_filter_and_fixed_memory(self):
        for cores in (1,4,8):
            for arch in ('cache','spm'):
                cmd,args=runner.make_command(arch,'parallel',cores,4096,32,8,Path('/tmp/head-scaling-test'))
                self.assertIn(f'--num-cpus={cores}',cmd)
                for arg in ('--num-dirs=2','--num-l2caches=8','--mem-type=DDR4_2400_8x8',
                            '--mem-channels=2','--l2_size=512KiB','--topology=MicroHeadMesh'):
                    self.assertIn(arg,cmd)
                self.assertEqual(args[1:4],['4096','32','8'])
                self.assertEqual(args[-1],str(cores))
                self.assertNotIn(f'system.cpu[{cores}].spmLoadPorts=1',cmd)

    def test_fixed_mesh_mapping(self):
        module=types.ModuleType('topologies.BaseTopology');module.SimpleTopology=object
        with patch.dict(sys.modules,{'topologies.BaseTopology':module}):
            spec=importlib.util.spec_from_file_location('test_mesh',HERE/'MicroHeadMesh.py')
            mesh=importlib.util.module_from_spec(spec);spec.loader.exec_module(mesh)
        node=lambda **kw:types.SimpleNamespace(**kw)
        for cores in (1,4,8):
            nodes=[node(type=t) for t,count in [('L0Cache_Controller',cores),('L1Cache_Controller',cores),
                   ('L2Cache_Controller',8),('Directory_Controller',2)] for _ in range(count)]
            options=node(num_cpus=cores,num_l2caches=8,num_dirs=2,router_latency=1,link_latency=1)
            network=node()
            mesh.MicroHeadMesh(nodes).makeTopology(options,network,node,node,node)
            self.assertEqual(len(network.routers),8)
            self.assertEqual(len(network.int_links),20)
            dirs=[l.int_node.router_id for l in network.ext_links if l.ext_node.type=='Directory_Controller']
            self.assertEqual(dirs,[0,4])


if __name__=='__main__':unittest.main()
