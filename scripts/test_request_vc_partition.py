"""Routing-callsite regressions for the fixed-budget request VC partition."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
G=ROOT/'gem5/src/mem/ruby/network/garnet'

class PartitionTests(unittest.TestCase):
    def test_default_off_and_no_extra_storage(self):
        text=(G/'GarnetNetwork.py').read_text()
        self.assertIn('cacheflex_request_vc_partition = Param.Bool(False',text)
        self.assertIn('vcs_per_vnet = Param.UInt32(4',text)
        self.assertIn('buffers_per_ctrl_vc = Param.UInt32(1',text)
    def test_injection_and_every_hop_use_destination(self):
        ni=(G/'NetworkInterface.cc').read_text();out=(G/'OutputUnit.cc').read_text();sa=(G/'SwitchAllocator.cc').read_text()
        self.assertIn('calculateVC(vnet, dest_nodes[ctr])',ni)
        self.assertIn('requestVCAllowed(vnet, delta, m_vc_per_vnet, dest)',ni)
        self.assertEqual(out.count('requestVCAllowed('),2)
        self.assertGreaterEqual(sa.count('get_route().dest_ni'),2)
    def test_ordering_does_not_rejoin_dependency_classes(self):
        for file in ('SwitchAllocator.cc','NetworkInterface.cc'):
            text=(G/file).read_text()
            self.assertIn('sameRequestVCClass(',text)
    def test_profile_opts_in_only_when_finite(self):
        text=(ROOT/'scripts/finite_micro_se.py').read_text()
        self.assertIn('if finite:\n        ruby_system.network.cacheflex_request_vc_partition=True',text)
    def test_instantiated_stress_changes_only_partition_flag(self):
        from fa_headroom_profiles import leaves
        base=ROOT/'results/cacheflex_flash_attention/finite_model_20260926'
        for case in ('stress_cache','stress_spm'):
            a=base/'experiment_v3/runs'/case;b=base/'experiment_v5/runs'/case
            if not (b/'config.json').exists():self.skipTest('new stress configuration not yet available')
            x=dict(leaves(json.loads((a/'config.json').read_text().replace(str(a),'<OUT>'))))
            y=dict(leaves(json.loads((b/'config.json').read_text().replace(str(b),'<OUT>'))))
            delta={k:(x.get(k),y.get(k)) for k in x.keys()|y.keys() if x.get(k)!=y.get(k)}
            self.assertEqual(delta,{'system.ruby.network.cacheflex_request_vc_partition':(None,True)})

if __name__=='__main__':unittest.main()
