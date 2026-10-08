import unittest
import configparser
from analyze_causal import decompose
from run_causal import audit, config_key, parameters

class FactorialTest(unittest.TestCase):
    def test_additive(self):
        values = {m:10-2*bool(m&1)-3*bool(m&2) for m in range(4)}
        r = decompose(values, ('a','b'))
        self.assertEqual(r['contributions'], {'a':2,'b':3})
        self.assertEqual(r['residual'], 5)
        self.assertEqual(r['pair_synergy']['a+b'], 0)

    def test_interaction_and_negative(self):
        # A hurts by 1 alone, B saves 2 alone; jointly save another 6.
        r = decompose({0:10,1:11,2:8,3:3}, ('a','b'))
        self.assertEqual(r['standalone']['a'], -1)
        self.assertEqual(r['pair_synergy']['a+b'], 6)
        self.assertEqual(r['contributions'], {'a':2,'b':5})
        self.assertEqual(sum(r['contributions'].values())+r['residual'], 10)

    def test_incomplete(self):
        with self.assertRaises(AssertionError):
            decompose({0:10}, ('a',))

    def test_config_whitelist(self):
        base=configparser.ConfigParser()
        base['system.cpu0']={'clock':'400'}
        for key in parameters(15):
            section,field=config_key(key)
            if section not in base:base[section]={}
            base[section][field]='0'
        actual=configparser.ConfigParser()
        actual.read_dict({s:dict(base[s]) for s in base.sections()})
        for key,value in parameters(15).items():
            section,field=config_key(key)
            actual[section][field]=value.lower()
        audit(base,actual,15)
        actual['system.cpu0']['clock']='200'
        with self.assertRaises(AssertionError):audit(base,actual,15)

if __name__ == '__main__':
    unittest.main()
