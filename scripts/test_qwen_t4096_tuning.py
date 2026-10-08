"""Admission safety and unchanged scientific checks for the L3-11 launch."""
import ast
import json
import unittest
from qwen_t4096_tuning import BASE, PRIOR, JOURNAL, GIB, read, slots_for, run_parts, fn


class T4096Launch(unittest.TestCase):
    def test_budget_covers_future_old_jobs(self):
        budget=read(JOURNAL/'budget.json')
        for pending in range(50):
            for refs in range(6):
                n,reserved,old=slots_for(pending,refs,budget)
                self.assertLessEqual(n,4)
                self.assertGreaterEqual(n,0)
                self.assertEqual(old,min(budget['max_worker_slots'],pending+refs))
                if n:
                    self.assertLessEqual(reserved+n*8*GIB,budget['ram_limit'])
        self.assertEqual(slots_for(23,0,budget)[0],0)
        self.assertEqual(slots_for(16,0,budget)[0],4)

    def test_five_missing_cases_close_equal_geometry_set(self):
        plan=read(BASE/'selection.json')
        self.assertEqual(len(plan['selected']),5)
        for arm in ('cache','private','l3'):
            previous={(r['case']['r'],r['case']['bc']) for r in plan['reused_t4096'] if r['case']['arm']==arm}
            added={(r['r'],r['bc']) for r in plan['selected'] if r['arm']==arm}
            self.assertFalse(previous & added)
            self.assertEqual(previous | added,{(128,256),(256,256),(256,512)})

    def test_original_acceptance_and_commands_are_identical(self):
        source=ast.parse((PRIOR/'selected_runner.py').read_text())
        derived=ast.parse((BASE/'selected_runner.py').read_text())
        for old,new in zip(run_parts(source),run_parts(derived)):
            self.assertEqual(ast.dump(ast.Module(body=old,type_ignores=[])),ast.dump(ast.Module(body=new,type_ignores=[])))
        for name in ('work','get_reference'):
            self.assertEqual(ast.dump(fn(source,name)),ast.dump(fn(derived,name)))


if __name__=='__main__':
    unittest.main()
