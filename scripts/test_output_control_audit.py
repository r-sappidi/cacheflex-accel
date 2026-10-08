import unittest
from resume_output_resident import committed_work

class CommittedWorkTests(unittest.TestCase):
    def test_timing_changes_do_not_change_architectural_work(self):
        a={'simInsts':100,'simOps':120,'simTicks':200,
           'system.cpu0.commit.committedInstType_0::SimdFloatMultAcc':40,
           'system.cpu0.commit.commitSquashedInsts':8}
        b=dict(a,simTicks=201)
        b['system.cpu0.commit.commitSquashedInsts']=9
        self.assertEqual(committed_work(a),committed_work(b))
    def test_changed_work_is_detected(self):
        a={'simInsts':100,'simOps':120,
           'system.cpu0.commit.committedInstType_0::SimdFloatMultAcc':40}
        for key in a:
            b=dict(a);b[key]+=1
            self.assertNotEqual(committed_work(a),committed_work(b))

if __name__=='__main__':unittest.main()
