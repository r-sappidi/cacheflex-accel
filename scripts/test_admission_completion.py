"""Regression model for synchronous retry before completion-list removal.

Real simulator regression and unchanged-default checks are queued separately.
"""
import unittest

def admits(records, outstanding, incoming, total, fetch_cap, bridge_cap, fixed=True):
    live=[kind for kind,active in records if active or not fixed]
    fetch=live.count('fetch');bridge=live.count('bridge')
    shared=outstanding-(fetch if fetch_cap else 0)-(bridge if bridge_cap else 0)
    if outstanding>=total:return False
    if incoming=='fetch' and fetch_cap:return fetch<fetch_cap
    if incoming=='bridge' and bridge_cap:return bridge<bridge_cap
    return shared<total-fetch_cap-bridge_cap

class CompletionTests(unittest.TestCase):
    def test_last_fetch_retry(self):
        # markRemoved has reduced total to zero; old record remains in list.
        records=[('fetch',False)]
        self.assertFalse(admits(records,0,'fetch',17,1,0,fixed=False))
        self.assertTrue(admits(records,0,'fetch',17,1,0))

    def test_bridge_completion_and_alias_count(self):
        records=[('bridge',False)]+[('bridge',True)]*15
        self.assertFalse(admits(records,15,'bridge',32,0,16,fixed=False))
        self.assertTrue(admits(records,15,'bridge',32,0,16))
        self.assertFalse(admits([('bridge',True)]*16,16,'bridge',32,0,16))

    def test_bounds_and_default_unchanged(self):
        for total,f,b in ((16,0,0),(17,1,0),(16,1,0),(32,0,16),(33,1,16)):
            for nf in range((f or total)+1):
                for nb in range((b or total)+1):
                    for ns in range(total+1):
                        if nf+nb+ns>total:continue
                        records=[('fetch',True)]*nf+[('bridge',True)]*nb+[('other',True)]*ns
                        for kind in ('fetch','bridge','other'):
                            self.assertEqual(admits(records,nf+nb+ns,kind,total,f,b),admits(records,nf+nb+ns,kind,total,f,b,False))
                            if admits(records,nf+nb+ns,kind,total,f,b):
                                self.assertLess(nf+nb+ns,total)
                                if kind=='fetch' and f:self.assertLess(nf,f)
                                elif kind=='bridge' and b:self.assertLess(nb,b)
                                else:self.assertLess(nf+nb+ns-(nf if f else 0)-(nb if b else 0),total-f-b)

if __name__=='__main__':unittest.main()
