"""Host-only sampler/constraint/persistence checks; no gem5 launches."""
import tempfile, unittest
from pathlib import Path
import optuna
from optuna_fa import suggest, eligible, key

class TunerTests(unittest.TestCase):
    def test_conditional_space(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        for arch in ('cache','spm'):
            study=optuna.create_study(sampler=optuna.samplers.TPESampler(
                seed=137,multivariate=True,group=True,constant_liar=True,n_startup_trials=5))
            valid=0
            for i in range(100):
                trial=study.ask();c,br,bc=suggest(trial,arch)
                self.assertFalse(c['fusion'] and c['interleave'])
                self.assertLessEqual(c['group'],1024//br)
                if arch=='cache':self.assertEqual(c['placement'],'cache');self.assertEqual(c['offset'],0)
                if c['pinned']:self.assertEqual(c['placement'],'k');self.assertEqual(c['overlap'],1)
                if not eligible(c,1024,br,bc):
                    study.tell(trial,state=optuna.trial.TrialState.PRUNED)
                else:
                    valid+=1;study.tell(trial,float(br+bc+c['group']))
            self.assertGreater(valid,10)
    def test_canonical_group(self):
        base=dict(Br=128,Bc=128,qk_kernel='native',unroll=1,phase_mode='plain')
        a=suggest(optuna.trial.FixedTrial(dict(base,group=8)),'cache')
        b=suggest(optuna.trial.FixedTrial(dict(base,group=64)),'cache')
        self.assertEqual(a,b)
    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            storage='sqlite:///'+str(Path(d)/'test.sqlite3')
            s=optuna.create_study(study_name='cache',storage=storage)
            t=s.ask();c,br,bc=suggest(t,'cache');ident=key(dict(c=c,br=br,bc=bc))
            t.set_user_attr('candidate_key',ident);s.tell(t,42.0)
            loaded=optuna.load_study(study_name='cache',storage=storage)
            self.assertEqual(loaded.best_value,42.0)
            self.assertEqual(loaded.best_trial.user_attrs['candidate_key'],ident)

if __name__=='__main__':unittest.main()
