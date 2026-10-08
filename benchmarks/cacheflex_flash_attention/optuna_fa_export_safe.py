"""Resume unchanged Optuna search using verified post-ROI line exports.

Original simulator, builder, Optuna module, trials and failed bytes preserved.
Only dormant code/export stub in generated ELF copies are changed.
"""
from pathlib import Path
import hashlib,json,sys,time,subprocess
import optuna
import optuna_fa as original
from fa_export_patch import patch,verify_dump

Base=original.Evaluator

class ExportSafeEvaluator(Base):
    def guest(self,c,rich):return patch(super().guest(c,rich))
    def run(self,c,br,bc,rich=True,rep=0):
        t=1024 if rich else max(256,2*bc)
        ident=original.key(dict(config=c,br=br,bc=bc,t=t,rich=rich,rep=rep))
        p=self.out/'runs'/ident;accepted=p/'accepted.json'
        needs_archive=False
        if accepted.exists():
            try:verify_dump(json.loads(accepted.read_text()))
            except AssertionError:needs_archive=True
        elif (p/'run.log').exists():needs_archive=True
        if needs_archive:
            archive=self.out/'raw_export_diagnostics';archive.mkdir(exist_ok=True)
            target=archive/ident;assert not target.exists(),('already archived; inspect',target)
            p.rename(target)
        return verify_dump(super().run(c,br,bc,rich,rep))

def main():
    out=Path(sys.argv[1]).resolve()
    repair=out.parent/'optuna_export_repair_20260913'
    while not (repair/'summary.json').exists():
        state=json.loads((repair/'status.json').read_text())
        if state['stage']=='failed':raise RuntimeError(state.get('error'))
        time.sleep(30)
    summary=json.loads((repair/'summary.json').read_text())
    assert summary['stage']=='complete' and len(summary['completed'])==4
    subprocess.run([sys.executable,str(Path(__file__).with_name('check_fa_export_controls.py'))],check=True)
    files=[Path(__file__),Path(__file__).with_name('fa_export_patch.py'),Path(__file__).with_name('fa_export_hook.S'),Path(__file__).with_name('check_fa_export_controls.py')]
    manifest=out/'export_repair_hashes.json'
    current={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if manifest.exists():assert json.loads(manifest.read_text())==current
    else:manifest.write_text(json.dumps(current,indent=2))
    study=optuna.load_study(study_name='cache',storage='sqlite:///'+str(out/'studies.sqlite3'))
    if not study.user_attrs.get('export_repair_requeued'):
        failed=study.trials[8];assert failed.state==optuna.trial.TrialState.FAIL
        study.enqueue_trial(failed.params)
        study.set_user_attr('export_repair_requeued',True)
        study.set_user_attr('export_repair','Post-ROI guest timing-load bounce buffer; strict dump/guest hash matching')
    original.Evaluator=ExportSafeEvaluator
    original.main()

if __name__=='__main__':main()
