"""Real frozen-simulator controls for the post-ROI export repair."""
from _paths import FA_DIR
from pathlib import Path
import concurrent.futures as cf,json,shutil
from optuna_fa import Evaluator,DEFAULT,key,OLD
from fa_export_patch import patch,verify_dump

def main():
    root=OLD.parent;old=root/'optuna_20260912';out=root/'optuna_export_repair_20260913'
    out.mkdir();shutil.copy2(old/'gem5.opt',out/'gem5.opt')
    evaluator=Evaluator(out)
    for p in (old/'references').glob('*.npy'):shutil.copy2(p,out/'references'/p.name)
    for bc in (128,):evaluator.getref(bc)
    original_guest=evaluator.guest
    def guest(c,rich):
        p=old/'builds'/key(dict(c,rich=int(rich)))/'guest'
        return patch(p if p.exists() else original_guest(c,rich))
    evaluator.guest=guest
    bad=dict(DEFAULT,group=2,unroll=4,rows=8,pipeline=1,fusion=1)
    cases=[('failed_cache',bad,32),('matched_spm',dict(bad,placement='k'),32),
           ('cache_control',dict(DEFAULT),64),('spm_control',dict(DEFAULT,placement='k'),64)]
    state={'stage':'running','completed':[]}
    def save():(out/'status.json').write_text(json.dumps(state,indent=2))
    save()
    try:
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(evaluator.run,c,br,128,True):name for name,c,br in cases}
            for future in cf.as_completed(futures):
                row=verify_dump(future.result());row['case']=futures[future]
                state['completed'].append(row);save();print(row['case'],row['roi_us'],'PASS',flush=True)
        state['stage']='complete';save();(out/'summary.json').write_text(json.dumps(state,indent=2))
    except BaseException as e:state['stage']='failed';state['error']=repr(e);save();raise

if __name__=='__main__':main()
