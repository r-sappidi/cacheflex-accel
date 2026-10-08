"""Validate export fix on failed staged case, audit history, resume frozen queue.

Keep queue_correct_best.py bytes unchanged (active Optuna hashes that module).
Three asserted source substitutions wrap its existing export boundaries only.
"""
from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess,sys
from fa_export_patch import patch,verify_dump
from fa_independent_reference import check
from validate_attribution import read
from run_causal import audit
import numpy as np
import queue_correct_best as queue

def main():
    out=Path(sys.argv[1]).resolve();repair=out/'export_repair';repair.mkdir(exist_ok=True)
    lease=(repair/'resume.lock').open('w');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (out/'queue.lock').open('r+') as qlock:
        fcntl.flock(qlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**values):
        (repair/'status.json').write_text(json.dumps(dict(stage=stage,pid=os.getpid(),**values),indent=2))
    status('repaired_export_regression')
    failed=out/'runs/ff2554f62fe98221';archive=out/'raw_export_diagnostics'
    if not failed.exists():failed=archive/'ff2554f62fe98221'
    old_output,old_counters,old_cfg=read(failed,16)
    args=json.loads((failed/'command.json').read_text());guest=Path(args[args.index('--cmd')+1])
    test=repair/'failed_case';test.mkdir(exist_ok=True)
    if not (test/'accepted.json').exists():
        args=[f'--outdir={test}' if x.startswith('--outdir=') else x for x in args]
        args[args.index('--cmd')+1]=str(patch(guest))
        (test/'command.json').write_text(json.dumps(args,indent=2))
        with (test/'run.log').open('w') as log:
            subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=7200,
               env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
        output,counters,cfg=read(test,16);audit(old_cfg,cfg,0)
        assert output==old_output
        assert {k:v for k,v in counters.items() if k!='path'}=={k:v for k,v in old_counters.items() if k!='path'}
        row=dict(path=str(test),output=output,counters=counters)
        verify_dump(row)
        row['numerical']=check(test/'output.fp16',1024,expected=np.load(out/'references/rich_t1024_bc512.npy'))
        (test/'accepted.json').write_text(json.dumps(row,indent=2))
    print('Failed case repaired: identical ROI counters/output and valid export',flush=True)
    status('auditing_saved_exports')
    # Snapshot metadata; retain every raw guest/result directory or archive it.
    for name in ('status.json','progress.json'):
        dst=repair/(name+'.before');src=out/name
        if src.exists() and not dst.exists():shutil.copy2(src,dst)
    archive.mkdir(exist_ok=True);audited=0;moved=[]
    for p in sorted((out/'runs').iterdir()):
        accepted=p/'accepted.json';reason=None
        if accepted.exists():
            audited+=1
            try:verify_dump(json.loads(accepted.read_text()))
            except AssertionError:reason='guest/dump hash mismatch'
        elif (p/'run.log').exists():reason='incomplete or failed validation'
        if reason:
            target=archive/p.name;assert not target.exists(),('inspect duplicate archive',target)
            p.rename(target);moved.append(dict(path=str(target),reason=reason))
    (repair/'audit.json').write_text(json.dumps(dict(audited=audited,archived=moved),indent=2))
    print('Export audit',audited,'accepted rows;',len(moved),'artifacts preserved for rerun',flush=True)
    source=Path(queue.__file__).read_text()
    changes=[
      ("row=json.loads(done.read_text())","row=export_verify(json.loads(done.read_text()))"),
      ("output,counters,cfg=read(p,16);audit(baseline_cfg,cfg,0)",
       "output,counters,cfg=read(p,16);audit(baseline_cfg,cfg,0)\n        export_verify(dict(path=str(p),output=output))"),
      ("return p/'guest'","return export_patch(p/'guest')")]
    for before,after in changes:
        assert source.count(before)==1,before
        source=source.replace(before,after)
    files=[Path(__file__),Path(queue.__file__),Path(__file__).with_name('fa_export_patch.py'),Path(__file__).with_name('fa_export_hook.S')]
    (repair/'adapter_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    (repair/'queue_adapted_snapshot.py').write_text(source)
    # __file__ retains original location so original source/build manifest and
    # all path calculations remain valid. Original module is never rewritten.
    scope=dict(__name__='staged_export_adapter',__file__=queue.__file__,export_patch=patch,export_verify=verify_dump)
    exec(compile(source,str(repair/'queue_adapted_snapshot.py'),'exec'),scope)
    sys.argv=[queue.__file__,str(out),'--resume']
    status('regression_passed_resuming',audited=audited,archived=len(moved))
    scope['main']()

if __name__=='__main__':main()
