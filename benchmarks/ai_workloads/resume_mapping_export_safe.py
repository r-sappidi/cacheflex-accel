"""Replay gates and resume the unchanged candidate plan in a separate directory.

Frozen old guests are patched only after ROI. Original artifacts are untouched.
Every previously measured case must reproduce all extracted epoch counters.
"""
import json
from pathlib import Path
import sys
from repair_screen_export import patch
from run_mapping_screen import sha

HERE = Path(__file__).resolve().parent


def main():
    old, new = map(lambda p: Path(p).resolve(), sys.argv[1:3])
    assert not new.exists()
    assert json.loads((old/'status.json').read_text())['stage'] == 'failed'
    manifest = json.loads((old/'manifest.json').read_text())
    for name, digest in manifest.items():
        assert sha(name) == digest, ('old dependency changed', name)
    source = (HERE/'run_mapping_screen.py').read_text()
    def replace(before, after):
        nonlocal source
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    replace("        for name in ('projection','mlp_separate','mlp_combined'):\n", """        shutil.copytree(OLD/'inputs',out/'inputs',dirs_exist_ok=True)
        for name in ('projection','mlp_separate','mlp_combined'):
""")
    begin = source.index("            cmd=['bash',str(HERE/'build_screen_projection.sh')")
    end = source.index('        paths=list(', begin)
    source = source[:begin] + """            shutil.copytree(OLD/'builds'/name,build)
            for arch in ('cache','spm'):patch(build/arch)
""" + source[end:]
    replace("            save(path/'accepted.json',row);", """            prior_path=OLD/'runs'/name
            if (prior_path/'stats.txt').exists():
                old_epochs=epochs(prior_path,3)
                assert times==old_epochs,('ROI counter drift',name,times,old_epochs)
                old_text=(prior_path/'run.log').read_text()
                old_checks=[dict(re.findall(r'(\\w+)=([^\\s]+)',line)) for line in old_text.splitlines() if line.startswith('MLP_RESULT' if mlp else 'SCREEN_RESULT')]
                assert checks==old_checks,('guest result drift',name)
                save(path/'repair_equivalence.json',dict(passed=True,original=str(prior_path),epochs=times,checks=checks))
            save(path/'accepted.json',row);""")
    # Explicitly test the original failure before any further target screening.
    replace("        # Breadth first: initial point per shape/family before refinements.", """        status('original_failure_regression')
        failed=json.loads((OLD/'runs/0306fac00b53872e_cache/case.json').read_text())
        pair(failed)
        # Breadth first: initial point per shape/family before refinements.""")
    replace("                if round_index<len(points):pair(points[round_index])", "                if round_index<len(points) and points[round_index]!=failed:pair(points[round_index])")
    # Record the exact adapted runner before it starts; no silent source mutations.
    snapshot = new.with_suffix('.runner.py')
    assert not snapshot.exists()
    snapshot.write_text(source)
    sys.argv = [str(HERE/'run_mapping_screen.py'), str(new)]
    namespace = dict(__name__='__main__', __file__=str(HERE/'run_mapping_screen.py'), OLD=old, patch=patch)
    exec(compile(source, str(snapshot), 'exec'), namespace)


if __name__ == '__main__':
    main()
