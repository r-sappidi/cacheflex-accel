"""Link an isolated diagnostic simulator from SCons's exact manifest."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
out=Path(sys.argv[1]).resolve();root=Path(__file__).resolve().parents[2]
spec=json.loads((out/'link_manifest.json').read_text());args=spec['argv']
args[args.index('-o')+1]=str(out/'gem5.opt')
with (out/'link.log').open('w') as log:
 subprocess.run(args,cwd=root/'gem5/build/ARM_MESI_Three_Level_SPM',
                env=dict(spec['environment'],LD_LIBRARY_PATH=os.environ['LD_LIBRARY_PATH']),
                stdout=log,stderr=subprocess.STDOUT,check=True)
text=subprocess.check_output([str(out/'gem5.opt'),'--debug-help'],text=True)
assert 'CacheFlexResidency' in text
(out/'simulator.sha256').write_text(hashlib.sha256((out/'gem5.opt').read_bytes()).hexdigest()+'\n')
