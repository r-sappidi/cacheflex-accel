"""Reuse identical upstream libraries; exclude worker startup from operator ROI."""
import json,shutil,subprocess
from build import BASE,HERE,sha,save
for mode in ('upstream_sve','ported_sve'):
    old=BASE/mode;out=BASE/(mode+'_ready');out.mkdir()
    shutil.copy2(HERE/'driver.cpp',out/'driver.cpp')
    cmd=json.loads((old/'link.json').read_text())
    cmd=[str(out/'driver.cpp') if x==str(old/'driver.cpp') else str(out/'guest') if x==str(old/'guest') else x for x in cmd]
    save(out/'link.json',cmd)
    with (out/'build.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    objdump=cmd[0].replace('g++','objdump')
    with (out/'guest.dis').open('w') as f:subprocess.run([objdump,'-d','-C',str(out/'guest')],stdout=f,check=True)
    provenance=json.loads((old/'provenance.json').read_text());provenance['worker_readiness']='disjoint add graph outside ROI; no FA warmup'
    save(out/'provenance.json',provenance)
    frozen=json.loads((old/'frozen.json').read_text())
    frozen.update({str(p):sha(p) for p in out.iterdir() if p.is_file()})
    save(out/'frozen.json',frozen)
