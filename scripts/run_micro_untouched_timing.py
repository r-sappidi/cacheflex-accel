"""Untouched upstream FA timing controls; explicitly NOT numerical acceptance."""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'results/cacheflex_flash_attention'
UP = B / 'micro_upstream_20260918'
CONTROL = B / 'micro_paper_reproduction_v2_20260921'
OUT = B / 'micro_untouched_timing_v2_20260921'
CXX = B / 'micro_toolchain_audit_20260918/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++'
TILES = [(64,384), (32,384), (64,256), (32,256), (64,128), (32,128)]
ENV = dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib')
LOCK = threading.RLock()
STOP = threading.Event()


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(p, value):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)


def flat(obj, prefix=''):
    if isinstance(obj, dict):
        return {k:v for a,b in obj.items() for k,v in flat(b,prefix+'.'+a).items()}
    if isinstance(obj, list):
        return {k:v for a,b in enumerate(obj) for k,v in flat(b,prefix+f'[{a}]').items()}
    return {prefix:obj}


def main():
    OUT.mkdir()
    preserved = OUT/'preserved'; preserved.mkdir()
    builds = OUT/'builds'; builds.mkdir()
    (OUT/'runs').mkdir()
    state = dict(pid=os.getpid(), stage='building', active={}, completed=[], max_simulators=2,
                 arithmetic_validation=False, purpose='timing-only, untouched upstream FA')
    def status(stage=None):
        with LOCK:
            if stage: state['stage'] = stage
            state['updated'] = time.time(); save(OUT/'status.json',state)
    status()
    try:
        revision = subprocess.check_output(['git','-C',str(UP),'rev-parse','HEAD'],text=True).strip()
        assert revision == '82ac0b9de1ed408ce5cd53067267c8a855b4eef3'
        shutil.copytree(UP/'experiments/end2end/src',builds/'src')
        shutil.copytree(UP/'gem5/include',preserved/'include')
        for source,dest in [
            (UP/'gem5/cacheflex_sw/spm_compiler.py', preserved/'spm_compiler.py'),
            (UP/'gem5/util/m5/src/abi/arm64/m5op.S',preserved/'m5op.S'),
            (UP/'experiments/end2end/build_vl16.sh',preserved/'build_vl16.sh'),
            (UP/'experiments/end2end/scripts/sweep_vl16.sh',preserved/'sweep_vl16.sh'),
            (CONTROL/'preserved/classic.opt',preserved/'classic.opt'),
            (Path(__file__),preserved/'runner.py')]:
            shutil.copy2(source,dest)
        shutil.copytree(CONTROL/'preserved/classic_configs',preserved/'configs',
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        source_audit = {str(p.relative_to(builds/'src')):sha(p) for p in (builds/'src').rglob('*') if p.is_file()}
        assert all(sha(UP/'experiments/end2end/src'/n)==h for n,h in source_audit.items())
        save(OUT/'source_audit.json',source_audit)
        flags = ['-O3','-std=c++17','-march=armv8.2-a+sve+fp16','-static','-Isrc',
                 '-DGEM5','-DGEM5_SE','-I'+str(preserved/'include')]
        commands = [
            [str(CXX),'-c','-I'+str(preserved/'include'),str(preserved/'m5op.S'),'-o',str(builds/'m5op.o')],
            [str(CXX),*flags,'src/cache_flash_attn.cpp',str(builds/'m5op.o'),'-lm','-o',str(builds/'cache')],
            [str(CXX),*flags,'-DVL_16','-mbranch-protection=none','-S','-o',str(builds/'spm.s'),'src/spm_flash_attn.cpp'],
            ['python3',str(preserved/'spm_compiler.py'),str(builds/'spm.s'),str(builds/'spm.enc.s')],
            [str(CXX),'-static','-o',str(builds/'spm'),str(builds/'spm.enc.s'),str(builds/'m5op.o'),'-lm']]
        save(OUT/'build_commands.json',commands)
        with (OUT/'build.log').open('w') as log:
            for cmd in commands: subprocess.run(cmd,cwd=builds,stdout=log,stderr=subprocess.STDOUT,check=True)
        save(OUT/'provenance.json',dict(revision=revision,compiler=str(CXX),compiler_sha256=sha(CXX),
             version=subprocess.check_output([str(CXX),'--version'],text=True),
             source_unmodified=True,conversion_fix=False,exports_added=False,extra_debug_flags=False,
             flags='Matches upstream build_vl16.sh; isolated paths and freshly assembled upstream m5op stub',
             simulator_sha256=sha(preserved/'classic.opt'),
             limitations=['Original paper-winning binary/config unavailable','No arithmetic validity claim',
                          'CPUProgressEvent at 100kHz for observation, matching corrected control']))
        options = shlex.split(re.search(r'GEM5_CONFIG="(.*?)"',(preserved/'sweep_vl16.sh').read_text(),re.S)[1].replace('\\\n',' '))
        originals = {a:json.loads((CONTROL/'runs'/f'classic_{a}_target_br64_bc384/config.json').read_text()) for a in ('cache','spm')}
        frozen = [p for p in preserved.rglob('*') if p.is_file() and p.suffix != '.pyc'] + [builds/'cache',builds/'spm']
        manifest = {str(p):sha(p) for p in frozen}; save(OUT/'manifest.json',manifest)
        save(OUT/'plan.json',dict(T=4096,Hq=32,Hkv=8,d=64,VL=16,cores=1,tiles=TILES,
             model='upstream Classic',small_smokes='T320/Hq1/Hkv1, both kernels; execution only',
             primary='First stats ROI, cold as upstream layer extractor',secondary='Printed warm times when present',
             target_speedup=1.34,numerical_gate=False))

        def audit(p,guest,args):
            expected=json.loads(json.dumps(originals[guest.name]))
            cpu=expected['system']['cpu'][0]
            cpu['workload'][0]['cmd']=[str(guest)]+args
            cpu['workload'][0]['executable']=str(guest)
            for redirect in expected['system']['redirect_paths']:
                for i,path in enumerate(redirect['host_paths']):
                    q=Path(path)
                    if q.parent.name=='fs' and q.name in ('proc','sys','tmp'):
                        redirect['host_paths'][i]=str(p/'fs'/q.name)
            actual=json.loads((p/'config.json').read_text())
            a,b=flat(actual),flat(expected)
            changes={k:[b.get(k),a.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
            assert not changes,changes
            save(p/'config_audit.json',dict(passed=True,fields=len(a),against='Corrected upstream-core frozen control; only argv/executable/output paths changed'))

        def report():
            rows=[r for r in state['completed'] if not r['smoke']]
            text=['# Untouched MICRO FA: timing only','',
                  'Arithmetic correctness NOT accepted. Target: T4096/VL16, published ratio 1.34x.','',
                  '| Kernel | Br | Bc | Cold ROI ms |','|---|---:|---:|---:|']
            text += [f"| {r['arch']} | {r['Br']} | {r['Bc']} | {r['roi_us']/1000:.6f} |" for r in rows]
            for br,bc in TILES:
                pair={r['arch']:r for r in rows if (r['Br'],r['Bc'])==(br,bc)}
                if len(pair)==2: text += ['',f"Br{br}/Bc{bc} timing-only ratio: {pair['cache']['roi_us']/pair['spm']['roi_us']:.6f}x."]
            if len(rows)==12:
                wins={a:min((r for r in rows if r['arch']==a),key=lambda r:r['roi_us']) for a in ('cache','spm')}
                result=dict(winners=wins,speedup=wins['cache']['roi_us']/wins['spm']['roi_us'],arithmetic_validation=False)
                save(OUT/'summary.json',result)
                text += ['',f"Independently selected minima: {result['speedup']:.6f}x (timing only)."]
            (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')

        def run(arch,br,bc,smoke=False):
            label=f'{arch}_{"smoke" if smoke else "t4096"}_br{br}_bc{bc}'
            p=OUT/'runs'/label; p.mkdir(); guest=builds/arch
            args=[str(x) for x in ((320,1,1,64,1,br,bc) if smoke else (4096,32,8,64,1,br,bc))]
            assert all(sha(Path(f))==h for f,h in manifest.items())
            cmd=[str(preserved/'classic.opt'),f'--outdir={p}',str(preserved/'configs/deprecated/example/se.py'),
                 *options,'--num-cpus=1','--prog-interval=100kHz','-c',str(guest),'-o',' '.join(args)]
            save(p/'command.json',cmd); start=time.monotonic()
            try:
                with (p/'run.log').open('w') as log:
                    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=ENV)
                    with LOCK: state['active'][label]=dict(pid=proc.pid,path=str(p)); status()
                    checked=False
                    try:
                        while proc.poll() is None:
                            if STOP.is_set(): raise RuntimeError('peer failed')
                            if time.monotonic()-start > (1800 if smoke else 48*3600): raise RuntimeError('host timeout')
                            if shutil.disk_usage(OUT).free < 50*1024**3: raise RuntimeError('disk guard')
                            if not checked and (p/'config.json').exists():
                                try: audit(p,guest,args)
                                except json.JSONDecodeError: pass
                                else: checked=True
                            save(p/'progress.json',dict(pid=proc.pid,host_seconds=time.monotonic()-start))
                            try: proc.wait(timeout=5)
                            except subprocess.TimeoutExpired: pass
                    finally:
                        if proc.poll() is None: proc.terminate(); proc.wait(timeout=30)
                assert proc.returncode==0,proc.returncode
                audit(p,guest,args)
                log=(p/'run.log').read_text(errors='replace')
                assert 'CHECKSUM:' in log and 'exiting with last active thread context' in log
                sections=(p/'stats.txt').read_text().split('---------- Begin Simulation Statistics')
                assert len(sections)>1
                def stat(name): return float(re.search(r'^'+re.escape(name)+r'\s+(\S+)',sections[1],re.M)[1])
                row=dict(arch=arch,Br=br,Bc=bc,smoke=smoke,roi_us=stat('simTicks')/stat('simFreq')*1e6,
                         host_seconds=time.monotonic()-start,checksum=re.findall(r'CHECKSUM:\s*(\S+)',log),
                         printed_timings=[s for s in log.splitlines() if re.search(r'COLD:|WARM|avg:',s)],
                         arithmetic_validation=False,path=str(p))
                assert all(sha(Path(f))==h for f,h in manifest.items())
                save(p/'timing_only_complete.json',row)
                with LOCK: state['completed'].append(row); state['active'].pop(label); report(); status()
            except BaseException:
                STOP.set(); raise

        for smoke,tiles in [(True,[(64,384)]),(False,TILES)]:
            for br,bc in tiles:
                status('execution_smokes' if smoke else f'timing_sweep_br{br}_bc{bc}')
                with cf.ThreadPoolExecutor(max_workers=2) as pool:
                    jobs=[pool.submit(run,a,br,bc,smoke) for a in ('cache','spm')]
                    for job in cf.as_completed(jobs): job.result()
        status('completed_timing_only')
    except BaseException as exc:
        STOP.set(); state['error']=repr(exc); status('failed'); raise


if __name__=='__main__': main()
