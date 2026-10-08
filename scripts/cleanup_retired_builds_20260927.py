"""Recoverable cleanup of explicit retired simulators and MICRO objects.

No source, result, trace, current simulator, current incremental build, guest,
or toolchain deletion. Review plan first; execute rechecks open files and hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time
from cleanup_builds_20260921 import ROOT, FA, live_files, sha, save

OUT=ROOT/'results/build_cleanup_20260927'
MANIFESTS=[
    FA/'llama_cpu_20260925/experiment_export_safe_20260926/inputs.json',
    FA/'spm_set_spread_20260927/experiment/inputs.json',
    ROOT/'results/ai_workloads/swiglu_finite_transfer_20260927/inputs.json',
]
KEEP=[
    FA/'output_resident_20260924/build_model_v6/gem5.opt',
    FA/'spm_early_address_20260924/build_v4/gem5.opt',
    FA/'finite_model_20260926/build_v3/gem5.opt',
    FA/'spm_set_spread_20260927/build/gem5.opt',
    ROOT/'gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt',
]

def identity(p):
    assert p.resolve()==p and not p.is_symlink() and p.is_file(),p
    s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]

def protect():
    # Active manifest must match recorded hashes. Failed/paused queues protect
    # paths too, but their scripts may have been repaired since they stopped.
    active=json.loads(MANIFESTS[0].read_text())
    for p,h in active.items():assert sha(Path(p))==h,('active dependency drift',p)
    paths=set(active)
    for manifest in MANIFESTS[1:]:paths.update(json.loads(manifest.read_text()))
    paths.update(str(p) for p in KEEP)
    return paths,active

def plan():
    OUT.mkdir()
    protected,active=protect();mapped=live_files()
    tracked={str(ROOT/p) for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if p}
    micro=FA/'micro_upstream_20260918/gem5'
    tracked.update(str(micro/p) for p in subprocess.check_output(['git','ls-files','-z'],cwd=micro).decode().split('\0') if p)
    candidates={FA/'output_resident_20260924'/b/'gem5.opt':'Superseded output-resident simulator; validated v6 retained'
        for b in ('build_model','build_model_v2','build_model_v3','build_model_v4','build_model_v5')}
    candidates[FA/'spm_early_address_20260924/build_v2/gem5.opt']='Superseded early-address prototype; active v4 retained'
    for p in (micro/'build').rglob('*'):
        if p.is_file() and not p.is_symlink() and p.suffix in ('.o','.os'):
            candidates[p]='Inactive MICRO generated object; source and final executable retained'
    rows=[];skipped=[]
    for p,reason in sorted(candidates.items()):
        if not p.exists():continue
        if str(p) in protected or str(p) in mapped or str(p) in tracked:
            skipped.append(str(p));continue
        assert p.is_relative_to(ROOT) and p.resolve()==p
        with p.open('rb') as f:assert f.read(4)==b'\x7fELF',p
        s=p.stat();rows.append(dict(path=str(p),member=str(p.relative_to(ROOT)),
            identity=identity(p),allocated=s.st_blocks*512,reason=reason))
    keep_hashes={str(p):sha(p) for p in KEEP}
    save(OUT/'plan.json',rows);save(OUT/'skipped.json',skipped)
    save(OUT/'protected_paths.json',sorted(protected))
    save(OUT/'active_dependencies.json',active);save(OUT/'retained_builds.json',keep_hashes)
    print(json.dumps(dict(files=len(rows),allocated_GiB=sum(r['allocated'] for r in rows)/2**30,
        skipped=len(skipped),categories={why:sum(r['allocated'] for r in rows if r['reason']==why)/2**30 for why in sorted({r['reason'] for r in rows})}),indent=2))

def execute():
    assert not (OUT/'complete.json').exists()
    rows=json.loads((OUT/'plan.json').read_text())
    protected,active=protect();mapped=live_files()
    assert active==json.loads((OUT/'active_dependencies.json').read_text())
    keep=json.loads((OUT/'retained_builds.json').read_text())
    for p,h in keep.items():assert sha(Path(p))==h,('retained build changed',p)
    for r in rows:
        p=Path(r['path']);assert identity(p)==r['identity']
        assert str(p) not in protected and str(p) not in mapped
        r['sha256']=sha(p)
    save(OUT/'deleted_manifest.json',rows)
    archive=OUT/'retired_builds.tar.zst';assert not archive.exists()
    save(OUT/'status.json',dict(stage='archiving',pid=__import__('os').getpid()))
    subprocess.run(['tar','-I','zstd -1 -T1','-cf',str(archive),'-C',str(ROOT),'--']+
                   [r['member'] for r in rows],check=True)
    save(OUT/'status.json',dict(stage='verifying_archive'))
    expected={r['member']:r['sha256'] for r in rows};seen=set()
    proc=subprocess.Popen(['zstd','-dc',str(archive)],stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
        for member in tar:
            assert member.isfile() and member.name in expected
            h=hashlib.sha256();stream=tar.extractfile(member)
            for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
            assert h.hexdigest()==expected[member.name],member.name
            seen.add(member.name)
    while proc.stdout.read(1024*1024):pass
    assert proc.wait()==0 and seen==set(expected)
    # Recheck all identities and dependencies after archiving, before deletion.
    protected,active=protect();mapped=live_files()
    for r in rows:
        p=Path(r['path']);assert identity(p)==r['identity']
        assert str(p) not in protected and str(p) not in mapped
    save(OUT/'status.json',dict(stage='removing_verified_originals'))
    for r in rows:Path(r['path']).unlink()
    protect()
    for p,h in keep.items():assert sha(Path(p))==h
    original=sum(r['allocated'] for r in rows);compressed=archive.stat().st_blocks*512
    report=dict(files=len(rows),original_allocated_bytes=original,archive_allocated_bytes=compressed,
        net_reclaimed_bytes=original-compressed,archive_sha256=sha(archive),
        active_dependency_hashes_verified=len(active),retained_builds_verified=len(keep),
        preserved='All sources, configs, results, traces, current builds, guests and toolchains',
        recovery=f'tar -I zstd -xf {archive} -C {ROOT}',completed=time.time())
    save(OUT/'complete.json',report);save(OUT/'status.json',dict(stage='complete'))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    execute() if a.execute else plan()
