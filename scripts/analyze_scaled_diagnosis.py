"""Rolling read-only latency and proxy-validity reports, accepted runs only."""
import json
from pathlib import Path
import sys
import time
from run_single_diagnosis import phase,prior
from analyze_fa_retention_pcs import a,instruction_bytes
from run_clean_controls import save

def main():
    root=Path(sys.argv[1]).resolve();out=root/'analysis';out.mkdir(exist_ok=True)
    exp=root/'experiment';seen={}
    while True:
        for f in sorted((exp/'runs').glob('*/accepted.json')):
            if str(f) in seen:continue
            row=json.loads(f.read_text());run=f.parent
            guest=exp/'builds'/row['arch']/'guest.export-safe';debug=guest.with_name('guest.debug')
            a.phase=phase;res=a.analyze(run,debug);gd=guest.read_bytes();dd=debug.read_bytes()
            for pc in res['pcs']:
                addr=int(pc['pc'],16);assert instruction_bytes(gd,addr)==instruction_bytes(dd,addr)
            save(out/(run.name+'.json'),res);seen[str(f)]=row
        lookup={(r['arch'],r['T'],r['name']):r for r in seen.values()}
        comparisons=[]
        for arch in ('cache','spm'):
            for t in (256,4096):
                for mode,ctrl in [('fetch17','shared17'),('fetch16','new_default'),('bridge32','shared32'),('both33','shared33')]:
                    x=lookup.get((arch,t,ctrl));y=lookup.get((arch,t,mode))
                    if x and y:
                        b=x['counters']['roi_us'];v=y['counters']['roi_us']
                        comparisons.append(dict(arch=arch,T=t,control=ctrl,intervention=mode,
                          control_us=b,intervention_us=v,recovered_us=b-v,recovered_percent=100*(b-v)/b))
        save(out/'equal_credit_latency_effects.json',comparisons)
        proxy=[]
        for arch in ('cache','spm'):
            r=lookup.get((arch,4096,'old_default'))
            if not r:continue
            original=json.loads((prior(arch,16384)/'accepted.json').read_text())
            keys=('roi_us','dram_read_bytes','private_l2_misses','llc_misses','packets_injected::total','average_packet_queueing_latency','m_spmcpRequestsIssued')
            proxy.append(dict(arch=arch,scaled={k:r['counters'].get(k) for k in keys},
                  original={k:original['counters'].get(k) for k in keys}))
        save(out/'proxy_comparison.json',dict(rows=proxy,warning='Different QK active lanes and tile/group sizes. Similar footprint ratios do not establish timing equivalence; do not extrapolate latency gains automatically.'))
        lines=['# Admission isolation: end-to-end latency effects','',
          'Positive recovery means faster. Comparisons below hold total sequencer credits equal.',
          'Bridge partitions still share the core LSQ retry gate; this does not isolate every coupling.','',
          '| Architecture | T | Control → intervention | Recovered us | Recovered % |',
          '|---|---:|---|---:|---:|']
        for r in comparisons:lines.append(f"| {r['arch']} | {r['T']} | {r['control']} → {r['intervention']} | {r['recovered_us']:.4f} | {r['recovered_percent']:.3f}% |")
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
        stage=json.loads((exp/'status.json').read_text())['stage']
        save(out/'status.json',dict(stage=stage,analyzed=len(seen),updated=time.time()))
        if stage in ('failed','implemented_stages_complete'):return
        time.sleep(20)

if __name__=='__main__':main()
