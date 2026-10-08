"""Extract the observed first divergence and document the reviewed restart."""
import collections
import gzip
import itertools
import json
from pathlib import Path
from audit_output_resident_control import OUT,OLD,NEW,ROOT
from build_access_timing import sha,save

def main():
    assert (OUT/'complete.json').exists()
    before=collections.deque(maxlen=8)
    with gzip.open(OUT/'old/trace.gz','rt') as a,gzip.open(OUT/'new/trace.gz','rt') as b:
        for i,(x,y) in enumerate(itertools.zip_longest(a,b)):
            if x!=y:
                assert '2687999600: system.cpu5.commit' in x and 'Committing instruction with fault' in x
                assert '2687999600: system.cpu5.commit' in y and 'Stores outstanding, fault must wait' in y
                excerpt={'identical_lines_before_divergence':i,'prior':list(before),
                         'old':x,'new':y,'old_next':list(itertools.islice(a,12)),
                         'new_next':list(itertools.islice(b,12))}
                break
            before.append(x)
        else:raise AssertionError('Expected divergence absent')
    save(OUT/'first_divergence.json',excerpt)
    evidence={}
    for name in ('old','new'):
        events=[]
        with gzip.open(OUT/name/'trace.gz','rt') as f:
            for l in f:
                if 'system.cpu5' in l and (
                    'Resp Recv: SN=40300' in l or
                    'Store/Cmd Sent to Cache: SN=40300' in l or
                    '[sn:40301] Committing instruction with fault' in l):events.append(l.rstrip())
        evidence[name]=events
    save(OUT/'fault_copy_order.json',evidence)
    source=ROOT/'gem5/src/cpu/o3/lsq_unit.cc'
    assert 'if (entry.retired && !entry.ready)' in source.read_text()
    summary=json.loads((OUT/'summary.json').read_text())
    approved=dict(cause='Committed SPMCP now participates in stores-to-drain at a fault boundary',
                  first_divergence_tick=2687999600,core=5,faulting_sn=40301,
                  faulting_pc='0x401e58',copy_sn=40300,
                  old_fault_tick=2687999600,new_fault_tick=2688191200,
                  new_copy_completion_tick=2688190801,local_delay_cycles=479,
                  fault_type='Not identified by these debug flags; do not call this a page fault',
                  numerical_and_config_valid=True,committed_work_unchanged=True,
                  source=sha(source),summary=summary,
                  next='Fresh cache control + exact same-model SPM repeat, then output-resident pair')
    save(OUT/'approved.json',approved)
    (OUT/'REPORT.md').write_text('''# Output-resident control audit

The queue failed a historical counter-equality assertion, not arithmetic or a
simulator crash. T4096 V-SPM checks all 262,144 outputs, matches the historical
output bytes/hash, unchanged configuration and bounded-resource checks, and
all committed instruction-type counts. Copies issued/completed remain 524,288.

ROI changes from 5715.4256 to 5715.0284 us (-0.006950%). Received network
packets rise 0.012866%; flits rise 0.017159%. Thousands of changed histogram
and per-PC fields are not thousands of independent failures.

## First divergence

Progress counters agree through tick 2,680,000,000; the next sample differs.
Two diagnostic prefix replays use the frozen old/new models, unchanged guest
and configuration, stop at 2,690,000,000, and trace only the final 10M ticks.
Their first 516,233 trace lines agree. At tick 2,687,999,600 core5's regular LDR
at PC 0x401e58 (SN40301) is faulting. The older SPMCP SN40300 has just been
issued and freed its regular SQ entry. The old model handles the fault;
the new model waits for the committed bridge shadow to become ready.
The new copy completes at 2,688,190,801; the fault is handled at 2,688,191,200.
This is 479 extra local CPU cycles at this one boundary, NOT an additive
end-to-end penalty. Subsequent multicore request ordering/arbitration diverges;
the measured overall run is actually 0.3972 us faster.

The source chain is LSQUnit::commitStores marking both bridge types retired,
LSQUnit::hasPendingWriteback including unfinished committed copies,
LSQ::hasStoresToWB, then Commit::commitHead's fault-drain check. This is the
intended precise-boundary drain behavior. The debug flags do not identify the
fault type; no claim that this is an OS page fault.

## Resolution

Keep the repaired ordering. Do not whitelist every changed counter or roll
back correctness to reproduce old timings. Reuse unchanged, accepted T512
gates; rerun the interrupted T4096 cache control and repeat repaired V-SPM.
Require the repeat to match ALL finite numeric non-host stats from the first
repaired V-SPM run. Require numerical/configuration/resource and committed-work
checks for controls. Then run output-SPM and its matched cache kernel with the
same frozen simulator and compare to the fresh controls. Historical counters
remain diagnostic evidence, not the correctness oracle for a changed model.
''')

if __name__=='__main__':main()
