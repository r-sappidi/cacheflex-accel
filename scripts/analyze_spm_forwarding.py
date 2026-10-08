"""Report matched latency and preserve a directly observed unresolved-store chain."""
import gzip
import json
from pathlib import Path
import re
from build_access_timing import ROOT,save
from run_access_timing import stats

BASE=ROOT/'results/cacheflex_flash_attention/spm_forwarding_20260924'

def main():
    out=BASE/'analysis';out.mkdir(exist_ok=True)
    trace=BASE/'trace_t512/trace.gz'
    # Store data/address execution follows the previous row's add/ZIP chain.
    # Preserve exact lines, rather than assigning aggregate retries to cycles.
    evidence=[];load_times=[];issue=None;store_exec={};store_done={}
    with gzip.open(trace,'rt') as f:
        for line in f:
            match=re.match(r'(\d+): system.cpu0\.',line)
            if not match:continue
            tick=int(match[1])
            if tick>317306000:break
            if tick<317190000:continue
            if 'SN=1551468 ' in line and 'Exec Load' in line:load_times.append(tick)
            if 'SN=1551468 ' in line and 'Load Issue to Cache' in line:issue=tick
            for sn in (1551394,1551395):
                if f'[sn:{sn}]' in line:
                    if 'Doing write' in line:store_exec[sn]=tick
                    if 'Completing store' in line:store_done[sn]=tick
            if any(str(n) in line for n in (1551468,1551394,1551395)):
                if 'Exec Load' not in line or len(load_times)==1:evidence.append(line)
    assert load_times[0]==317202400 and issue==317300400
    assert store_exec=={1551394:issue,1551395:issue}
    assert all(t>issue for t in store_done.values()) and len(store_done)==2
    (out/'unresolved_store_evidence.txt').write_text(''.join(evidence))
    record=dict(core=0,load=1551468,load_slot=2,
        older_stores=[1551394,1551395],older_store_slots=[0,1],
        first_attempt=load_times[0],successful_issue=issue,
        store_address_data_execution=store_exec,store_completions=store_done,
        observed_load_wait_cycles=(issue-load_times[0])/400,
        note='One local dependency wait, not end-to-end runtime attribution.')
    save(out/'dependency.json',record)
    off=stats(BASE/'experiment_v2/runs/fa_off_t512')
    on=stats(BASE/'experiment_v2/runs/fa_on_t512')
    assert off==on,'T512 outcome changed'
    lines=['# Forwarding outcome and dependency evidence','',
        'T512: OFF and ON both 172.2084 us, 1.000000x speedup. All finite '
        'numeric non-host counters match exactly, including zero forwarded '
        'loads. Both pass independent arithmetic and exact output checks. '
        'Focused regression has 259 forwarded loads and no numerical errors.','',
        'The automatic queue initially stopped because it required positive '
        'forwarding even in FA. That was an inappropriate applicability gate: '
        'zero use is a valid negative result. No model or tolerance changed; '
        'the focused regression still requires positive forwarding. T4096 '
        'continues in experiment_v3 using the validated small gates.','',
        '## Concrete trace chain','',
        '- Core 0 SPM load SN1551468 reads slot 2 (128 bytes across ways 0–1).',
        '- Its first attempt is tick 317202400; it issues at 317300400 '
        '(245 CPU cycles later).',
        '- Older SPM stores SN1551394/1551395 resolve to slots 0/1 at '
        '317300400, exactly when the load can issue.',
        '- Those stores complete later, at 317304801/317305201. The load '
        'does not wait for their completion once their addresses are known.',
        '- Thus this wait is conservative unresolved-store disambiguation, '
        'not waiting for data from an overlapping store. Forwarding cannot '
        'help this particular chain.','',
        'At PC4026bc/4026c0 the kernel loads O; FADD and ZIP produce store '
        'values for PC4026d4/4026d8. The next row has different slots. '
        'The current LSQ waits on any older store whose effective address is '
        'not yet available. Earlier address-only resolution or safe speculative '
        'disambiguation is a distinct follow-up mechanism; neither is included '
        'in this forwarding experiment.','',
        'This one trace interval is evidence of a mechanism, not a claim that '
        '245 cycles are additive end-to-end loss or that every retry has the '
        'same cause. T4096 timing is the target measurement.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
