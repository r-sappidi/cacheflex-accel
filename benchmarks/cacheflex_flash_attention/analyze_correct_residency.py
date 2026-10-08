"""Actual grouped-buffer extents and current DWARF labels for corrected FA."""
from pathlib import Path
import json,re,sys
import analyze_residency as base
from analyze_spm_residency import qualify
from analyze_correct_fa import phase
def ranges(log,t,g,d=64,br=64,bc=128):
    rows=min(t,g*br)
    shared=dict(Q=8*t*d*2,K=t*d*2,V=t*d*2,O=8*t*d*2,KT=t*d*2)
    local=dict(qap=rows*d*2,S=br*bc*2,P=br*bc*2,PV=8*d*2,pap=8*bc*2,Oacc=rows*d*4,m=rows*4,l=rows*4)
    result=[]
    for line in log.splitlines():
        if not line.startswith(('FLASH_SHARED ','FLASH_ADDR ')):continue
        fields=dict(re.findall(r'(\w+)=(0x[0-9a-f]+|\d+)',line))
        for name,size in (local if 'tid' in fields else shared).items():
            if name in fields:
                start=int(fields[name],0);label=f"core{fields['tid']}.{name}" if 'tid' in fields else name
                result.append((start,start+size,label))
    assert len(result)==69
    for i,(a,b,n) in enumerate(result):
        assert all(b<=c or a>=d for c,d,_ in result[i+1:]),('overlapping ranges',n)
    return result
if __name__=='__main__':
    run=Path(sys.argv[1]);g=int(sys.argv[2]);args=json.loads((run/'command.json').read_text())
    elf=Path(args[args.index('--cmd')+1]).with_name('guest.debug')
    # Recover the unprinted packed-K buffer from actual operand addresses,
    # not guessed allocator offsets. This guest stages only K; coherent QK's
    # sole ld1h z2,[x22] is its packed-B (K-transpose) operand.
    attribution=json.loads((run/'correct_attribution.json').read_text())
    spm=attribution['traffic']['m_spmcpRequestsIssued']>0
    bpcs={int(x['pc'],16) for x in attribution['pcs'] if x['phase']=='QK' and re.search(r'ld1h.*z2\.h.*\[x22\]',x['instruction'])}
    if not spm:assert len(bpcs)==1,bpcs
    lines=set()
    with (run/'debug.trace').open() as stream:
        for line in stream:
            if 'CFRES ISSUE ' not in line:continue
            if spm and 'type=SPMCP_fetch ' not in line:continue
            if not spm and not any(f'pc={hex(pc)} ' in line for pc in bpcs):continue
            f=dict(re.findall(r'(\w+)=([^\s]+)',line))
            if not spm and f['type']!='LD':continue
            va=int(f['va'],0);size=64 if spm else int(f['bytes'],0)
            lines.update(range(va&~63,((va+size-1)&~63)+1,64))
            if len(lines)==2048:break
    assert len(lines)==2048 and max(lines)-min(lines)+64==131072,('incomplete packed K coverage',len(lines))
    krange=(min(lines),max(lines)+64,'Kcache')
    known=ranges((run/'run.log').read_text(),1024,g)
    assert all(krange[1]<=a or krange[0]>=b for a,b,_ in known)
    base.buffers=lambda log,t:known+[krange];base.phase=phase
    from fast_residency_events import events as original_events
    def roi_events(path):
        for event in original_events(path):
            yield event
            if event[2]=='END':break
    base.events=roi_events
    base.analyze(run,elf,1024,fast_occupancy=True);qualify(run)
    (run/'buffer_limits.json').write_text(json.dumps({'group':g,'local_rows':min(1024,g*64),'Kcache_range':krange,'Kcache_evidence':'2048 contiguous distinct 64B source lines from K-only SPMCP or decoded QK B loads; no allocator inference.'},indent=2))
