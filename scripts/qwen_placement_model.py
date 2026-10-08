"""Placement/ownership accounting and resource-bound model for causal Qwen FA.

No simulator launch or imports of experiment coordinators. Exact schedule counts
are separate from estimated instructions/cache traffic. New KV schedules are
specifications, not claims that a compatible checked guest already exists.
"""
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import itertools
import math


@dataclass(frozen=True)
class Candidate:
    t: int = 4096
    ownership: str = 'query_head'  # query_legacy, query_head, kv
    placement: str = 'v'         # cache, q, o, v, qv, ov, qo, qov, kv
    lifetime: str = 'head'       # head, kv_group, all_heads
    br: int = 16
    bc: int = 256
    r: int = 256
    hq: int = 14
    hkv: int = 2
    d: int = 64
    cores: int = 8
    causal: bool = True
    stride: int = 0
    sync: str = 'batch'          # dependency is optimistic overlap diagnostic
    q_adapter: str = 'six_instruction'

    def __post_init__(self):
        # R is a free tuning dimension only in the legacy query-group schedule.
        if self.ownership=='query_head':object.__setattr__(self,'r',self.t//self.cores)
        if self.ownership=='kv':object.__setattr__(self,'r',self.br)

    @property
    def key(self):
        return '_'.join(str(x) for x in asdict(self).values())


@dataclass(frozen=True)
class Hardware:
    clock_ghz: float = 2.5
    sets: int = 1024
    ways: int = 8
    eligible_ways: int = 6
    line: int = 64
    l1_bytes: int = 65536
    commit_width: int = 5
    spm_issue_per_cycle: int = 1
    spm_load_latency: int = 7
    load_entries: int = 32
    store_entries: int = 48
    network_flit_bytes: int = 16
    merge_link_bytes_per_cycle: int = 16


HW = Hardware()


def validate(c):
    if c.ownership not in ('query_legacy','query_head','kv'):
        raise ValueError('unsupported ownership')
    if c.placement not in ('cache','q','o','v','qv','ov','qo','qov','kv'):
        raise ValueError('unsupported placement')
    if c.cores != 8 or c.d != 64 or c.br != 16:
        raise ValueError('initial model is eight cores, d64, Br16')
    if c.t <= 0 or c.t % 256 or c.bc not in (128,256,512) or c.t % c.bc:
        raise ValueError('invalid T/Bc geometry')
    if c.t//16 % c.br or c.hkv <= 0 or c.hq % c.hkv:
        raise ValueError('invalid head/ownership geometry')
    if c.stride not in (0,512) or c.sync not in ('batch','dependency'):
        raise ValueError('invalid mapping/synchronization')
    if c.q_adapter not in ('six_instruction','ideal_broadcast'):
        raise ValueError('invalid adapter')
    if c.ownership == 'kv':
        if c.placement not in ('cache','kv') or c.lifetime not in ('kv_group','all_heads'):
            raise ValueError('KV ownership supports cache/KV and group/all-head lifetimes')
        if (c.t//16) % c.bc:
            raise ValueError('KV token ownership boundaries must align with Bc')
    elif c.lifetime != 'head' or c.placement == 'kv':
        raise ValueError('invalid query lifetime/placement')
    if c.ownership == 'query_legacy' and (c.placement not in ('cache','v') or
                                          c.t % c.r or c.r % c.br):
        raise ValueError('legacy guest supports cache/V with integral groups')


def ranges(c, core):
    """Complementary early/late token blocks; partitions [0,T) exactly."""
    n=c.t//(2*c.cores)
    return [(core*n,(core+1)*n),((2*c.cores-1-core)*n,(2*c.cores-core)*n)]


def layout(c):
    """Exact slot claims for specified ABI; does not predict coherent tags."""
    validate(c)
    slots={};regions={}
    def put(name, way, slot):
        if not 0 <= way < HW.eligible_ways or not 0 <= slot < HW.sets:
            raise ValueError('SPM layout exceeds eligible geometry')
        physical=(way,(slot+way*c.stride)%HW.sets)
        if physical in slots:
            raise ValueError('SPM slot overlap')
        slots[physical]=name
        regions[name]=regions.get(name,0)+64
    if c.ownership == 'kv' and c.placement == 'kv':
        # K: packed columns; V: row-major. Each vector's consecutive 64B
        # segments occupy consecutive ways, not a second SRAM allocation.
        heads=1 if c.lifetime=='kv_group' else c.hkv
        rows=c.t//c.cores
        for name in ('k','v'):
            for h in range(heads):
                region=(0 if name=='k' else heads)+h
                for line in range(rows*128//64):
                    put(name, (region*rows*2+line)//HW.sets,
                        (region*rows*2+line)%HW.sets)
        abi='candidate linear-64B slot ABI; wide-load lowering requires a new guest'
    else:
        rows=c.t//c.cores
        if 'q' in c.placement:
            for s in range(rows*2):put('q',4,s)
        if 'o' in c.placement:
            for w in range(4):
                for s in range(rows):put('o',w,s)
        if 'v' in c.placement:
            start=0 if c.ownership=='query_legacy' else 512
            for w in range(2):
                for s in range(start,start+c.bc):put('v',w,s)
        abi='FA-33 query ABI (stride512 requires validated remapping profile)'
    occupancy=[0]*HW.sets
    for _,s in slots:occupancy[s]+=1
    hist={str(i):occupancy.count(i) for i in sorted(set(occupancy))}
    free=[HW.ways-x for x in occupancy]
    return dict(regions=regions,spm_bytes=len(slots)*64,
                coherent_bytes=HW.sets*HW.ways*64-len(slots)*64,
                claimed_ways_histogram=hist,min_coherent_ways=min(free),
                mean_inverse_coherent_ways=sum(1/x for x in free)/HW.sets,
                slots=sorted([w,s,n] for (w,s),n in slots.items()),abi=abi)


@lru_cache(maxsize=512)
def account(c):
    """Exact copies, tile arithmetic, ownership, merge records for this schedule."""
    validate(c);mapping=layout(c)
    columns=('tasks','pairs','updates','copies','q_rows_packed','o_rows_initialized',
             'v_rows_staged','k_rows_staged','q_rows_staged','o_rows_staged',
             'release_lines','merge_records','merge_rows')
    cores=[dict.fromkeys(columns,0) for _ in range(c.cores)]
    # Per-head aggregates enable sum_h max_core(stage completion), rather
    # than hiding synchronization in a mean-core work estimate.
    heads=[]
    for head in range(c.hq):
        work=[dict.fromkeys(columns,0) for _ in range(c.cores)]
        if c.ownership=='query_legacy':
            groups=c.t//c.r
            for g in range(groups):
                core=(head*groups+(g-head)%groups)%c.cores
                x=work[core];x['tasks']+=1;x['q_rows_packed']+=c.r
                x['o_rows_initialized']+=c.r
                end=min(c.t,math.ceil((g+1)*c.r/c.bc)*c.bc) if c.causal else c.t
                x['updates']+=c.r*(end//c.bc);x['pairs']+=c.r*end
                if c.placement=='v':x['v_rows_staged']+=end
        elif c.ownership=='query_head':
            for core,x in enumerate(work):
                rr=ranges(c,core);rows=c.t//c.cores
                x['tasks']=1;x['q_rows_packed']=rows;x['o_rows_initialized']=rows
                end=math.ceil(rr[-1][1]/c.bc)*c.bc if c.causal else c.t
                if 'v' in c.placement:x['v_rows_staged']=end
                if 'q' in c.placement:x['q_rows_staged']=rows
                if 'o' in c.placement:x['o_rows_staged']=rows
                for a,b in rr:
                    for q in range(a,b,c.br):
                        panels=math.ceil((q+c.br)/c.bc) if c.causal else c.t//c.bc
                        x['updates']+=c.br*panels
                x['pairs']=x['updates']*c.bc
        else:
            # All cores process the same Q blocks against disjoint KV slices.
            # Empty causal partitions are skipped; merge owner rotates per Q block.
            first=head==0 if c.lifetime=='all_heads' else head%(c.hq//c.hkv)==0
            for core,x in enumerate(work):
                x['tasks']=1
                if first and c.placement=='kv':
                    n=c.t//c.cores*(c.hkv if c.lifetime=='all_heads' else 1)
                    x['k_rows_staged']=x['v_rows_staged']=n
                for q in range(0,c.t,c.br):
                    panels=sum(1 for a,b in ranges(c,core) for k in range(a,b,c.bc)
                               if not c.causal or k<q+c.br)
                    if panels:
                        x['q_rows_packed']+=c.br;x['o_rows_initialized']+=c.br
                        x['merge_records']+=c.br
                        x['updates']+=c.br*panels
                x['pairs']=x['updates']*c.bc
            for q in range(0,c.t,c.br):
                participants=sum(any(not c.causal or a<q+c.br for a,b in ranges(c,i))
                                 for i in range(c.cores))
                work[(q//c.br)%c.cores]['merge_rows']+=c.br*max(0,participants-1)
        for x in work:
            x['copies']=2*(x['q_rows_staged']+x['k_rows_staged']+x['v_rows_staged'])+4*x['o_rows_staged']
            if c.ownership=='query_head':x['release_lines']=mapping['spm_bytes']//64
            # Legacy releases at end of all tasks; KV at end of residency lifetime.
            elif c.ownership=='kv' and ((c.lifetime=='all_heads' and head==c.hq-1) or
                    (c.lifetime=='kv_group' and (head+1)%(c.hq//c.hkv)==0)):
                x['release_lines']=mapping['spm_bytes']//64
        for core,x in enumerate(work):
            for k,v in x.items():cores[core][k]+=v
        heads.append(work)
    if c.ownership=='query_legacy':
        for core,x in enumerate(cores):
            x['release_lines']=mapping['spm_bytes']//64
            heads[-1][core]['release_lines']=x['release_lines']
    totals={k:sum(x[k] for x in cores) for k in columns}
    useful=c.hq*c.t*(c.t+1)//2 if c.causal else c.hq*c.t*c.t
    # Partial row is FP32 O[64], maximum and denominator. Round each record
    # to 64B for the specified coherent software-merge layout.
    record_bytes=math.ceil((4*c.d+8)/64)*64
    merge_payload=totals['merge_rows']*record_bytes
    # These are exact algorithmic byte/work counts, NOT network/DRAM traffic.
    return dict(candidate=asdict(c),layout=mapping,cores=cores,heads=heads,totals=totals,
        useful_pairs=useful,padded_pairs=totals['pairs']-useful,
        gemm_flops=4*c.d*totals['pairs'],staged_bytes=totals['copies']*64,
        initial_k_pack_read_write_bytes=4*c.hkv*c.t*c.d,
        query_pack_read_write_bytes=4*c.d*totals['q_rows_packed'],
        partial_record_bytes=record_bytes,merge_payload_bytes=merge_payload,
        # Two max-rescaling subtractions, two denominator multiplies + add,
        # and 2 multiplies + add per O element. Max comparison and exp separate.
        merge_fp32_flops=totals['merge_rows']*(3*c.d+5),
        merge_comparisons=totals['merge_rows'],
        merge_exp_evaluations=totals['merge_rows']*2,
        output_bytes=c.hq*c.t*c.d*2,
        imbalance=max(x['pairs'] for x in cores)*c.cores/totals['pairs'],
        accuracy={'exact':'ownership, rectangular tile work, useful pairs, copy payloads, slot claims, specified merge records',
                  'estimated':'instruction overhead, cache traffic, timing, contention, merge instruction lowering'})


@dataclass(frozen=True)
class Scenario:
    coherent_bytes_cycle: float = 32
    effective_fp16_flops_cycle: float = 256
    copy_bytes_cycle: float = 16
    miss_floor: float = .1
    capacity_penalty: float = 1
    merge_bytes_cycle: float = 8
    alpha: float = 1   # fitted global positive scale, never a stall fraction


def phase_dag(nodes):
    """Max-plus critical path. Nodes explicitly list predecessor IDs."""
    ends={}
    for node in nodes:
        if node['id'] in ends or node['cycles']<0:
            raise ValueError('duplicate node or negative time')
        ends[node['id']]=max((ends[d] for d in node['deps']),default=0)+node['cycles']
    return max(ends.values(),default=0)


def estimate(c, s, detailed=False):
    a=account(c);place=c.placement;spmq='q' in place;spmo='o' in place
    spmk=place=='kv';spmv='v' in place
    # Instruction/traffic estimates reflect the checked 16x128 QK and paired
    # 16x64 PV structure. They are NOT exact compiled instruction counts.
    nodes=[];previous=[];phase_totals={k:0. for k in ('install_pack','qk','softmax_pack','pv','merge_export')}
    peak_coherent=0
    for h,work in enumerate(a['heads']):
        done=[]
        for core,x in enumerate(work):
            rows=c.br if c.ownership=='kv' else (c.r if c.ownership=='query_legacy' else c.t//8)
            scratch=4*c.br*c.bc+32*c.bc+4*c.br
            footprint=scratch+(0 if spmq else rows*128)+(0 if spmo else rows*256)+rows*8
            footprint+=(0 if spmk else c.bc*128)+(0 if spmv else c.bc*128)
            peak_coherent=max(peak_coherent,footprint)
            # Explicit uncertainty in residency. No addresses -> no claimed exact miss count.
            l1_pressure=max(0,footprint/HW.l1_bytes-1)
            set_pressure=footprint/(a['layout']['min_coherent_ways']*HW.sets*64)
            amplification=1+s.capacity_penalty*max(0,set_pressure-.5)
            miss_fraction=min(1,s.miss_floor+.12*l1_pressure)
            def memtime(n):
                return n*((1-miss_fraction)/128+miss_fraction/s.coherent_bytes_cycle)*amplification
            pairs=x['pairs'];updates=x['updates']
            qk_flops=pairs*128;pv_flops=pairs*128
            broadcasts=pairs/16 # 128 Q broadcasts / (16*128) QK tile
            q_extra=broadcasts*5 if spmq and c.q_adapter=='six_instruction' else 0
            # Each QK 16x128 tile reads K=16KiB, Q=2KiB logically.
            kbytes=pairs*8;qbytes=pairs
            # PV loads one 128B V row and 32B P per sixteen output rows.
            vbytes=pairs*8;pbytes=pairs*2
            qk_issue=(qk_flops/256+broadcasts+pairs/32+q_extra)/HW.commit_width
            qk_spm=((broadcasts if spmq else 0)+(pairs/32 if spmk else 0))/HW.spm_issue_per_cycle
            qk_memory=memtime((0 if spmk else kbytes)+(0 if spmq else qbytes))
            qk=max(qk_flops/s.effective_fp16_flops_cycle,qk_issue,qk_spm,qk_memory)
            o_ops=2*updates
            pv_issue=(pv_flops/256+pairs/16*5+updates*12)/HW.commit_width
            pv_spm=((pairs/16 if spmv else 0)+(o_ops if spmo else 0))/HW.spm_issue_per_cycle
            # Enforce finite load-window latency/throughput bound too.
            pv_spm=max(pv_spm,pv_spm*HW.spm_load_latency/HW.load_entries)
            pv=max(pv_flops/s.effective_fp16_flops_cycle,pv_issue,pv_spm,
                   memtime(pbytes+(0 if spmv else vbytes)+(0 if spmo else updates*512)))
            soft=max(pairs*.45/HW.commit_width,memtime(pairs*10+updates*32))
            packing=(x['q_rows_packed']*256+x['o_rows_initialized']*256)/s.coherent_bytes_cycle
            copies=x['copies']*64/s.copy_bytes_cycle
            install=packing+copies+(x['copies']*4+x['release_lines'])/HW.commit_width
            merge=(x['merge_rows']*(3*c.d+5)/64+
                   x['merge_rows']*a['partial_record_bytes']/s.merge_bytes_cycle+
                   x['merge_rows']*2*16)
            export=(c.t*c.d*2/c.cores)/s.coherent_bytes_cycle
            values=[('install_pack',install),('qk',qk),('softmax_pack',soft),('pv',pv),('merge_export',merge+export)]
            start_deps=([previous[core]] if previous and c.ownership=='query_legacy' else previous[:])
            deps=start_deps;last=None
            for name,cy in values:
                ident=f'{h}:{core}:{name}'
                # Explicit no-DSB scenario is an optimistic dependency-envelope,
                # not proof that this overlap is achievable by the existing guest.
                used_deps=start_deps if c.sync=='dependency' and name=='qk' else deps
                nodes.append(dict(id=ident,deps=used_deps,cycles=cy,name=name))
                phase_totals[name]+=cy
                deps=[ident];last=ident
            done.append(last)
            if c.sync=='dependency':done.append(f'{h}:{core}:install_pack')
        # Per-head synchronization in the specified reference schedule.
        previous=done
    # K packing is a shared initial stage, evenly partitioned across cores.
    cycles=phase_dag(nodes)+a['initial_k_pack_read_write_bytes']/8/s.coherent_bytes_cycle
    result=dict(us=cycles/(HW.clock_ghz*1000)*s.alpha,
                per_phase_core_cycle_demands=phase_totals,coherent_reuse_footprint=peak_coherent,
                uncertainty='scenario envelope, not a confidence interval or physical upper bound',
                schedule_approximation='phase demands aggregated per head; tile interleaving and finite-queue contention are unresolved; KV blockwise merging especially requires validation')
    if detailed:result['nodes']=nodes
    return result


def scenario_space():
    return [Scenario(coherent_bytes_cycle=b,effective_fp16_flops_cycle=f,
                     copy_bytes_cycle=cp,miss_floor=m,capacity_penalty=p)
            for b,f,cp,m,p in itertools.product((16,32),(128,256),(8,32),(.05,.2),(.5,2))]


def calibrate(observations):
    """Only caller-supplied training timings participate; positive global scale."""
    scored=[]
    for s in scenario_space():
        pred=[estimate(c,s)['us'] for c,y in observations]
        ratios=[y/p for p,(c,y) in zip(pred,observations)]
        alpha=math.exp(sum(math.log(x) for x in ratios)/len(ratios))
        error=math.sqrt(sum(math.log(p*alpha/y)**2 for p,(c,y) in zip(pred,observations))/len(pred))
        scored.append((error,replace(s,alpha=alpha)))
    scored.sort(key=lambda x:x[0])
    return scored[:8]


def predict(c, fitted):
    values=sorted(estimate(c,s)['us'] for error,s in fitted)
    n=len(values)
    if not n:raise ValueError('empty calibrated ensemble')
    return dict(median_us=(values[(n-1)//2]+values[n//2])/2,min_us=values[0],max_us=values[-1])


def candidates(t):
    for bc in (128,256,512):
        for place in ('cache','v','qv','ov','qo','qov'):
            for stride in ((0,512) if place not in ('cache',) else (0,)):
                yield Candidate(t=t,bc=bc,placement=place,stride=stride)
        for place in ('cache','v'):
            for r in (128,256,512):
                yield Candidate(t=t,ownership='query_legacy',placement=place,bc=bc,r=r)
        for lifetime in ('kv_group','all_heads'):
            for place in ('cache','kv'):
                yield Candidate(t=t,ownership='kv',placement=place,lifetime=lifetime,bc=bc)
