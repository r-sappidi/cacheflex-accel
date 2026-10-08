"""FA-35 bounded equal-family search; exact structural filters, no failed timing ranks."""
import itertools,math
NAMES=['legacy_cache','legacy_v','head_cache','head_v','head_qo','head_qov','head_qv','head_ov','legacy_l3','kv_cache','kv_group','kv_both']

def case(mode,br=16,bc=256,r=None,t=4096,hq=14):
    if r is None:r=256 if mode in (0,1,8) else (64 if mode>=9 else 512)
    return dict(t=t,mode=mode,br=br,bc=bc,r=r,hq=hq,hkv=2,causal=True)

def name(c):return f"t{c['t']}_q{c['hq']}_{NAMES[c['mode']]}_br{c['br']}_bc{c['bc']}_r{c['r']}"
def arithmetic_class(c):return 'kv' if c['mode']>=9 else 'query'
def refkey(c):return (c['t'],c['bc'],c['hq'],c['hkv'],c['causal'],arithmetic_class(c))

def describe(c):
    t,br,bc,r,hq,hkv,mode=(c[k] for k in ('t','br','bc','r','hq','hkv','mode'))
    assert t in (256,512,1024,2048,4096) and br in (16,32,64) and bc in (128,256,512)
    assert mode in range(12) and hkv==2 and hq in (2,14) and c['causal']
    assert r>0 and r<=512 and r%br==0 and t%r==0 and t%bc==0
    if 2<=mode<=7:assert r<=t//8 and (t//8)%r==0 and r%(2*br)==0
    if mode>=9:assert t%2048==0 and bc<=t//16 and (t//16)%bc==0
    tasks=[0]*8;updates=[0]*8;copies=[0]*8
    if mode in (0,1,8):
        groups=t//r
        for h in range(hq):
            for g in range(groups):
                core=(h*groups+(g-h)%groups)%8;end=math.ceil((g+1)*r/bc)*bc
                tasks[core]+=1;updates[core]+=r*(end//bc)
                if mode in (1,8):copies[core]+=2*end
    elif mode<9:
        q=mode in (4,5,6);o=mode in (4,5,7);v=mode in (3,5,6,7)
        for core in range(8):
            for sl in range(t//8//r):
                rr=[core*(t//16)+sl*(r//2),(15-core)*(t//16)+sl*(r//2)]
                tasks[core]+=hq
                updates[core]+=hq*sum(br*math.ceil((qb+br)/bc) for a in rr for qb in range(a,a+r//2,br))
                copies[core]+=hq*(int(q)*2*r+int(o)*4*r+int(v)*2*math.ceil((rr[1]+r//2)/bc)*bc)
    else:
        for core in range(8):
            starts=[core*(t//16),(15-core)*(t//16)]
            tasks[core]=hq*(t//r)
            updates[core]=hq*sum(br for qb in range(0,t,br) for a in starts for k in range(a,a+t//16,bc) if k<qb+br)
            copies[core]=hkv*(t//8)*4 if mode>=10 else 0
    slots=set()
    if mode in (1,8):slots={(w,s) for w in range(2) for s in range(bc)}
    elif 2<=mode<=7:
        if mode in (4,5,6):slots|={(4,s) for s in range(2*r)}
        if mode in (4,5,7):slots|={(w,s) for w in range(4) for s in range(r)}
        if mode in (3,5,6,7):slots|={(w,s) for w in range(2) for s in range(512,512+bc)}
    elif mode>=10:
        heads=hkv if mode==11 else 1;local=t//8
        slots={(w,s) for w in range(4) for s in range(heads*local//128*64)}|{(w,s) for w in (4,5) for s in range(heads*local)}
    assert all(0<=w<6 and 0<=s<1024 for w,s in slots)
    return dict(tasks=tasks,updates=updates,copies=copies,commands=sum(updates) if mode==8 else 0,
                spm_bytes=len(slots)*64,coherent_bytes=524288-len(slots)*64,
                gemm_flops=256*bc*sum(updates),copy_payload_bytes=sum(copies)*64,
                max_mean_work=max(updates)*8/sum(updates),
                useful_pairs=hq*t*(t+1)//2,computed_pairs=bc*sum(updates),
                note='counts/claims exact; cache misses, overlap and timing not predicted')

def anchors():return [case(m) for m in (6,7,0,1,2,3,4,5,8,9,10,11)]

def axial():
    result=[];rejected=[]
    for m in range(12):
        a=case(m)
        for dim,values in [('bc',(128,256,512)),('br',(16,32,64)),('r',(128,256,512) if m<9 else (16,64,128,256))]:
            for v in values:
                c=dict(a,**{dim:v})
                if c==a:continue
                try:describe(c)
                except AssertionError:
                    rejected.append(dict(case=c,reason='ownership/tile alignment or physical SPM bound'));continue
                if c not in result:result.append(c)
    return result,rejected

def gate(c):
    c=dict(c)
    if c['mode']>=9:
        c['t']=max(2048,16*c['bc']);c['hq']=2
    else:
        c['t']=max(256,16*c['br'],c['bc']);c['r']=min(c['r'],c['t']//8) if c['mode']>=2 and c['mode']<=7 else min(c['r'],c['t'])
    describe(c);return c

def interactions(rows,limit=4):
    """Measured one-factor effects select crossed settings; equal max budget/family."""
    chosen=[];audit=[]
    for m in range(12):
        seen={name(r['case']) for r in rows if r['case']['mode']==m and r['case']['t']==4096 and r['case']['hq']==14}
        data=[r for r in rows if name(r['case']) in seen]
        assert data,('missing family',m)
        winner=min(data,key=lambda r:r['roi_us']);a=case(m)
        levels={}
        for dim in ('br','bc','r'):
            best={}
            for row in data:
                v=row['case'][dim];best[v]=min(best.get(v,float('inf')),row['roi_us'])
            levels[dim]=[v for v,_ in sorted(best.items(),key=lambda x:x[1])[:2]]
        pool=[]
        for br,bc,r in itertools.product(levels['br'],levels['bc'],levels['r']):
            c=case(m,br,bc,r)
            if name(c) in seen:continue
            try:describe(c)
            except AssertionError:continue
            score=1
            for dim in ('br','bc','r'):
                subset=[x['roi_us'] for x in data if x['case'][dim]==c[dim]]
                score*=min(subset)/winner['roi_us']
            pool.append((score,c))
        pool.sort(key=lambda x:(x[0],name(x[1])))
        selected=[c for _,c in pool[:limit]];chosen+=selected
        audit.append(dict(mode=m,measured_winner=winner['name'],selected=selected,
                          method='cross top two measured levels per axis; multiplicative ordering heuristic, not a latency prediction'))
    return chosen,audit

def neighbors(rows,limit=2):
    chosen=[];audit=[]
    for m in range(12):
        data=[r for r in rows if r['case']['mode']==m and r['case']['t']==4096 and r['case']['hq']==14]
        winner=min(data,key=lambda x:x['roi_us']);w=winner['case'];seen={name(x['case']) for x in data};pool=[]
        for dim,levels in [('bc',[128,256,512]),('br',[16,32,64]),('r',[128,256,512] if m<9 else [16,64,128,256])]:
            for value in levels:
                if value==w[dim] or abs(levels.index(value)-levels.index(w[dim]))!=1:continue
                c=dict(w,**{dim:value})
                if name(c) in seen:continue
                try:describe(c)
                except AssertionError:continue
                pool.append(c)
        selected=pool[:limit];chosen+=selected
        audit.append(dict(mode=m,winner=winner['name'],selected=selected,untested_neighbors=pool[limit:],
                          scope='bounded neighbor check, not exhaustive global optimality'))
    return chosen,audit
