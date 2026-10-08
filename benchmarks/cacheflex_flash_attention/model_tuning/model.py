"""Loop-count/footprint model + empirically calibrated ranking, not cycle attribution.

Exact: allocation, loop/copy counts. Approximate: reuse-boundary pressure and
traffic between caches. Payload counts NEVER imply LLC misses or DRAM bytes.
"""
import itertools,json,math
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
B=ROOT/'results/cacheflex_flash_attention'
BR=(16,32,64)
BC=(128,256,512,1024)
QR=(64,128,192,256,384,512,768,1024)
FAMILIES=('q_cache','kv_cache','q_spm','kv_spm')
FEATURES=('compute_gflop','stream_mib','copy_mib','accumulator_mib',
          'packing_mib','reduction_mib','barrier_k','l1_pressure_mib','l2_pressure_mib')

def label(c):
    return f"t{c['t']}_{c['family']}_br{c['br']}_bc{c['bc']}_r{c['r']}"

def describe(c):
    t,br,bc,r=(c[k] for k in ('t','br','bc','r'));p=8;d=64;sets=1024
    kv=c['family'].startswith('kv_');spm=c['family'].endswith('_spm')
    reasons=[]
    if c['family'] not in FAMILIES:reasons.append('unknown family')
    if t<=0 or t%1024:reasons.append('T must give each core whole 128-token K panels')
    if br<=0 or br%16 or r<br or r%br:reasons.append('query grouping must respect 16-row microkernel')
    if bc<=0 or bc%128 or bc>sets or sets%bc:
        reasons.append('current SPM PV address increment cannot cross a 1024-set boundary; Bc must divide1024')
    local=t//p;krange=local if kv else t;qrange=t if kv else local
    if r>qrange:reasons.append('redundant query group larger than per-worker query domain')
    if bc>krange:reasons.append('redundant Bc larger than per-worker KV domain')
    staged_rows=(local if kv else bc) if spm else 0
    max_ways=2*math.ceil(staged_rows/sets)
    if max_ways>6:reasons.append('more than six live SPM slots per set')
    if reasons:return dict(valid=False,reasons=reasons)
    groups=math.ceil(qrange/r);tiles=math.ceil(krange/bc)
    spm_bytes=staged_rows*128
    # Lazy per-slot claims: NOT rounded whole-way subtraction. Query staging
    # occupies two slots only in the first Bc sets; untouched slots stay coherent.
    occupancy=[2*(staged_rows//sets+(i<staged_rows%sets)) for i in range(sets)]
    coherent_capacity=512*1024-spm_bytes
    q_state=128*r+256*r+8*r
    scratch=4*br*bc+32*bc+2048
    hot=q_state+scratch+128*bc*(1+int(not spm))
    copies=(2*t if kv else 2*t*p*groups) if spm else 0
    # Per-core useful panel traversal, assuming reuse inside the query group.
    # L1-pressure proxy below captures possible loss of that reuse, not a miss count.
    stream=groups*krange*128*(1+int(not spm))
    acc=qrange*tiles*(16*d+16)  # two output read/write pairs + m/l proxy
    packing=4*t*d/p+4*qrange*d
    reduction=t*(4*d+8) if kv else 0
    features=[4*t*t*d/p/1e9,stream/2**20,copies*64/p/2**20,acc/2**20,
              packing/2**20,reduction/2**20,(2*groups if kv else 0)/1000,
              stream/2**20*max(0,hot/(64*1024)-1),
              stream/2**20*max(0,hot/coherent_capacity-1)]
    return dict(valid=True,spm_payload_bytes_per_core=spm_bytes,
        max_spm_ways_per_set=max_ways,min_coherent_ways_per_set=8-max(occupancy),
        coherent_capacity_bytes=coherent_capacity,coherent_hot_bytes=hot,
        q_and_state_bytes=q_state,scratch_bytes=scratch,
        spmcp_count=copies,groups_per_core=groups,kv_tiles_per_group=tiles,
        l1_pressure_ratio=hot/(64*1024),l2_pressure_ratio=hot/coherent_capacity,
        logical_stream_bytes_per_core=stream,features=dict(zip(FEATURES,features)),
        caveat='Logical traversal/pressure proxies, not measured cache misses, DRAM bytes or additive latency')

def candidates(t):
    good=[];bad=[]
    for family,br,bc,r in itertools.product(FAMILIES,BR,BC,QR):
        c=dict(t=t,family=family,br=br,bc=bc,r=r);m=describe(c)
        (good if m['valid'] else bad).append(dict(case=c,model=m))
    return good,bad

def observations():
    rows=[]
    old=B/'kv_partition_20260925/experiment_v3/status.json'
    for x in json.loads(old.read_text())['completed']:
        c=x['case']
        if c['n']!=8 or c['l2']!=512 or c['stage']!='target':continue
        case=dict(t=c['t'],family=('kv' if c['kv'] else 'q')+('_spm' if c['spm'] else '_cache'),
                  br=c['br'],bc=c['bc'],r=c['br']*c['group'])
        rows.append(dict(case=case,roi_us=x['roi_us'],source=x['path'],packer='original'))
    new=B/'cache_tuning_20260925/experiment_export_safe_20260926/status.json'
    if new.exists():
        for x in json.loads(new.read_text())['completed']:
            c=x['case']
            if c['t']<4096:continue
            case=dict(t=c['t'],family='q_cache',br=c['br'],bc=c['bc'],r=c['br']*c['g'])
            rows.append(dict(case=case,roi_us=x['roi_us'],source=x['path'],packer=c['mode']))
    return rows

def fit(rows):
    # Equal relative-error weight; nonnegative cost coefficients. Coefficients
    # are ranking surrogates with overlapping features, NOT causal fractions.
    x=np.array([[describe(a['case'])['features'][k] for k in FEATURES] for a in rows])
    y=np.array([a['roi_us'] for a in rows]);scale=np.maximum(x.max(axis=0),1e-12)
    coeff,_=nnls((x/scale)/y[:,None],np.ones(len(y)),maxiter=10000)
    return dict(coefficients=dict(zip(FEATURES,(coeff/scale).tolist())),
        samples=len(rows),limitations='Sparse historical calibration; old/new packing mixed; not a physical critical-path simulator')

def predict(c,calibration):
    f=describe(c)['features']
    return sum(f[k]*calibration['coefficients'][k] for k in FEATURES)

def holdout(rows):
    train=[r for r in rows if r['case']['t']==4096]
    test=[r for r in rows if r['case']['t']==16384]
    m=fit(train)
    records=[dict(case=r['case'],predicted_us=predict(r['case'],m),measured_us=r['roi_us']) for r in test]
    a=np.array([r['predicted_us'] for r in records]);b=np.array([r['measured_us'] for r in records])
    return dict(training='T4096 only',held_out='historical T16384',records=records,
        median_relative_error=float(np.median(np.abs(a-b)/b)),
        spearman=float(spearmanr(a,b).statistic),
        caveat='Diagnostic of extrapolation; subsequent candidate selection may fit all available data')

def select(t,rows,seen,per_family=2,neighbors=None):
    calibration=fit(rows);pool,_=candidates(t);selected=[]
    for family in FAMILIES:
        xs=[a['case'] for a in pool if a['case']['family']==family and label(a['case']) not in seen]
        if neighbors:
            w=neighbors[family]
            adjacent={}
            for k,values in (('br',BR),('bc',BC),('r',QR)):
                i=values.index(w[k]);adjacent[k]=values[max(0,i-1):i]+values[i+1:i+2]
            near=[c for c in xs if sum(c[k]!=w[k] for k in ('br','bc','r'))==1
                  and all(c[k]==w[k] or c[k] in adjacent[k] for k in adjacent)]
            if near:xs=near
        xs.sort(key=lambda c:predict(c,calibration))
        if not xs:continue
        selected.append(dict(case=xs[0],predicted_us=predict(xs[0],calibration),reason='model-ranked candidate'))
        if per_family>1 and len(xs)>1:
            # Force a distinct reuse/footprint point instead of trusting a sparse
            # fitted model to prune all alternatives. Smallest distance to an
            # L1/L2 capacity transition, preferring different Bc/R.
            alternatives=[c for c in xs[1:] if c['r']!=xs[0]['r'] or c['bc']!=xs[0]['bc']]
            if not alternatives:alternatives=xs[1:]
            def boundary(c):
                d=describe(c)
                return min(abs(math.log2(d['l1_pressure_ratio'])),abs(math.log2(d['l2_pressure_ratio'])))
            c=min(alternatives,key=boundary)
            selected.append(dict(case=c,predicted_us=predict(c,calibration),reason='capacity-boundary exploration'))
    return dict(calibration=calibration,selected=selected)

if __name__=='__main__':
    rows=observations();print(json.dumps(dict(holdout=holdout(rows),
        screen=select(4096,rows,set())),indent=2))
