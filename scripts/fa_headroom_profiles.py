"""Predeclared, disjoint execution/on-chip-memory diagnostic bundles."""
import copy
import re

EXECUTION_FUS=(0,1,2,3,5,7)
VARIANTS={'control':(False,False),'execution':(True,False),
          'memory':(False,True),'combined':(True,True)}

def parameters(config,variant):
    execution,memory=VARIANTS[variant];p={}
    for i,cpu in enumerate(config['system']['cpu']):
        c=f'system.cpu[{i}]'
        if execution:
            for field in ('fetchWidth','decodeWidth','renameWidth','dispatchWidth',
                          'issueWidth','wbWidth','commitWidth','squashWidth'):
                p[c+'.'+field]=12  # Frozen build's MaxWidth; no rebuild needed.
            for field,value in dict(fetchQueueSize=128,numROBEntries=512,numIQEntries=256,
                                    numPhysIntRegs=512,numPhysCCRegs=2560,numPhysFloatRegs=512,
                                    numPhysVecRegs=512,numPhysVecPredRegs=256).items():p[c+'.'+field]=value
            for f in EXECUTION_FUS:
                fu=cpu['fuPool']['FUList'][f]
                assert fu['count']>0
                p[c+f'.fuPool.FUList[{f}].count']=16
                for j,op in enumerate(fu['opList']):
                    assert not any(x in op['opClass'] for x in ('Mem','Load','Store'))
                    p[c+f'.fuPool.FUList[{f}].opList[{j}].opLat']=1
                    p[c+f'.fuPool.FUList[{f}].opList[{j}].pipelined']=True
        if memory:
            for field in ('cacheLoadPorts','cacheStorePorts','spmLoadPorts','spmStorePorts'):p[c+'.'+field]=16
            for field in ('LQEntries','SQEntries'):p[c+'.'+field]=128
            # Active coherent/SPM memory-operation execution/AGU throughput is
            # classified with memory, not arithmetic. Do not enable dormant FUs.
            assert cpu['fuPool']['FUList'][9]['count']>0
            p[c+'.fuPool.FUList[9].count']=16
            l0=f'system.ruby.l0_cntrl{i}';l1=f'system.ruby.l1_cntrl{i}'
            for field in ('request_latency','response_latency','spm_copy_request_latency',
                          'spm_load_request_latency','spm_request_latency','recycle_latency'):
                p[l0+'.'+field]=1
            for field in ('l1_request_latency','l1_response_latency','spm_access_response_latency',
                          'spm_response_latency','recycle_latency'):p[l1+'.'+field]=1
            p[l0+'.transitions_per_cycle']=128;p[l1+'.transitions_per_cycle']=128
            for name,value in (('dataMissCredits',128),('instMissCredits',32),('writeCredits',64)):
                p[l0+'.'+name+'.capacity']=value
            p[l1+'.missCredits.capacity']=128;p[l1+'.writeCredits.capacity']=64
    if memory:
        ruby=config['system']['ruby']
        for name,obj in ruby.items():
            if re.fullmatch(r'l2_cntrl\d+',name):
                for field in ('l2_request_latency','l2_response_latency','recycle_latency'):
                    p['system.ruby.'+name+'.'+field]=1
                p['system.ruby.'+name+'.transitions_per_cycle']=128
            elif re.fullmatch(r'dir_cntrl\d+',name):
                for field in ('directory_latency','recycle_latency'):p['system.ruby.'+name+'.'+field]=1
                p['system.ruby.'+name+'.transitions_per_cycle']=128
        p['system.ruby.microLlcMissCredits.capacity']=512
        p['system.ruby.microLlcWriteCredits.capacity']=128
        p['system.ruby.network.buffers_per_data_vc']=16
        p['system.ruby.network.buffers_per_ctrl_vc']=8
    return p

def leaves(obj,path=''):
    if isinstance(obj,dict):
        for k,v in obj.items():yield from leaves(v,path+'.'+k if path else k)
    elif isinstance(obj,list):
        for i,v in enumerate(obj):yield from leaves(v,path+f'[{i}]')
    else:yield path,obj

def expected_changes(config,variant):
    p=parameters(config,variant)
    if VARIANTS[variant][1]:
        # --link-width-bits=1024 sets bytes/flit consistently throughout Garnet.
        for path,value in leaves(config):
            if path.startswith('system.ruby.network.') and path.rsplit('.',1)[-1] in ('width','ni_flit_size'):
                assert value==16,(path,value)
                p[path]=128
    return p

def audit_objects(old,new,variant):
    expected=expected_changes(old,variant);a=dict(leaves(old));b=dict(leaves(new))
    assert a.keys()==b.keys(),'Configuration shape changed'
    assert set(expected)<=set(a),'Expected parameter missing'
    changed=[]
    for k,v in a.items():
        wanted=expected.get(k,v)
        assert b[k]==wanted,(k,v,b[k],wanted)
        if v!=b[k]:changed.append(dict(path=k,old=v,new=b[k]))
    return changed

def set_path(obj,path,value):
    parts=re.findall(r'[^.\[\]]+|\[\d+\]',path);target=obj
    for part in parts[:-1]:target=target[int(part[1:-1])] if part.startswith('[') else target[part]
    last=parts[-1]
    if last.startswith('['):target[int(last[1:-1])]=value
    else:target[last]=value

def transformed(config,variant):
    out=copy.deepcopy(config)
    for k,v in expected_changes(config,variant).items():set_path(out,k,v)
    return out
