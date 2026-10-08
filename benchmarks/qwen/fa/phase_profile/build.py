import json,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent
ROOT=R/'results/cacheflex_flash_attention/phase_profile_20261005';B=ROOT/'build_v1';B.mkdir(parents=True)
SRC=R/'results/retained_protocol_restart_20261002/gem5/src';OLD=R/'results/cacheflex_flash_attention/pv_diag_20261005/build_v1'
sources={}
for name,p in [('pseudo_inst.cc',OLD/'pseudo_inst.cc'),('hook.hh',OLD/'hook.hh'),('Sequencer.cc',SRC/'mem/ruby/system/Sequencer.cc'),('lsq_unit.cc',SRC/'cpu/o3/lsq_unit.cc')]:
 sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();s=p.read_text()
 if name=='Sequencer.cc':
  s='#include "phase_api.hh"\n'+s
  assert s.count('++m_spmcpRequestsIssued;')==2
  s=s.replace('++m_spmcpRequestsIssued;','++m_spmcpRequestsIssued;\n        faCopyBegin(m_version,reinterpret_cast<uintptr_t>(pkt));')
  assert s.count('++m_spmcpRequestsCompleted;')==1
  s=s.replace('++m_spmcpRequestsCompleted;','++m_spmcpRequestsCompleted;\n        faCopyEnd(m_version,reinterpret_cast<uintptr_t>(srequest->pkt));')
 if name=='lsq_unit.cc':
  s='#include "phase_api.hh"\n'+s
  n='LSQUnit::recvTimingResp(PacketPtr pkt)\n{';assert n in s;s=s.replace(n,n+'\n    faMemEnd(cpu->cpuId(),reinterpret_cast<uintptr_t>(pkt));')
  n='++stats.spmRawBlockedUniqueLoads;';assert s.count(n)==1;s=s.replace(n,n+'\n                faRawBegin(cpu->cpuId(),load_inst->seqNum);')
  n='spmRawWaitStart.erase(rawWait);';assert s.count(n)==2;s=s.replace(n,'faRawEnd(cpu->cpuId(),load_inst->seqNum,false);\n            '+n)
  n='if (it->first > squashed_num)\n                it = spmRawWaitStart.erase(it);';assert n in s;s=s.replace(n,'if (it->first > squashed_num) {\n                faRawEnd(cpu->cpuId(),it->first,true);\n                it = spmRawWaitStart.erase(it);\n            }')
  n='++stats.spmPortBlockedEvents;';assert s.count(n)==2;s=s.replace(n,n+'\n            faPortBlock(cpu->cpuId(),true);')
  a=s.index('LSQUnit::trySendPacket(');b=s.index('\nvoid\nLSQUnit::startStaleTranslationFlush',a);body=s[a:b]
  n='    if (ret) {\n';assert body.count(n)==1;body=body.replace(n,n+'        if(isLoad && data_pkt->needsResponse())faMemBegin(cpu->cpuId(),reinterpret_cast<uintptr_t>(data_pkt),data_pkt->getSize(),isSpmLsq);\n')
  n='    } else {\n        if (cache_got_blocked) {';assert n in body;body=body.replace(n,'    } else {\n        if(isLoad && !isSpmLsq)faPortBlock(cpu->cpuId(),false);\n        if (cache_got_blocked) {')
  s=s[:a]+body+s[b:]
 (B/name).write_text(s)
for name in ['phase_api.hh','observer.hh']:shutil.copy2(H/name,B/name)
(B/'original_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
print('Isolated passive observer sources prepared; guest unchanged')
