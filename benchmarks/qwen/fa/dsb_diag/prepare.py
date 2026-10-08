from pathlib import Path
import json,hashlib,shutil
R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent
B=R/'results/cacheflex_flash_attention/dsb_diag_20261003/build_v1';B.mkdir(parents=True,exist_ok=False)
S=R/'results/retained_protocol_restart_20261002/gem5/src/cpu/o3'
shutil.copy2(H/'observer.hh',B/'observer.hh')
for name in ['commit','lsq','rename','mem_dep_unit']:
 s=(S/(name+'.cc')).read_text();(B/(name+'.original.cc')).write_text(s)
 # Place helper after normal includes, immediately before gem5 namespace.
 anchor='namespace gem5\n';assert anchor in s;s=s.replace(anchor,'#include "observer.hh"\n\n'+anchor,1)
 if name=='commit':
  a=s.index('Commit::Commit(');b=s.index('\n{',a)+2;s=s[:b]+'\n    dsbdiag::state(cpu);'+s[b:]
  needle='        if (inst_num > 0 || iewStage->hasStoresToWB(tid)) {'
  s=s.replace(needle,'''        if (dsbdiag::enabled() && dsbdiag::isDsb(head_inst)) {
            dsbdiag::begin(cpu,head_inst);
            bool waiting=iewStage->hasStoresToWB(tid);
            dsbdiag::finish(cpu,head_inst,inst_num,waiting);
        }
'''+needle)
  needle='    if (inst->isFullMemBarrier()) {';s=s.replace(needle,'    if(dsbdiag::enabled() && dsbdiag::isDsb(inst))++dsbdiag::state(cpu).stats->retired;\n'+needle)
 if name=='lsq':
  needle='LSQ::hasStoresToWB(ThreadID tid)\n{'
  s=s.replace(needle,needle+'''
    auto &obs=dsbdiag::state(cpu);
    if(dsbdiag::enabled() && obs.probe && obs.tid==tid){
        std::set<InstSeqNum> counted[3];
        auto visit=[&](LSQUnit &u,bool spm){
            for(auto it=u.storeQueue.begin();it!=u.storeQueue.end();++it){
                if(!it->valid() || it->instruction()->seqNum>=obs.probe || it->completed())continue;
                auto inst=it->instruction();unsigned dom=dsbdiag::domain(inst);
                unsigned index=dom==4?2:(spm?1:0);counted[index].insert(inst->seqNum);
                if(it->canWB())obs.mask|=1u<<index;
            }
            for(auto it=u.loadQueue.begin();it!=u.loadQueue.end();++it){
                if(!it->valid())continue;auto inst=it->instruction();
                if(inst->seqNum<obs.probe && !inst->isExecuted()){
                    unsigned dom=dsbdiag::domain(inst);counted[dom==4?2:(spm?1:0)].insert(inst->seqNum);
                }
            }
            for(const auto &[sn,entry]:u.spmShadowTable)if(sn<obs.probe && !entry.ready){
                counted[2].insert(sn);if(entry.retired)obs.mask|=4;
            }
            for(const auto &[sn,entry]:u.detachedLoads)if(sn<obs.probe)counted[spm?1:0].insert(sn);
        };
        visit(thread.at(tid),false);visit(spmThread.at(tid),true);
        if(cfReduction::busy(cpu->system,cpu->tcBase(tid)->contextId()))obs.mask|=2;
        for(unsigned i=0;i<3;++i)obs.counts[i]=counted[i].size();
    }
''')
 if name=='rename':
  needle='            serializeAfter(insts_to_rename, tid);';s=s.replace(needle,'            if(dsbdiag::enabled())dsbdiag::state(cpu).serialize=dsbdiag::isDsb(inst)?inst->seqNum:0;\n'+needle)
  needle='        DPRINTF(Rename,"[tid:%i] Stall: Serialize stall and ROB is not "'
  s=s.replace(needle,'''        if(dsbdiag::enabled()){
            auto &d=dsbdiag::state(cpu);auto front=serializeInst[tid];
            if(d.serialize && front && front->seqNum>d.serialize && !front->isSquashed() && d.renameTick!=curTick()){
                d.renameTick=curTick();++d.stats->renameWaitSamples;
                if(front->isMemRef() && dsbdiag::domain(front)==1){
                    ++d.stats->renameCoherentFrontSamples;
                    if(d.front!=front->seqNum){d.front=front->seqNum;++d.stats->renameCoherentFrontUnique;}
                }
            }
        }
'''+needle)
  s=s.replace('        serializeInst[tid] = NULL;','        if(dsbdiag::enabled())dsbdiag::state(cpu).serialize=0;\n        serializeInst[tid] = NULL;')
 if name=='mem_dep_unit':
  s=s.replace('    id = tid;','    id = tid;\n    dsbdiag::memdepCpu[_name]=cpu;')
  needle='    // If no store entry, then instruction can issue as soon as the registers'
  s=s.replace(needle,'''    if(dsbdiag::enabled()){
        bool dsb=false;for(auto &e:store_entries)if(dsbdiag::isDsb(e->inst))dsb=true;
        if(dsb){auto &d=dsbdiag::state(dsbdiag::memdepCpu.at(_name));auto domain=dsbdiag::domain(inst);
            if(domain==4)++d.stats->youngerBridgeDependencies;
            else if(domain==2)++d.stats->youngerSpmDependencies;
            else ++d.stats->youngerCoherentDependencies;
        }
    }
'''+needle)
 (B/(name+'.cc')).write_text(s)
(B/'source_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in B.iterdir() if p.is_file()},indent=2))
print(B)
