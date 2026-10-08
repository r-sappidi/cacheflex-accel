from pathlib import Path
import shutil,json,hashlib
R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent
B=R/'results/cacheflex_flash_attention/spm_isolation_20261004/build_v1'
S=R/'results/retained_protocol_restart_20261002/gem5/src/cpu/o3/lsq_unit.cc'
B.mkdir(parents=True,exist_ok=False);shutil.copy2(H/'observer.hh',B/'observer.hh')
s=S.read_text();(B/'lsq_unit.original.cc').write_text(s)
s=s.replace('namespace gem5\n','#include "observer.hh"\n\nnamespace gem5\n',1)
s=s.replace('    cpu = cpu_ptr;','    cpu = cpu_ptr;\n    spmisolation::state(cpu);',1)
old='''        const auto used = storeQueue.size() +
            (boundedTracking && isSpmLsq ? spmShadowTable.size() : 0);
        assert(used <= storeQueue.capacity());
        if (boundedTracking && used > stats.boundedMaxStoreOccupancy.value())
            stats.boundedMaxStoreOccupancy = used;
        return storeQueue.capacity() - used;'''
new='''        const auto total = storeQueue.size() +
            (boundedTracking && isSpmLsq ? spmShadowTable.size() : 0);
        size_t copies=0;
        if(boundedTracking && isSpmLsq)
            for(const auto &[sn,e]:spmShadowTable) if(e.isCopy) ++copies;
        const auto cap=storeQueue.capacity();
        const auto budget=boundedTracking && isSpmLsq ? spmisolation::budget() : 0;
        const auto used=total-(budget?copies:0);
        assert(used<=cap);
        assert(!budget || copies<=budget);
        unsigned free=cap-used;
        if(budget)free=std::min<unsigned>(free,budget-copies);
        if (boundedTracking && used > stats.boundedMaxStoreOccupancy.value())
            stats.boundedMaxStoreOccupancy = used;
        if(boundedTracking && isSpmLsq){
            auto &o=spmisolation::state(cpu);
            if(copies>o.stats->maxCopyShadows.value())o.stats->maxCopyShadows=copies;
            if(total>o.stats->maxTotalOccupancy.value())o.stats->maxTotalOccupancy=total;
            if(o.tick!=curTick()){
                o.tick=curTick();++o.stats->samples;
                if(total>=cap){++o.stats->baselineFull;if(free)++o.stats->relieved;}
            }
        }
        return free;'''
assert s.count(old)==1;s=s.replace(old,new)
(B/'lsq_unit.cc').write_text(s)
(B/'source_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [S,H/'prepare.py',H/'observer.hh',B/'lsq_unit.cc',B/'observer.hh']},indent=2))
