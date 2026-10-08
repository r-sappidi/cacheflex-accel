"""Isolated observational network-interface overlay; retained sources untouched."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[4]
B=R/'results/cacheflex_flash_attention/packet_diag_20261004/build_v1'
S=R/'results/retained_protocol_restart_20261002/gem5/src/mem/ruby/network/garnet/NetworkInterface.cc'
def prepare():
 B.mkdir(parents=True,exist_ok=False)
 s=S.read_text();(B/'NetworkInterface.original.cc').write_text(s)
 def insert(anchor,text):
  nonlocal s
  assert s.count(anchor)==1,anchor
  s=s.replace(anchor,text+anchor)
 insert('        for (int i = 0; i < num_flits; i++) {', '''        if (DTRACE(CacheFlexTiming)) {
            using namespace MESI_Three_Level_SPM;
            if (auto *m = dynamic_cast<const ResponseMsg*>(new_net_msg_ptr)) {
                CFT_MSG("NI_CLASS", m,
                    "packet=%d kind=response type=%s spm_dst=%#x born=%llu source_ready=%llu",
                    packet_id, CoherenceResponseType_to_string(m->getType()),
                    m->getDstSPMAddr(), m->getTime(), msg_ptr->getLastEnqueueTime());
            } else if (auto *m = dynamic_cast<const RequestMsg*>(new_net_msg_ptr)) {
                CFT_MSG("NI_CLASS", m,
                    "packet=%d kind=request type=%s spm_dst=%#x born=%llu source_ready=%llu",
                    packet_id, CoherenceRequestType_to_string(m->getType()),
                    m->getDstSPMAddr(), m->getTime(), msg_ptr->getLastEnqueueTime());
            }
        }
''')
 insert('   int vc = oPort->vcRoundRobin();', '''   // Read-only snapshots of every ready output VC before arbitration.
   // Only pure readiness/credit/order queries; no queue/resource mutation.
   if (DTRACE(CacheFlexTiming)) {
       for (int q = 0; q < niOutVcs.size(); ++q) {
           int vn = get_vnet(q);
           if (!oPort->isVnetSupported(vn) || !niOutVcs[q].isReady(curTick()))
               continue;
           bool older = false;
           if (m_net_ptr->isVNetOrdered(vn)) {
               int base = vn * m_vc_per_vnet;
               for (int j = base; j < base + m_vc_per_vnet; ++j) {
                   if (m_net_ptr->sameRequestVCClass(vn, q, j, m_vc_per_vnet) &&
                       niOutVcs[j].isReady(curTick()) &&
                       m_ni_out_vcs_enqueue_time[j] < m_ni_out_vcs_enqueue_time[q])
                       older = true;
               }
           }
           CFT_FLIT("NI_STATE", niOutVcs[q].peekTopFlit(),
               "credit=%d older=%d", outVcState[q].has_credit(), older);
       }
   }
''')
 insert('                    iPort->m_stall_queue.push_back(t_flit);', '''                    CFT_FLIT("NI_EJECT_STALL", t_flit,
                        "quota=%d direct=%d", iPort->messageEnqueuedThisCycle,
                        m_net_ptr->directReductionEnabled());
''')
 (B/'NetworkInterface.cc').write_text(s)
 (B/'source_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [S,*B.glob('*.cc'),Path(__file__).resolve()]},indent=2)+'\n')
if __name__=='__main__':prepare()
