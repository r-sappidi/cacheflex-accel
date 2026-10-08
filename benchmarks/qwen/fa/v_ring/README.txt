FA-36: V-only optimistic rotation with legacy query-task ownership.
Prior FA-34 tested K+V movement and complementary query slices; best full T4096
ring remained slower than V-only. FA-35 tunes placements, not V-only rotation.
New: preserve legacy h/group/core assignment, coherent packed K/Q/O, Br16 and
R256 at T4096. V uses existing hook slots ways0/1 sets768..1023 (baseline uses
sets0..255). Cyclic ordinary and ring arms share this mapping and synchronization.
Modes1/16/17: ordinary legacy V; cyclic V with copies; cyclic V with rotations.
No new simulator or hardware sync instruction: reuse accepted FA-34 optimistic
hook; DSB plus two pthread barriers per step retain synchronization costs.
Every group wave visits all8 tiles to forward even future panels; arithmetic
retains legacy per-group causal endpoint. K packing remains coherent, once/ROI.
T1024 uses R128/Bc128 (8 groups) for a fully occupied gate/pilot; T4096 uses
R256/Bc256 (16 groups), preserving target winning task ownership exactly.
Nine cases: three modes each at T1024 Hq2, T1024 Hq14, T4096 Hq14; Hkv2/d64.
Full independent cyclic/ordinary oracle, dense small check, exact cyclic pair
output, export hashes, copies/rotation bytes/work/config/finite-resource checks.
Fail closed. Existing accepted hook directed tests reused; hashes frozen.
Decision: full ROI vs matched cyclic ordinary AND same-new-ELF V baseline;
retain best FA-35 private timing as additional context, not a matched ELF ratio.
No finite-link or hardware-sync benefit claim. No extra backing SRAM.
One simulator,8GiB lease/6GiB initial AS with admitted growth; coordinator2GiB,
two allowed CPUs, shared20% budget and64GiB headroom. Host RSS heartbeat plus
namespace-local RSS used conservatively in restricted process namespaces.
