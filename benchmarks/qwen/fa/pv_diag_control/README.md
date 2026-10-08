# FA-48 inert-startup controls

All original 18 FA48 cases passed, but flags1..5 share identical ROI-start and
K-packing timelines while flag0 differs BEFORE any intervention. Therefore their
raw effect against flag0 includes a startup-state confound. Preserve those cases.

This continuation reuses the EXACT original guest and simulator, with guest flag2
but engine environment flag0. Guest flag2 leaves probe/fence conditions unchanged;
zero service is selected only by the engine environment. Cases: inert async256
and spm_inert ordinary SPM, each T256 then T1024. All original full validation
retained with expected service/overlap based on explicit engine_flags=0. No model
or guest edits. Compare inert to zero_service: command/config/guest flag identical,
only engine service changes. Require inert ROI start and pre-intervention phase
vectors identical to earlier nonzero-flag cases before causal conclusions.
Also report ordinary SPM with the same nonzero guest flag, to expose control-state
sensitivity. Startup effects are not attributed to PV service or CPU interference.
Two workers, existing broker10GiB lease/2CPUs each,20%CPU/RAM and64GiB headroom.
