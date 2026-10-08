# Llama-shaped benchmarks

[FA iterative tuning](fa/iterative/README.txt) implements FA-41 for Llama-3.2-1B:
T4096, batch1, 32 query heads, eight KV heads, head dimension64, eight CPU cores.
Cache and rolling-V private SPM are independently improved from the transferred
Qwen kernel through one-variable, counter-guided local iterations.

Current status: `results/ordered_fa_followup_20261005/llama/status.json`.
Accepted records are in the current result log; initial reports remain under
`results/llama_fa_iterative_20261004/experiment_v1/`. `decisions.json` records the rationale before each target.
FA-41 final review completed October 6 (24 accepted); read
`results/ordered_fa_followup_20261005/llama/FINAL_REVIEW.md` despite the historical
`awaiting_final_review` status label. FA-51c later completed four 16-credit
cache/SPM flag-0/15 cases under
`results/cacheflex_flash_attention/softmax_tune_20261005/fa51c_llama/`; these require
scientific review. See the current result log for timings. Do not relaunch a queue
from stale post-reboot status. Frozen guest, simulator, and active runner inputs must remain unchanged.

Earlier `cacheflex_flash_attention/llama_cpu` is a library reference with different
precision/profile; its timings are not a baseline for this model-shaped study.
