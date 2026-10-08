"""Build the independent address/data-readiness guest regression."""
import subprocess
from build_access_timing import ROOT
OUT=ROOT/'results/cacheflex_flash_attention/spm_early_address_20260924/tests'
OUT.mkdir(parents=True,exist_ok=True)
old=ROOT/'results/cacheflex_flash_attention/output_resident_20260924/roundtrip'
script=(old/'build.sh').read_text().replace(str(old),str(OUT))
script=script.replace('tests/test_output_resident.cpp','tests/test_spm_early_address.cpp')
(OUT/'build.sh').write_text(script)
subprocess.run(['bash',str(OUT/'build.sh')],cwd=ROOT,check=True)
