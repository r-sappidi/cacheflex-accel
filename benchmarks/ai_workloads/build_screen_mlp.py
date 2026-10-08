"""Isolated MLP build with safe post-ROI export; original study unchanged."""
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
source=(HERE/'build_mlp_study.sh').read_text()
source=source.replace('here="$(cd "$(dirname "$0")" && pwd)"', f'here="{HERE}"')
source=source.replace('cp "$here/mlp_study.cpp" "$out/mlp_study.cpp"',
                      'python3 "$here/screen_reference.py" mlp-source "$out/mlp_study.cpp"')
source=source.replace('"$here/mlp_study.cpp"', '"$out/mlp_study.cpp"')
subprocess.run(['bash','-s','--',str(Path(sys.argv[1]).resolve())],input=source,text=True,check=True)
