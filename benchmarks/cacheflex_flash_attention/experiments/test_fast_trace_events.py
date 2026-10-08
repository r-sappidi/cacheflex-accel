import itertools,sys
from _paths import FA_DIR
from pathlib import Path
from analyze_residency import events as old
from fast_residency_events import events as new
p=Path(sys.argv[1]);n=int(sys.argv[2]) if len(sys.argv)>2 else 100000
assert list(itertools.islice(old(p),n))==list(itertools.islice(new(p),n))
print('PASS',n,'events match exactly')
