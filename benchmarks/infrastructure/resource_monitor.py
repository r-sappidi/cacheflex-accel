"""Standalone broker heartbeat + resource monitor (OPS-25, 2026-10-06).

Replaces the heartbeat that ran inside results/retained_protocol_restart_20261002/
run_spm_only.py (old supervisor, not to be resumed). Same thresholds: hold new
admissions when user RSS > ram_limit - 4 GiB or MemAvailable < 64 GiB + unused
budget. Writes broker.json heartbeat/hold_new and spm_only_resources.json every 5 s.
Single instance (flock). Does not create, release or modify leases.
"""
import fcntl, json, os, sys, time
from pathlib import Path
R = Path('/data2/rsappidi/cacheflex_mc'); G = 2 ** 30
sys.path[:0] = [str(R / 'scripts'), str(R / 'results/parallel_overlap_20260930')]
import broker
from parallel_budget_scheduler import owned, memory
OUT = R / 'results/retained_protocol_restart_20261002/spm_only_resources.json'
LOCK = R / 'results/parallel_overlap_20260930/resource_monitor.lock'


def save(p, d):
    q = p.with_suffix(p.suffix + '.tmp'); q.write_text(json.dumps(d, indent=2) + '\n'); q.replace(p)


def main():
    lock = LOCK.open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    cfg = json.loads((R / 'results/parallel_budget_20260929/budget.json').read_text())
    os.sched_setaffinity(0, cfg['cpu_set'][-1:]); os.nice(5)
    while True:
        rss = sum(x['rss'] for x in owned().values()); m = memory()
        hold = rss > cfg['ram_limit'] - 4 * G or m['MemAvailable'] < 64 * G + max(0, cfg['ram_limit'] - rss)
        with broker.locked() as b:
            b['heartbeat'] = time.time(); b['owner'] = os.getpid(); b['hold_new'] = hold
        save(OUT, dict(updated=time.time(), user_rss_bytes=rss, ram_limit_bytes=cfg['ram_limit'], pressure=hold,
                       available_bytes=m['MemAvailable'], monitor='benchmarks/infrastructure/resource_monitor.py'))
        time.sleep(5)


if __name__ == '__main__':
    main()
