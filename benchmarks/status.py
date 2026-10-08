#!/usr/bin/env python3
"""Read live queue states from the accepted-result registry; never mutate jobs."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def count(value):
    return len(value) if isinstance(value, (dict, list, tuple)) else value or 0


def boot_time():
    try:
        for line in Path('/proc/stat').read_text().splitlines():
            if line.startswith('btime '):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def predates_boot(state, boot):
    updated = state.get('updated')
    return (boot is not None and isinstance(updated, (int, float))
            and updated < boot)


def pid_observation(pid):
    if not isinstance(pid, int) or pid <= 0:
        return 'no usable PID'
    try:
        Path(f'/proc/{pid}').stat()
    except FileNotFoundError:
        return 'absent from this /proc namespace'
    except OSError:
        return 'process visibility unavailable'
    return 'PID exists; ownership, start time and command NOT verified'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='include completed and historical queues')
    args = parser.parse_args()
    registry = ROOT / 'docs/experiments/current_results.json'
    if not registry.exists():
        parser.error('Run python3 scripts/refresh_current_result_log.py first')
    data = json.loads(registry.read_text())
    boot = boot_time()
    print(f"Registry refreshed: {data['updated']}; reading status files live.")
    print('Queue records are not process-liveness checks; verify host PID identity before signalling.')
    if boot is not None:
        print(f'Host boot: {datetime.fromtimestamp(boot, timezone.utc).isoformat()}')
    shown = 0
    for queue in data['queues']:
        path = ROOT / queue['path'] / 'status.json'
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            print(f"{queue['ledger']}: unavailable ({error})")
            continue
        stage = state.get('stage', 'unknown')
        active = state.get('active') or {}
        terminal = stage.startswith(('complete', 'failed', 'cancelled', 'superseded'))
        if not args.all and terminal and not active:
            continue
        shown += 1
        print(f"\n{queue['ledger']}: {stage}; accepted={count(state.get('completed'))}, "
              f"recorded_active={count(active)}, pending={count(state.get('pending'))}, "
              f"failed={count(state.get('failed'))}")
        if not terminal and predates_boot(state, boot):
            print('  PRE-BOOT RECORD: stage is historical, not evidence of a live queue.')
        if isinstance(active, dict):
            for name, info in active.items():
                if isinstance(info, dict):
                    pid = info.get('pid')
                    print(f"  {name}: recorded PID={pid}, paused={info.get('paused', 'unspecified')}; {pid_observation(pid)}")
        print(f'  {path.relative_to(ROOT)}')
    if not shown:
        print('No nonterminal queues in this registry. Use --all for historical states.')
    print('\nRumur: results/rumur_refresh_20261001/verification_v6/status.json')
    print('Resources: results/parallel_overlap_20260930/broker.json')
    try:
        broker = json.loads((ROOT / 'results/parallel_overlap_20260930/broker.json').read_text())
        heartbeat = broker.get('heartbeat')
        age = (datetime.now(timezone.utc).timestamp() - heartbeat
               if isinstance(heartbeat, (int, float)) else None)
        stale = age is None or age > 60 or predates_boot({'updated': heartbeat}, boot)
        print(f"  Monitor heartbeat: {'STALE / unavailable' if stale else 'recent (not an identity check)'}; "
              f"recorded hold_new={broker.get('hold_new')}; {pid_observation(broker.get('owner'))}")
    except (OSError, ValueError) as error:
        print(f'  Broker unavailable: {error}')


if __name__ == '__main__':
    main()
