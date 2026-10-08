"""Validate the analytical handoff without launching simulations or fabricating guests."""
import argparse,json
from pathlib import Path
from qwen_placement_model import Candidate, account


def check(plan):
    if plan.get('schema')!='cacheflex-placement-plan-v1':raise ValueError('unsupported schema')
    if plan.get('auto_launch') is not False:raise ValueError('analytical plan must not auto-launch')
    policy=plan['resource_policy']
    if not (0 < policy['cpu_host_fraction'] <= .2 and 0 < policy['ram_host_fraction'] <= .2 and
            0 < policy['max_new_simulators'] <= 2):raise ValueError('resource policy exceeds authorization')
    seen=set();jobs=[]
    for job in plan['jobs']:
        c=Candidate(**job['case']);m=account(c)
        if c.key in seen:raise ValueError('duplicate job')
        seen.add(c.key)
        if not plan['model_validation_passed'] and job['performance_selection_allowed']:
            raise ValueError('failed model promoted a performance candidate')
        if m['staged_bytes']!=job['exact']['staged_bytes']:raise ValueError('stale work accounting')
        if job['state'] not in ('requires_guest_implementation','requires_isolated_runner_adapter'):
            raise ValueError('unverified execution state')
        jobs.append(dict(id=c.key,state=job['state'],launch_ready=False,
                         reason='requires checked guest/runner, full correctness gates, and live shared-broker admission'))
    return dict(valid=True,jobs=jobs,launch_ready=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('plan',type=Path);a=p.parse_args()
    print(json.dumps(check(json.loads(a.plan.read_text())),indent=2))
