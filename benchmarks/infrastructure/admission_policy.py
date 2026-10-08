"""Host-only memory policy. No simulated hardware or timing parameters."""
GIB=2**30
QUANTUM=GIB//2
MIN_HEADROOM=3*GIB
GROW_MARGIN=2*GIB
GROW_STEP=GIB
BASE_RESERVE=6*GIB

def initial_limit(vms,old_limit):
    """Trim only unallocated allowance; retain >=3GiB slack in adopted jobs."""
    desired=((vms+MIN_HEADROOM+QUANTUM-1)//QUANTUM)*QUANTUM
    return min(old_limit,desired) if old_limit>=vms+MIN_HEADROOM else desired

def can_grow(reserved,cap,pressure):
    return not pressure and reserved+GROW_STEP<=cap
