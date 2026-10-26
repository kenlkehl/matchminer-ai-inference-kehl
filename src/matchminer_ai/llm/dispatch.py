"""Capacity-aware endpoint choice shared by synchronous and async clients."""


def least_loaded_endpoint(entries):
    """Choose (identity, active, limit, last_used) by occupancy, then recency.

    Dispatch only actual ready requests. Reserving one waiting queue consumer per
    endpoint gives sparse arrivals equal shares even when one GPU is much faster.
    """
    available = (entry for entry in entries if entry[2] > 0 and entry[1] < entry[2])
    selected = min(available, key=lambda entry: (entry[1] / entry[2], entry[3]), default=None)
    return selected[0] if selected is not None else None
