import numpy as np


def generate_single_vessel(
    last_arrival_time=0,
    min_length=2,
    max_length=10,
    min_workload=50,
    max_workload=2000,
    arrival_scale=20,
):
    # 1. Length: Normal distribution clipped to the provided bounds
    mean = (min_length + max_length) / 2.0
    std = max(1.0, (max_length - min_length) / 4.0)
    length = np.random.normal(mean, std)
    length = np.clip(length, min_length, max_length)

    # 2. Workload: Correlated to length with mild noise
    base_workload = (length**2) * (max_workload / (max_length**2))
    workload = base_workload + np.random.normal(0, base_workload * 0.1)
    workload = np.clip(workload, min_workload, max_workload)

    # 3. Arrival: Exponential distribution (Poisson Process)
    inter_arrival_time = np.random.exponential(scale=arrival_scale)
    arrival_time = last_arrival_time + inter_arrival_time

    return [float(length), float(workload), float(arrival_time)]
