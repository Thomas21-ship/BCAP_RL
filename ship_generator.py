import numpy as np

def generate_single_vessel(last_arrival_time=0):
    # 1. Length: Normal distribution (Mean=12, StdDev=4) clipped between 4 and 20
    length = np.random.normal(12, 4)
    length = np.clip(length, 4, 20)
    
    # 2. Workload: Correlated to length (longer ship = more containers)
    # We use length^2 to simulate volume, plus some random noise
    base_workload = (length**2) * 12.5  # Scales 4-20 length to ~200-5000 workload
    workload = base_workload + np.random.normal(0, base_workload * 0.1)
    workload = np.clip(workload, 100, 5000)
    
    # 3. Arrival: Exponential distribution (Poisson Process)
    # scale=40 means on average a ship arrives every 40 time steps
    inter_arrival_time = np.random.exponential(scale=40)
    arrival_time = last_arrival_time + inter_arrival_time
    
    return [float(length), float(workload), float(arrival_time)]

# Test
raw_vessel = generate_single_vessel(0)
print(f"Raw: {raw_vessel}")