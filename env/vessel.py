import numpy as np
from collections import deque


class Vessel:
    def __init__(self, vessel_id, length, workload, arrival_time):
        # --- IDENTITY & PHYSICAL STATS (from generator) ---
        self.id = vessel_id
        self.length = length
        self.workload = workload
        # Relative offset: negative = already arrived, positive = on the way
        self.arrival_time = arrival_time

        # --- DYNAMIC STATE (changes during simulation) ---
        self.status = "waiting" if arrival_time <= 0 else "on_the_way"
        self.containers_remaining = workload
        self.cranes_assigned = 0

    def is_finished(self):
        return self.containers_remaining <= 0

    def __repr__(self):
        return (
            f"Vessel(id={self.id}, length={self.length}, "
            f"status={self.status}, containers_remaining={self.containers_remaining}, "
            f"cranes_assigned={self.cranes_assigned})"
        )


class VesselQueue:
    def __init__(
        self,
        capacity,
        rng,
        current_time=0,
        arrival_scale=20,
        min_length=2,
        max_length=10,
        min_workload=50,
        max_workload=2000,
    ):
        if rng is None:
            raise ValueError("rng must be provided for deterministic seeding")
        self.queue = deque()
        self.capacity = capacity
        self.rng = rng
        self.current_time = current_time
        self.arrival_scale = arrival_scale
        self.min_length = min_length
        self.max_length = max_length
        self.min_workload = min_workload
        self.max_workload = max_workload
        self.last_eta = -int(self.rng.integers(0, 6))
        self._id_counter = 0

        for _ in range(capacity):
            self._enqueue_internal(self._generate_random_vessel())
        self.update_statuses()

    def _next_eta(self):
        interarrival = self.rng.exponential(scale=self.arrival_scale)
        self.last_eta += interarrival
        return self.last_eta

    def _generate_random_vessel(self):
        # 1. Length: Normal distribution clipped to bounds
        mean = (self.min_length + self.max_length) / 2.0
        std = max(1.0, (self.max_length - self.min_length) / 4.0)
        length = self.rng.normal(mean, std)
        length = np.clip(length, self.min_length, self.max_length)

        # 2. Workload: Correlated to length with mild noise
        base_workload = (length**2) * (self.max_workload / (self.max_length**2))
        workload = base_workload + self.rng.normal(0, base_workload * 0.1)
        workload = np.clip(workload, self.min_workload, self.max_workload)

        # 3. Arrival: Exponential (Poisson process) on an ETA timeline
        eta = self._next_eta()
        arrival_offset = eta - self.current_time

        self._id_counter += 1
        return Vessel(
            vessel_id=self._id_counter,
            length=int(np.round(length)),
            workload=float(workload),
            arrival_time=float(arrival_offset),
        )

    def _enqueue_internal(self, vessel):
        if len(self.queue) >= self.capacity:
            return False
        self.queue.append(vessel)
        return True

    def enqueue(self, vessel):
        return self._enqueue_internal(vessel)

    def dequeue(self):
        if not self.queue:
            return None
        return self.queue.popleft()

    def update_statuses(self):
        for vessel in self.queue:
            if vessel.status == "at_berth":
                continue
            if vessel.arrival_time <= 0:
                vessel.status = "waiting"
            else:
                vessel.status = "on_the_way"

    def advance_time(self, steps=1):
        self.current_time += steps
        for vessel in self.queue:
            if vessel.status != "at_berth":
                vessel.arrival_time -= steps
        self.update_statuses()

    def service_vessel_at(self, index):
        if index < 0 or index >= len(self.queue):
            return None
        vessel = self.queue[index]
        del self.queue[index]
        vessel.status = "at_berth"
        new_vessel = self._generate_random_vessel()
        self._enqueue_internal(new_vessel)
        return vessel


__all__ = ["Vessel", "VesselQueue"]
