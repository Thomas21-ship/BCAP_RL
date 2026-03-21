class Vessel:
    def __init__(self, vessel_id, length, workload, arrival_time):
        # --- IDENTITY & PHYSICAL STATS (from ship_generator) ---
        self.id = vessel_id
        self.length = length                        # In quay blocks (4-20)
        self.workload = workload                    # Total containers to process (100-5000)
        self.arrival_time = arrival_time            # Time step when vessel arrives

        # --- DYNAMIC STATE (changes during simulation) ---
        self.status = "waiting"                     # "waiting", "docked", "departed"
        self.containers_remaining = workload        # Counts down as cranes work
        self.cranes_assigned = 0                    # Cranes currently working this vessel

    def is_finished(self):
        # Returns True if all containers have been processed
        return self.containers_remaining <= 0

    def __repr__(self):
        # This controls what prints when you do print(vessel)
        return (f"Vessel(id={self.id}, length={self.length}, "
                f"status={self.status}, containers_remaining={self.containers_remaining}, "
                f"cranes_assigned={self.cranes_assigned})")
