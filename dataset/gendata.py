import json
import random
import os

def generate_instance(P, L, T):
    """
    Generates a single dictionary matching the rh2_baseline.py JSON schema.
    """
    def make_1d(length, dist_func):
        return [dist_func() for _ in range(length)]
    def make_2d(rows, cols, dist_func):
        return [[dist_func() for _ in range(cols)] for _ in range(rows)]

    capacity_hours = 120.0
    
    data = {
        "num_products": P,
        "num_lines": L,
        "num_periods": T,
        "capacity_per_line": [capacity_hours] * L,
        "eligibility_matrix": [[1 for _ in range(P)] for _ in range(L)],
        "processing_time_matrix": make_2d(L, P, lambda: random.uniform(11/60, 15/60)),
        "production_cost_matrix": make_2d(L, P, lambda: random.uniform(1.3, 1.6)),
        "demand_profile": make_2d(T, P, lambda: random.randint(0, 150)),
        "hazard_rate": make_1d(L, lambda: random.choice([2.0, 3.0]) / capacity_hours),
        "cm_time": make_1d(L, lambda: random.uniform(7/60, 11/60)),
        "pm_time": make_1d(L, lambda: random.uniform(2.5, 3.5)),
        "pm_cost": make_1d(L, lambda: random.uniform(10.0, 20.0)),
        "cm_cost": make_1d(L, lambda: random.uniform(2.0, 3.0)),
        
        "holding_cost": 1.75,
        "backlog_cost": 5.25,
        "per_product_backlog_penalty": 12.25,
        
        "first_setup_time": make_1d(L, lambda: random.uniform(1.2, 2.4)),
        "first_setup_cost": make_1d(L, lambda: random.uniform(25.0, 30.0)),
        "setup_time_matrix": [],
        "setup_cost_matrix": []
    }

    for l in range(L):
        t_mat, c_mat = [], []
        for i in range(P):
            t_row, c_row = [], []
            for j in range(P):
                if i == j:
                    t_row.append(0.0)
                    c_row.append(0.0)
                else:
                    t_row.append(random.uniform(1.2, 2.4))
                    c_row.append(random.uniform(25.0, 30.0))
            t_mat.append(t_row)
            c_mat.append(c_row)
        data["setup_time_matrix"].append(t_mat)
        data["setup_cost_matrix"].append(c_mat)

    return data

if __name__ == "__main__":
    # Table 2 configurations: (Scale, P, L, T)
    instances_config = [
        # Small (1-9)
        ("Small", 5, 2, 4), ("Small", 6, 2, 4), ("Small", 7, 2, 4),
        ("Small", 8, 3, 4), ("Small", 9, 3, 4), ("Small", 10, 3, 4),
        ("Small", 11, 4, 4), ("Small", 12, 4, 4), ("Small", 13, 4, 4),
        # Medium (10-18)
        ("Medium", 14, 5, 4), ("Medium", 15, 5, 4), ("Medium", 16, 5, 4),
        ("Medium", 17, 6, 4), ("Medium", 18, 6, 4), ("Medium", 19, 6, 4),
        ("Medium", 20, 7, 4), ("Medium", 21, 7, 4), ("Medium", 22, 7, 4),
        # Large (19-27)
        ("Large", 23, 8, 4), ("Large", 24, 8, 4), ("Large", 25, 8, 4),
        ("Large", 26, 9, 4), ("Large", 27, 9, 4), ("Large", 28, 9, 4),
        ("Large", 29, 10, 4), ("Large", 30, 10, 4), ("Large", 31, 10, 4),
    ]

    print("Generating Datasets using instance_P_L_T naming convention...\n")

    for scale, p, l, t in instances_config:
        # Create scale directory (e.g., dataset/Small)
        target_dir = os.path.join("dataset", scale)
        os.makedirs(target_dir, exist_ok=True)

        # Generate data
        instance_data = generate_instance(p, l, t)

        # Name file as instance_P_L_T.json (e.g., instance_5_2_4.json)
        filename = f"instance_{p}_{l}_{t}.json"
        filepath = os.path.join(target_dir, filename)

        with open(filepath, "w") as f:
            json.dump(instance_data, f, indent=2)

        print(f"Generated: {filepath}")

    print("\nDataset generation complete!")