import os
import random
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple

TAILLARD_SEEDS_15x15: Dict[str, Tuple[int, int, int, int]] = {
    "ta01": (840612802, 398197754, 15, 15),
    "ta02": (1314640371, 386720536, 15, 15),
    "ta03": (1227221349, 316176388, 15, 15),
    "ta04": (342269428, 1806358582, 15, 15),
    "ta05": (1603221416, 1501949241, 15, 15),
    "ta06": (1357584978, 1734077082, 15, 15),
    "ta07": (44531661, 1374316395, 15, 15),
    "ta08": (302545136, 2092186050, 15, 15),
    "ta09": (1153780144, 1393392374, 15, 15),
    "ta10": (73896786, 1544979948, 15, 15),
}


def _unif(seed: int, low: int, high: int) -> Tuple[int, int]:
    # Taillard RNG (minimal standard LCG), mirrors OR-Library jobshop2.txt
    m = 2147483647
    a = 16807
    b = 127773
    c = 2836
    k = seed // b
    seed = a * (seed % b) - k * c
    if seed < 0:
        seed += m
    value_0_1 = seed / m
    return seed, low + int(value_0_1 * (high - low + 1))


def _generate_taillard_instance(
    file_path: str,
    time_seed: int,
    machine_seed: int,
    num_jobs: int,
    num_machines: int,
) -> None:
    durations = [[0 for _ in range(num_machines)] for _ in range(num_jobs)]
    machines = [[j for j in range(num_machines)] for _ in range(num_jobs)]

    for i in range(num_jobs):
        for j in range(num_machines):
            time_seed, durations[i][j] = _unif(time_seed, 1, 99)

    for i in range(num_jobs):
        for j in range(num_machines):
            machine_seed, k = _unif(machine_seed, j, num_machines - 1)
            machines[i][j], machines[i][k] = machines[i][k], machines[i][j]

    with open(file_path, "w") as f:
        f.write(f"{num_jobs} {num_machines}\n")
        for i in range(num_jobs):
            row = []
            for j in range(num_machines):
                row.append(f"{machines[i][j]} {durations[i][j]}")
            f.write(" ".join(row) + "\n")


def ensure_taillard_instance(instance: str) -> str:
    """
    Ensure a Taillard instance file exists locally. Supports ta01-ta10 (15x15)
    via OR-Library seeds. Returns the resolved file path.
    """
    directory = Path("data/taillard_instances")
    directory.mkdir(parents=True, exist_ok=True)

    path = Path(instance)
    if path.exists():
        return str(path)

    name = path.stem if path.suffix else instance
    file_path = directory / f"{name}.txt"
    if not file_path.exists():
        if name not in TAILLARD_SEEDS_15x15:
            raise FileNotFoundError(f"Unknown Taillard instance '{name}'.")
        seed_time, seed_mach, num_jobs, num_machines = TAILLARD_SEEDS_15x15[name]
        _generate_taillard_instance(str(file_path), seed_time, seed_mach, num_jobs, num_machines)
    return str(file_path)


def ensure_datasets() -> str:
    """
    Backwards-compatible helper: ensure ta01 exists and return its path.
    """
    return ensure_taillard_instance("ta01")

def _load_cluster_profile_csv(csv_path: str) -> Dict[int, Dict[str, float]]:
    """
    Load deterministic node profiles from a CSV file.
    Expected columns: id, IPT, RAM, static_power, power_alpha, failure_rate
    """
    profiles: Dict[int, Dict[str, float]] = {}
    if not os.path.exists(csv_path):
        return profiles

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                node_id = int(row["id"])
            except (KeyError, ValueError):
                continue
            profiles[node_id] = {
                "IPT": float(row.get("IPT", 1.0)),
                "RAM": float(row.get("RAM", 8000)),
                "static_power": float(row.get("static_power", 20)),
                "power_alpha": float(row.get("power_alpha", 0.05)),
                "failure_rate": float(row.get("failure_rate", 0.001)),
            }
    return profiles


def get_google_cluster_resources(num_machines: int) -> Dict[int, Dict[str, Any]]:
    """
    Returns a dictionary of node profiles {machine_id: {attributes}}
    Simulates Google Cluster Traces with 3 classes of nodes:
    - High Perf (Class A): High IPT, High Power, Med Reliability
    - Balanced  (Class B): Med IPT, Med Power, High Reliability
    - Low Power (Class C): Low IPT, Low Power, Low Reliability (Edge Devices)
    If data/google_cluster_profiles.csv exists, it is used to provide
    deterministic, reproducible node profiles.
    """
    csv_profiles = _load_cluster_profile_csv("data/google_cluster_profiles.csv")
    if csv_profiles:
        # Trim/pad to requested number of machines deterministically
        ordered_ids = sorted(csv_profiles.keys())
        selected_ids = ordered_ids[:num_machines]
        profiles = {i: csv_profiles[node_id] for i, node_id in enumerate(selected_ids)}
        return profiles

    profiles: Dict[int, Dict[str, Any]] = {}
    rng = random.Random(42)  # reproducible fallback

    # Ratios: 20% High, 50% Med, 30% Low
    for i in range(num_machines):
        node_type = rng.choices(['A', 'B', 'C'], weights=[0.2, 0.5, 0.3])[0]
        
        if node_type == 'A':
            # Fast but Power Hungry
            profiles[i] = {
                'IPT': rng.uniform(1.8, 2.5),  # Instructions Per Time (Speedup)
                'RAM': 32000,
                'static_power': 100,
                'power_alpha': 0.15,
                'failure_rate': 0.0002
            }
        elif node_type == 'B':
            # Balanced
            profiles[i] = {
                'IPT': rng.uniform(1.0, 1.5),
                'RAM': 16000,
                'static_power': 50,
                'power_alpha': 0.08,
                'failure_rate': 0.0001 # Most reliable
            }
        else:
            # IoT / Edge Device (Slow, Low Power, Unreliable)
            profiles[i] = {
                'IPT': rng.uniform(0.3, 0.8),
                'RAM': 4000,
                'static_power': 10,
                'power_alpha': 0.02,
                'failure_rate': 0.002
            }
            
    return profiles

def parse_taillard_with_meta(file_path: str) -> Tuple[Dict[int, List[Dict[str, Any]]], int]:
    """
    Parses a Taillard JSSP text file.
    Supports:
      - (machine duration) pairs per job line
      - Taillard 2-block format: machine order block + processing time block
    Returns (jobs_dict, num_machines).
    """
    jobs: Dict[int, List[Dict[str, Any]]] = {}

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return jobs, 0

    with open(file_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    header = lines[0].split()
    num_jobs = int(header[0])
    num_machines = int(header[1])

    print(f"Parsing Taillard Instance: {num_jobs} Jobs, {num_machines} Machines")

    body = lines[1:]
    if len(body) == 2 * num_jobs:
        machine_lines = [list(map(int, body[i].split())) for i in range(num_jobs)]
        time_lines = [list(map(int, body[i + num_jobs].split())) for i in range(num_jobs)]
        for job_id in range(num_jobs):
            sequence = []
            for j in range(num_machines):
                sequence.append({"machine": machine_lines[job_id][j], "duration": time_lines[job_id][j]})
            jobs[job_id] = sequence
        return jobs, num_machines

    for job_id, line in enumerate(body):
        parts = list(map(int, line.split()))
        sequence = []
        if len(parts) >= 2 * num_machines:
            for i in range(0, len(parts), 2):
                if i + 1 >= len(parts):
                    break
                machine = parts[i]
                duration = parts[i + 1]
                sequence.append({"machine": machine, "duration": duration})
        else:
            for i in range(0, len(parts), 2):
                if i + 1 >= len(parts):
                    break
                duration = parts[i]
                machine = parts[i + 1]
                sequence.append({"machine": machine, "duration": duration})
        jobs[job_id] = sequence

    return jobs, num_machines


def parse_taillard(file_path: str) -> Dict[int, List[Dict[str, Any]]]:
    jobs, _ = parse_taillard_with_meta(file_path)
    return jobs
