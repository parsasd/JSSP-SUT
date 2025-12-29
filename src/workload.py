import os
import random

def ensure_datasets():
    """
    Checks if data/taillard_instances/ta01.txt exists.
    If not, creates a dummy TA01 file so the simulation can run immediately.
    """
    directory = "data/taillard_instances"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = f"{directory}/ta01.txt"
    if not os.path.exists(file_path):
        print(f"WARNING: {file_path} not found. Creating a DUMMY TA01 file for testing.")
        # Standard Taillard TA01 Format (15 Jobs, 15 Machines)
        # First line: Jobs Machines
        # Subsequent lines: Pairs of (Duration Machine)
        dummy_content = """15 15
54 0 34 1 61 2 2 3 17 4 52 5 95 6 26 7 22 8 20 9 46 10 33 11 50 12 85 13 47 14
95 0 20 1 31 2 11 3 32 4 41 5 60 6 78 7 8 8 26 9 10 10 70 11 37 12 18 13 44 14
15 0 22 1 38 2 71 3 45 4 47 5 35 6 92 7 60 8 68 9 93 10 11 11 50 12 40 13 25 14
48 0 88 1 54 2 20 3 67 4 38 5 44 6 73 7 92 8 8 9 51 10 7 11 16 12 55 13 65 14
40 0 92 1 20 2 23 3 83 4 82 5 33 6 49 7 93 8 71 9 52 10 65 11 63 12 56 13 28 14
72 0 45 1 29 2 52 3 85 4 58 5 69 6 36 7 14 8 26 9 82 10 93 11 18 12 88 13 77 14
89 0 71 1 56 2 40 3 53 4 19 5 57 6 52 7 94 8 36 9 63 10 57 11 25 12 45 13 14 14
33 0 54 1 84 2 8 3 80 4 39 5 46 6 82 7 71 8 50 9 67 10 42 11 18 12 65 13 77 14
52 0 43 1 73 2 15 3 20 4 47 5 68 6 40 7 88 8 5 9 63 10 24 11 93 12 88 13 51 14
78 0 26 1 45 2 23 3 14 4 25 5 29 6 33 7 62 8 50 9 70 10 47 11 81 12 55 13 40 14
93 0 16 1 28 2 54 3 69 4 42 5 65 6 47 7 88 8 8 9 20 10 55 11 14 12 60 13 26 14
85 0 94 1 5 2 39 3 20 4 23 5 36 6 63 7 42 8 68 9 29 10 24 11 50 12 58 13 18 14
60 0 55 1 8 2 52 3 5 4 48 5 20 6 51 7 44 8 29 9 58 10 45 11 67 12 40 13 54 14
68 0 70 1 88 2 40 3 63 4 52 5 45 6 71 7 24 8 60 9 8 10 20 11 54 12 25 13 93 14
18 0 14 1 65 2 88 3 71 4 45 5 93 6 40 7 50 8 47 9 20 10 52 11 5 12 24 13 60 14
"""
        with open(file_path, "w") as f:
            f.write(dummy_content)
    return file_path

def get_google_cluster_resources(num_machines):
    """
    Returns a dictionary of node profiles {machine_id: {attributes}}
    Simulates Google Cluster Traces with 3 classes of nodes:
    - High Perf (Class A): High IPT, High Power, Med Reliability
    - Balanced  (Class B): Med IPT, Med Power, High Reliability
    - Low Power (Class C): Low IPT, Low Power, Low Reliability (Edge Devices)
    """
    profiles = {}
    
    # Ratios: 20% High, 50% Med, 30% Low
    for i in range(num_machines):
        node_type = random.choices(['A', 'B', 'C'], weights=[0.2, 0.5, 0.3])[0]
        
        if node_type == 'A':
            # Fast but Power Hungry
            profiles[i] = {
                'IPT': random.uniform(1.8, 2.5),  # Instructions Per Time (Speedup)
                'RAM': 32000,
                'static_power': 100,
                'power_alpha': 0.15,
                'failure_rate': 0.0002
            }
        elif node_type == 'B':
            # Balanced
            profiles[i] = {
                'IPT': random.uniform(1.0, 1.5),
                'RAM': 16000,
                'static_power': 50,
                'power_alpha': 0.08,
                'failure_rate': 0.0001 # Most reliable
            }
        else:
            # IoT / Edge Device (Slow, Low Power, Unreliable)
            profiles[i] = {
                'IPT': random.uniform(0.3, 0.8),
                'RAM': 4000,
                'static_power': 10,
                'power_alpha': 0.02,
                'failure_rate': 0.002
            }
            
    return profiles

def parse_taillard(file_path):
    """
    Parses a standard Taillard JSSP text file.
    Output: A dict where Key=Job_ID, Value=List of (Machine, Duration) tuples
    """
    jobs = {}
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return jobs

    with open(file_path, 'r') as f:
        # Read all lines and strip whitespace
        lines = [l.strip() for l in f.readlines() if l.strip()]
        
    # The first line contains: Number of Jobs, Number of Machines
    header = lines[0].split()
    num_jobs = int(header[0])
    num_machines = int(header[1])
    
    print(f"Parsing Taillard Instance: {num_jobs} Jobs, {num_machines} Machines")

    # Taillard format: row = job, cols = pairs of (Duration, Machine)
    # BUT standard instances like ta01 often have:
    # Line 1: Times for Job 0
    # Line 2: Machines for Job 0
    # OR (Time Machine) pairs.
    # The dummy generator above uses (Duration Machine) pairs per line.
    
    # Let's handle the specific format generated above (Pairs)
    job_id = 0
    for line in lines[1:]:
        parts = list(map(int, line.split()))
        sequence = []
        # Step 2: Iterate pairs
        for i in range(0, len(parts), 2):
            if i+1 >= len(parts): break
            duration = parts[i]
            machine = parts[i+1]
            sequence.append({'machine': machine, 'duration': duration})
            
        jobs[job_id] = sequence
        job_id += 1
        
    return jobs