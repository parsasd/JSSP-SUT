import os
import pandas as pd
import networkx as nx

# YAFS IMPORTS
import yafs.core
from yafs.application import Application, Message
from yafs.topology import Topology
from yafs.selection import First_ShortestPath
from yafs.placement import Placement
from yafs.distribution import deterministic_distribution

from simple_population import SimplePopulation
from yafs_patch import apply_yafs_patches

# ==========================================
# 1. PLACEMENT POLICY
# ==========================================
class JSSPPlacement(Placement):
    def initial_allocation(self, sim, app_name):
        allocations = []
        app = sim.apps[app_name]
        
        data_source = getattr(app, "rb_module_data", app.modules)

        for module_name, module_info in data_source.items():
            if module_info.get("Type") == Application.TYPE_SOURCE:
                allocations.append((module_name, 0))
                continue

            if "target_machine" in module_info:
                target = module_info["target_machine"] + 1
            else:
                target = 0

            sim.deploy_module(app_name, module_name, app.services[module_name], ids=[target])
            
        return allocations

    def get_placement(self, sim, app, module_name, service):
        module_info = getattr(app, "rb_module_data", app.modules).get(module_name, {})
        if "target_machine" in module_info:
            target = module_info["target_machine"] + 1
            return [target]
        
        return [0]

# ==========================================
# 2. TOPOLOGY GENERATOR
# ==========================================
def create_topology():
    G = nx.Graph()
    # Node 0: Cloud/Controller
    G.add_node(0, IPT=1000, RAM=40000) 
    
    # Nodes 1-15: Edge Machines
    for i in range(1, 16): 
        G.add_node(i, IPT=500, RAM=8000)
    
    # Edges
    for i in range(1, 16):
        G.add_edge(0, i, BW=100, PR=1)
        
    t = Topology()
    t.G = G
    return t

# ==========================================
# 3. TAILLARD PARSER & APP BUILDER
# ==========================================
def load_and_build_apps(file_path):
    jobs = {}
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return []

    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    job_id = 0
    for line in lines[1:]: # Skip header
        parts = list(map(int, line.split()))
        sequence = []
        for i in range(0, len(parts), 2):
            duration = parts[i]
            machine = parts[i+1]
            sequence.append({'machine': machine, 'duration': duration})
        jobs[job_id] = sequence
        job_id += 1

    applications = []
    for j_id, ops in jobs.items():
        app_name = f"Job_{j_id}"
        app = Application(name=app_name)
        
        # Containers
        modules_dict = {}
        
        # A. Source
        src_name = f"{app_name}_Source"
        src_info = {"Type": Application.TYPE_SOURCE, "RAM": 10}
        
        modules_dict[src_name] = src_info         
        
        # B. Operations
        op_names = []
        for idx, data in enumerate(ops):
            op_name = f"{app_name}_Op_{idx}"
            op_names.append(op_name)
            
            op_info = {
                "Type": Application.TYPE_MODULE,
                "RAM": 50,
                "workload_duration": data['duration'],
                "target_machine": data['machine']
            }
            modules_dict[op_name] = op_info         

        # Store modules for placement lookup
        app.modules = modules_dict
        app.rb_module_data = modules_dict
        
        # D. Connectivity & Service Wiring
        msg_start = Message(f"Start_{j_id}", src_name, op_names[0], instructions=ops[0]["duration"], bytes=0)
        msg_start.last_idDes = []
        app.messages[msg_start.name] = msg_start
        start_dist = deterministic_distribution(name=f"Det_Start_{j_id}", time=100)
        app.add_service_source(src_name, start_dist, msg_start)

        message_in = msg_start
        for idx, op_name in enumerate(op_names):
            msg_out = ""
            dests = []
            probs = []
            if idx < len(op_names) - 1:
                next_op = op_names[idx + 1]
                msg_out = Message(
                    f"M_{op_name}_{next_op}",
                    op_name,
                    next_op,
                    instructions=ops[idx + 1]["duration"],
                    bytes=100,
                )
                msg_out.last_idDes = []
                app.messages[msg_out.name] = msg_out
                dests = [next_op]
                probs = [1.0]

            app.add_service_module(op_name, message_in, msg_out, distribution=lambda **_: True, module_dest=dests, p=probs)
            message_in = msg_out if msg_out else message_in
            
        applications.append(app)
        
    return applications

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    print("--- FINAL SOLVER RUN ---")
    apply_yafs_patches()
    
    if not os.path.exists("logs"): os.makedirs("logs")
    
    # Init Simulator
    s = yafs.core.Sim(create_topology(), default_results_path="logs/log_final")
    
    # Load Data
    apps = load_and_build_apps("data/taillard_instances/ta01.txt")
    print(f"Loaded {len(apps)} Jobs.")
    
    # Deploy
    print("Deploying Applications...")
    selection_policy = First_ShortestPath()
    population_policy = SimplePopulation(node_id=0, name="SimplePop")

    for app in apps:
        s.deploy_app(app, JSSPPlacement(name="JSSP"), selection_policy)
        if population_policy.name not in s.population_policy:
            s.population_policy[population_policy.name] = {
                "population_policy": population_policy,
                "apps": [],
            }
        s.population_policy[population_policy.name]["apps"].append(app.name)
        
    # Run
    print("Running Simulation...")
    s.run(until=100000) 
    
    # Result Analysis
    print("Simulation Finished.")
    
    final_path = "logs/log_final.csv"
    if os.path.exists(final_path):
        try:
            df = pd.read_csv(final_path)
            print(f"Log Rows: {len(df)}")
            if not df.empty:
                makespan = df['time'].max()
                print(f"\n✅ SUCCESS! FINAL MAKESPAN: {makespan}")
            else:
                print("❌ FAILURE: Log is empty. (Sources not placed)")
        except Exception as e:
            print(f"Error reading log: {e}")
    else:
        print(f"❌ FAILURE: File {final_path} not found.")

if __name__ == "__main__":
    main()
