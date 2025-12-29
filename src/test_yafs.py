import yafs.core
import networkx as nx
from yafs.topology import Topology
from yafs.selection import First_ShortestPath

# Import our custom scripts
from workload import parse_taillard
from job_factory import JobFactory
from placement import JSSPPlacement
from simple_population import SimplePopulation
from yafs_patch import apply_yafs_patches

def create_topology():
    G = nx.Graph()
    # Node 0: Cloud/Controller
    G.add_node(0, IPT=1000, RAM=32000, power_alpha=0.05, static_power=50) 
    
    # Nodes 1-15: Edge Machines (Matching Taillard's 15 machines)
    for i in range(1, 16): 
        G.add_node(i, IPT=500, RAM=8000, power_alpha=0.03, static_power=20)
    
    # Star topology
    for i in range(1, 16):
        G.add_edge(0, i, BW=100, PR=10)
        
    t = Topology()
    t.G = G
    return t

def main():
    apply_yafs_patches()
    print("--- 1. Parsing Data ---")
    raw_jobs = parse_taillard("data/taillard_instances/ta01.txt")

    print("--- 2. Building YAFS DAGs ---")
    factory = JobFactory(raw_jobs)
    applications = factory.create_applications()

    print("--- 3. Setup Simulator ---")
    s = yafs.core.Sim(create_topology(), default_results_path="logs/log_test")
    
    # Define Policies
    # Placement: Our custom JSSP logic
    placement_policy = JSSPPlacement(name="JSSP_Static")
    # Selection: Shortest Path (Standard for routing messages)
    selection_policy = First_ShortestPath()
    population_policy = SimplePopulation(node_id=0, name="SimplePop")

    # Deploy Applications
    for app in applications:
        s.deploy_app(app, placement_policy, selection_policy)
        if population_policy.name not in s.population_policy:
            s.population_policy[population_policy.name] = {
                "population_policy": population_policy,
                "apps": [],
            }
        s.population_policy[population_policy.name]["apps"].append(app.name)

    print("--- 4. Running Simulation ---")
    # Run for enough time to complete all jobs (e.g., 5000 units)
    s.run(until=5000)
    
    print("Done! Check 'logs/log_test.csv' for results.")

if __name__ == '__main__':
    main()
