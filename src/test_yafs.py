"""
Minimal smoke test for the updated JSSP pipeline.
Runs a short YAFS simulation with the default dataset and placement mapping.
"""
import yafs.core
from yafs.selection import First_ShortestPath

from placement import JSSPPlacement
from simple_population import SimplePopulation
from job_factory import JSSPWorkload
from workload import parse_taillard, ensure_datasets, get_google_cluster_resources
from yafs_patch import apply_yafs_patches
from ga_optimizer import create_topology, OP_SPECS


def main():
    apply_yafs_patches()
    ta_file = ensure_datasets()
    raw_jobs = parse_taillard(ta_file)
    node_profiles = get_google_cluster_resources(15)

    # Map each operation to its first eligible node
    module_node_map = {}
    for op in OP_SPECS:
        module_node_map[f"Op_{op['job_id']}_{op['op_idx']}"] = op["eligible_nodes"][0]

    workload_factory = JSSPWorkload(raw_jobs, OP_SPECS)
    release_times = {j: 0 for j in raw_jobs}
    app = workload_factory.create_application(release_times, module_node_map)

    topo = create_topology(node_profiles)
    sim = yafs.core.Sim(topo, default_results_path="logs/log_test")
    placement = JSSPPlacement(module_node_map)
    pop_policy = SimplePopulation(node_id=0, name="SimplePop")

    sim.deploy_app(app, placement, First_ShortestPath())
    sim.population_policy[pop_policy.name] = {"population_policy": pop_policy, "apps": [app.name]}

    sim.run(until=1000)
    print("Smoke test completed.")


if __name__ == "__main__":
    main()
