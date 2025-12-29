import argparse
import math
import os
import random
import csv
from typing import Dict, List, Tuple, Any

import networkx as nx
import numpy as np
import pandas as pd
from deap import base, creator, tools

# YAFS IMPORTS
import yafs.core
from yafs.topology import Topology
from yafs.selection import First_ShortestPath

from job_factory import JSSPWorkload
from placement import JSSPPlacement
from simple_population import SimplePopulation
from workload import parse_taillard, get_google_cluster_resources, ensure_datasets
from yafs_patch import apply_yafs_patches


# ==========================================
# DATASET LOADING
# ==========================================
TA_FILE = ensure_datasets()
GLOBAL_RAW_JOBS = parse_taillard(TA_FILE)
NUM_JOBS = len(GLOBAL_RAW_JOBS)
NUM_MACHINES_REQ = 15
GLOBAL_NODE_PROFILES = get_google_cluster_resources(NUM_MACHINES_REQ)


def build_operation_specs(raw_jobs: Dict[int, List[Dict[str, Any]]], node_profiles: Dict[int, Dict[str, Any]]):
    """
    Flatten operations and assign eligible nodes per operation.
    Eligible set picks a fast, medium, and slow node to approximate FJSP flexibility.
    """
    specs = []
    # Sort node ids by IPT (descending)
    sorted_nodes = sorted(node_profiles.items(), key=lambda kv: kv[1].get("IPT", 1.0), reverse=True)
    if not sorted_nodes:
        return specs

    tier_fast = [nid for nid, _ in sorted_nodes[: max(1, len(sorted_nodes) // 3)]]
    tier_mid = [nid for nid, _ in sorted_nodes[len(sorted_nodes) // 3: 2 * len(sorted_nodes) // 3]]
    tier_slow = [nid for nid, _ in sorted_nodes[2 * len(sorted_nodes) // 3:]]

    for job_id, ops in raw_jobs.items():
        for op_idx, op in enumerate(ops):
            eligible = []
            for tier in (tier_fast, tier_mid, tier_slow):
                if tier:
                    eligible.append(tier[(job_id + op_idx) % len(tier)])
            # Map to topology node ids (+1 offset, node 0 is controller)
            eligible_topo = [n + 1 for n in eligible]
            # Ensure uniqueness
            eligible_topo = list(dict.fromkeys(eligible_topo))
            specs.append(
                {
                    "job_id": job_id,
                    "op_idx": op_idx,
                    "duration": op["duration"],
                    "eligible_nodes": eligible_topo,
                }
            )
    return specs


OP_SPECS = build_operation_specs(GLOBAL_RAW_JOBS, GLOBAL_NODE_PROFILES)
NUM_OPS = len(OP_SPECS)


# ==========================================
# RL-AOS AGENT
# ==========================================
class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.5):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.last_state = None
        self.last_action = None

    def get_state(self, population, fitness_history, hv_history):
        # Diversity via Hamming distance over concatenated genome
        if not population:
            return ("Low", "Fast", "Flat")
        genomes = [tuple(ind) for ind in population]
        uniq = len(set(genomes))
        diversity_ratio = uniq / len(population)
        if diversity_ratio < 0.3:
            div_state = "Low"
        elif diversity_ratio < 0.7:
            div_state = "Med"
        else:
            div_state = "High"

        if len(fitness_history) < 2:
            imp_state = "Fast"
        else:
            delta = fitness_history[-2] - fitness_history[-1]
            if delta < 0.5:
                imp_state = "Stagnant"
            elif delta < 20.0:
                imp_state = "Slow"
            else:
                imp_state = "Fast"

        if len(hv_history) < 2:
            hv_state = "Flat"
        else:
            hv_delta = hv_history[-1] - hv_history[-2]
            if hv_delta < 1e-3:
                hv_state = "Flat"
            elif hv_delta < 0:
                hv_state = "Drop"
            else:
                hv_state = "Rise"
        return (div_state, imp_state, hv_state)

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if random.random() < self.epsilon:
            return random.choice(range(len(self.actions)))
        return int(np.argmax(self.q_table[state]))

    def learn(self, current_state, reward):
        if self.last_state is not None:
            old_q = self.q_table[self.last_state][self.last_action]
            max_future_q = np.max(self.q_table.get(current_state, np.zeros(len(self.actions))))
            new_q = old_q + self.lr * (reward + self.gamma * max_future_q - old_q)
            self.q_table[self.last_state][self.last_action] = new_q
        self.epsilon = max(0.05, self.epsilon * 0.98)


# ==========================================
# TOPOLOGY HELPERS
# ==========================================
def create_topology(node_profiles):
    G = nx.Graph()
    # Cloud node (Controller) - ID 0
    G.add_node(0, IPT=10, RAM=64000, static_power=200, power_alpha=0.01)

    # Edge nodes - IDs start at 1
    for m_id, props in node_profiles.items():
        node_id = m_id + 1
        G.add_node(node_id, **props)

    for node_id in range(1, len(node_profiles) + 1):
        G.add_edge(0, node_id, BW=1000, PR=0)
    t = Topology()
    t.G = G
    return t


# ==========================================
# FITNESS EVALUATION
# ==========================================
def decode_individual(individual: List[int]) -> Tuple[List[int], List[int]]:
    job_seq = list(individual[:NUM_JOBS])
    machine_assignments = list(individual[NUM_JOBS:])
    return job_seq, machine_assignments


def clamp_assignments(assignments: List[int]) -> List[int]:
    clamped = []
    for idx, gene in enumerate(assignments):
        eligible = OP_SPECS[idx]["eligible_nodes"]
        if gene in eligible:
            clamped.append(gene)
        else:
            clamped.append(eligible[0])
    return clamped


def hypervolume_minimize(population, reference_point):
    """
    Compute a simple hypervolume for minimization objectives.
    Objectives: makespan, energy, -reliability (all minimized).
    """
    hv = 0.0
    ref_m, ref_e, ref_r = reference_point
    for ind in population:
        m, e, r = ind.fitness.values
        hv += max(0, (ref_m - m)) * max(0, (ref_e - e)) * max(0, (ref_r - (-r)))
    return hv


def evaluate_schedule(individual: List[int]):
    if not GLOBAL_RAW_JOBS or not OP_SPECS:
        return (999999, 999999, -999999)

    apply_yafs_patches()
    job_seq, machine_assignments = decode_individual(individual)
    machine_assignments = clamp_assignments(machine_assignments)

    # Derive job release times from permutation
    release_times = {}
    current_time = 1
    gap = 10
    for job_id in job_seq:
        release_times[job_id] = current_time
        current_time += gap

    # Map operation modules to chosen nodes
    module_node_map = {}
    for idx, op in enumerate(OP_SPECS):
        module_name = f"Op_{op['job_id']}_{op['op_idx']}"
        module_node_map[module_name] = machine_assignments[idx]

    workload_factory = JSSPWorkload(GLOBAL_RAW_JOBS, OP_SPECS)
    app = workload_factory.create_application(release_times, module_node_map)

    log_file = f"logs/log_eval_{os.getpid()}_{random.randint(0, 999999)}.csv"

    topo = create_topology(GLOBAL_NODE_PROFILES)
    s = yafs.core.Sim(topo, default_results_path=log_file.replace(".csv", ""))
    pop_policy = SimplePopulation(node_id=0, name="SimplePop")
    s.deploy_app(app, JSSPPlacement(module_node_map), First_ShortestPath())
    s.population_policy[pop_policy.name] = {"population_policy": pop_policy, "apps": [app.name]}

    # Sample stochastic failure times for reliability modeling
    failure_times = {}
    rng = random.Random(1234)
    for node_id, attrs in topo.G.nodes(data=True):
        if node_id == 0:
            continue
        lam = attrs.get("failure_rate", 0.001)
        failure_times[node_id] = rng.expovariate(lam) if lam > 0 else math.inf

    s.run(until=200000)

    try:
        df = pd.read_csv(log_file)
        if os.path.exists(log_file):
            os.remove(log_file)
        if df.empty:
            return (999999, 999999, -999999)

        makespan = df["time_out"].max()

        processing_events = df[df["time_out"] > df["time_in"]]

        # Static energy
        cluster_static_power = sum(d.get("static_power", 0) for _, d in topo.G.nodes(data=True))
        static_energy = cluster_static_power * makespan

        # Dynamic + communication energy
        dynamic_energy = 0.0
        comm_energy = 0.0
        has_src = "TOPO.src" in processing_events.columns
        for _, row in processing_events.iterrows():
            duration = row["time_out"] - row["time_in"]
            node_id = int(row["TOPO.dst"])
            node_attrs = topo.G.nodes[node_id]
            p_alpha = node_attrs.get("power_alpha", 0.0)
            ipt = node_attrs.get("IPT", 1.0)
            p_dynamic = p_alpha * (ipt ** 3)
            dynamic_energy += p_dynamic * duration

            # simple communication proxy: per-hop energy if src!=dst
            if has_src and row["TOPO.src"] != row["TOPO.dst"]:
                comm_energy += 0.01 * duration

        total_energy = static_energy + dynamic_energy + comm_energy

        # Reliability with failure penalties
        reliability_log_sum = 0.0
        failure_penalty = 0.0
        for _, row in processing_events.iterrows():
            duration = row["time_out"] - row["time_in"]
            node_id = int(row["TOPO.dst"])
            node_attrs = topo.G.nodes[node_id]
            lam = node_attrs.get("failure_rate", 0.001)
            reliability_log_sum += (-lam * duration)
            failure_time = failure_times.get(node_id, math.inf)
            if row["time_out"] > failure_time:
                # penalize makespan and reliability when failure window is crossed
                failure_penalty = max(failure_penalty, 500.0)
                reliability_log_sum += -5.0

        makespan += failure_penalty

        return (float(makespan), float(total_energy), float(reliability_log_sum))

    except Exception:
        if os.path.exists(log_file):
            os.remove(log_file)
        return (999999, 999999, -999999)


# ==========================================
# GA SETUP
# ==========================================
def register_deap_classes():
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)


def init_individual():
    job_seq = random.sample(range(NUM_JOBS), NUM_JOBS)
    machine_assignments = [
        random.choice(OP_SPECS[i]["eligible_nodes"]) for i in range(NUM_OPS)
    ]
    return creator.Individual(job_seq + machine_assignments)


def op_balanced(offspring):
    # Ordered crossover on job sequence, uniform swap on machine choices
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.7:
            seg1, seg2 = c1[:NUM_JOBS], c2[:NUM_JOBS]
            tools.cxOrdered(seg1, seg2)
            c1[:NUM_JOBS] = seg1
            c2[:NUM_JOBS] = seg2
        for i in range(NUM_JOBS, len(c1)):
            if random.random() < 0.2:
                c1[i], c2[i] = c2[i], c1[i]
    for m in offspring:
        if random.random() < 0.3:
            seg = m[:NUM_JOBS]
            tools.mutShuffleIndexes(seg, indpb=0.05)
            m[:NUM_JOBS] = seg
        for idx in range(NUM_JOBS, len(m)):
            if random.random() < 0.05:
                m[idx] = random.choice(OP_SPECS[idx - NUM_JOBS]["eligible_nodes"])


def op_explore(offspring):
    # More disruptive crossover/mutation
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.9:
            seg1, seg2 = c1[:NUM_JOBS], c2[:NUM_JOBS]
            tools.cxPartialyMatched(seg1, seg2)
            c1[:NUM_JOBS] = seg1
            c2[:NUM_JOBS] = seg2
        for i in range(NUM_JOBS, len(c1)):
            if random.random() < 0.5:
                c1[i], c2[i] = c2[i], c1[i]
    for m in offspring:
        if random.random() < 0.8:
            seg = m[:NUM_JOBS]
            tools.mutShuffleIndexes(seg, indpb=0.3)
            m[:NUM_JOBS] = seg
        for idx in range(NUM_JOBS, len(m)):
            if random.random() < 0.3:
                m[idx] = random.choice(OP_SPECS[idx - NUM_JOBS]["eligible_nodes"])


def op_exploit(offspring):
    # Conservative refinement
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.6:
            seg1, seg2 = c1[:NUM_JOBS], c2[:NUM_JOBS]
            tools.cxOrdered(seg1, seg2)
            c1[:NUM_JOBS] = seg1
            c2[:NUM_JOBS] = seg2
    for m in offspring:
        if random.random() < 0.1:
            seg = m[:NUM_JOBS]
            tools.mutShuffleIndexes(seg, indpb=0.02)
            m[:NUM_JOBS] = seg
        for idx in range(NUM_JOBS, len(m)):
            if random.random() < 0.02:
                m[idx] = random.choice(OP_SPECS[idx - NUM_JOBS]["eligible_nodes"])


def build_toolbox():
    register_deap_classes()
    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_schedule)
    toolbox.register("select_parents", tools.selTournament, tournsize=3)
    toolbox.register("select_survivors", tools.selNSGA2)
    return toolbox


RL_ACTIONS = [op_balanced, op_explore, op_exploit]


# ==========================================
# TRAINING LOOP
# ==========================================
def run_rl_optimizer(args):
    print("--- STARTING RL-AOS-GA OPTIMIZER ---")
    os.makedirs("logs", exist_ok=True)
    toolbox = build_toolbox()

    log_filename = os.path.join(args.logdir, "training_log.csv")
    pareto_filename = os.path.join(args.logdir, "final_pareto.csv")
    os.makedirs(args.logdir, exist_ok=True)
    with open(log_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Generation",
                "Best_Makespan",
                "Best_Energy",
                "Best_Reliability",
                "Action",
                "Reward",
                "State_Div",
                "State_Imp",
                "State_HV",
                "Hypervolume",
            ]
        )

    pop = toolbox.population(n=args.pop_size)
    agent = QLearningAgent(actions=RL_ACTIONS)

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    pop = toolbox.select_survivors(pop, len(pop))
    best_makespan_history = [min(ind.fitness.values[0] for ind in pop)]
    hv_history = []

    # Reference point for HV: use max of population scaled up
    ref_m = max(ind.fitness.values[0] for ind in pop) * 1.2
    ref_e = max(ind.fitness.values[1] for ind in pop) * 1.2
    ref_r = max(-ind.fitness.values[2] for ind in pop) * 1.2
    hv = hypervolume_minimize(pop, (ref_m, ref_e, ref_r))
    hv_history.append(hv)

    for g in range(args.generations):
        state = agent.get_state(pop, best_makespan_history, hv_history)
        action_idx = agent.choose_action(state)
        agent.last_state = state
        agent.last_action = action_idx
        act_name = ["Balanced", "Explore", "Exploit"][action_idx]

        offspring = toolbox.select_parents(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        RL_ACTIONS[action_idx](offspring)

        # Invalidate fitness
        for ind in offspring:
            del ind.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        combined_pop = pop + offspring
        pop[:] = toolbox.select_survivors(combined_pop, len(pop))

        current_best_makespan = min(ind.fitness.values[0] for ind in pop)
        improvement = best_makespan_history[-1] - current_best_makespan

        ref_m = max(ind.fitness.values[0] for ind in pop) * 1.2
        ref_e = max(ind.fitness.values[1] for ind in pop) * 1.2
        ref_r = max(-ind.fitness.values[2] for ind in pop) * 1.2
        hv = hypervolume_minimize(pop, (ref_m, ref_e, ref_r))
        hv_history.append(hv)

        if improvement > 0:
            reward = 5 + improvement + (hv_history[-1] - hv_history[-2])
        elif improvement == 0:
            reward = -1 + (hv_history[-1] - hv_history[-2])
        else:
            reward = -5

        agent.learn(state, reward)
        best_makespan_history.append(current_best_makespan)

        best_gen_ind = pop[0]
        with open(log_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    g,
                    current_best_makespan,
                    best_gen_ind.fitness.values[1],
                    best_gen_ind.fitness.values[2],
                    act_name,
                    reward,
                    state[0],
                    state[1],
                    state[2],
                    hv,
                ]
            )

        print(
            f"Gen {g}: Act={act_name}, BestMake={current_best_makespan:.1f}, Energy={best_gen_ind.fitness.values[1]:.1f}, HV={hv:.2f}"
        )

    with open(pareto_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Makespan", "Energy", "Reliability", "Schedule"])
        for ind in pop:
            writer.writerow([ind.fitness.values[0], ind.fitness.values[1], ind.fitness.values[2], list(ind)])

    print(f"Data saved to {log_filename} and {pareto_filename}")


def run_nsga_baseline(args):
    print("--- STARTING STATIC NSGA-II BASELINE ---")
    os.makedirs(args.logdir, exist_ok=True)
    toolbox = build_toolbox()
    log_filename = os.path.join(args.logdir, "baseline_log.csv")
    pareto_filename = os.path.join(args.logdir, "baseline_pareto.csv")

    pop = toolbox.population(n=args.pop_size)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    pop = toolbox.select_survivors(pop, len(pop))

    for g in range(args.generations):
        offspring = toolbox.select_parents(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        op_balanced(offspring)

        for ind in offspring:
            del ind.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        combined_pop = pop + offspring
        pop[:] = toolbox.select_survivors(combined_pop, len(pop))

    with open(log_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Makespan", "Energy", "Reliability"])
        for ind in pop:
            writer.writerow(ind.fitness.values)

    with open(pareto_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Makespan", "Energy", "Reliability", "Schedule"])
        for ind in pop:
            writer.writerow([ind.fitness.values[0], ind.fitness.values[1], ind.fitness.values[2], list(ind)])
    print(f"Baseline data saved to {log_filename} and {pareto_filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="RL-AOS-GA for MO-FJSP on YAFS")
    parser.add_argument("--mode", choices=["rl", "nsga"], default="rl", help="Optimizer mode")
    parser.add_argument("--generations", type=int, default=30, help="Number of generations")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for logs")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "rl":
        run_rl_optimizer(args)
    else:
        run_nsga_baseline(args)


if __name__ == "__main__":
    main()
