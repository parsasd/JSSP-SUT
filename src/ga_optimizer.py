import os
import random
import numpy as np
import pandas as pd
import networkx as nx
from deap import base, creator, tools, algorithms
import csv

# YAFS IMPORTS
import yafs.core
from yafs.topology import Topology
from yafs.selection import First_ShortestPath
from simple_population import SimplePopulation
from yafs_patch import apply_yafs_patches

# CUSTOM IMPORTS
from job_factory import JSSPWorkload
from placement import JSSPPlacement
from workload import parse_taillard, get_google_cluster_resources, ensure_datasets

# ==========================================
# 1. RL-AOS AGENT
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

    def get_state(self, population, fitness_history):
        unique_inds = len(set(tuple(ind) for ind in population))
        diversity_ratio = unique_inds / len(population)
        if diversity_ratio < 0.3: div_state = "Low"
        elif diversity_ratio < 0.7: div_state = "Med"
        else: div_state = "High"

        if len(fitness_history) < 2:
            imp_state = "Fast"
        else:
            delta = fitness_history[-2] - fitness_history[-1]
            if delta < 0.5: imp_state = "Stagnant"
            elif delta < 20.0: imp_state = "Slow"
            else: imp_state = "Fast"
        return (div_state, imp_state)

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if random.random() < self.epsilon:
            return random.choice(range(len(self.actions))) 
        else:
            return np.argmax(self.q_table[state])

    def learn(self, current_state, reward):
        if self.last_state is not None:
            old_q = self.q_table[self.last_state][self.last_action]
            max_future_q = np.max(self.q_table.get(current_state, np.zeros(len(self.actions))))
            new_q = old_q + self.lr * (reward + self.gamma * max_future_q - old_q)
            self.q_table[self.last_state][self.last_action] = new_q
        self.epsilon = max(0.05, self.epsilon * 0.98) 

# ==========================================
# 2. DATASET LOADING
# ==========================================
TA_FILE = ensure_datasets()
GLOBAL_RAW_JOBS = parse_taillard(TA_FILE)
NUM_MACHINES_REQ = 15
random.seed(42)
GLOBAL_NODE_PROFILES = get_google_cluster_resources(NUM_MACHINES_REQ)

# ==========================================
# 3. SIMULATION COMPONENTS
# ==========================================
class QuietShortestPath(First_ShortestPath):
    def get_path(self, sim, app_name, message, topology_src, alloc_DES, alloc_module, traffic, from_des):
        try:
            des_list = alloc_module[app_name][message.dst]
            dst_node = alloc_DES[des_list[0]]
            path = list(nx.shortest_path(sim.topology.G, source=topology_src, target=dst_node))
            return [path], [des_list[0]]
        except Exception:
            return [], []

def create_topology(node_profiles):
    G = nx.Graph()
    # Cloud node (Controller) - ID 0
    G.add_node(0, IPT=10, RAM=64000, static_power=200, power_alpha=0.01) 
    
    # Edge nodes - IDs 1 to 15
    for m_id, props in node_profiles.items():
        node_id = m_id + 1
        G.add_node(node_id, **props)

    for node_id in range(1, len(node_profiles) + 1):
        G.add_edge(0, node_id, BW=1000, PR=0) 
    t = Topology()
    t.G = G
    return t

def evaluate_schedule(individual):
    if not GLOBAL_RAW_JOBS: return (999999, 999999, -999999)
    apply_yafs_patches()
    
    release_times = {}
    current_time = 1
    gap = 10 
    for job_id in individual:
        release_times[job_id] = current_time
        current_time += gap

    workload_factory = JSSPWorkload(GLOBAL_RAW_JOBS)
    app = workload_factory.create_application(release_times)

    log_file = f"logs/log_eval_{os.getpid()}_{random.randint(0,999999)}.csv"
    
    topo = create_topology(GLOBAL_NODE_PROFILES)
    s = yafs.core.Sim(topo, default_results_path=log_file.replace(".csv", ""))
    pop_policy = SimplePopulation(node_id=0, name="SimplePop")
    s.deploy_app(app, JSSPPlacement(name="JSSP"), QuietShortestPath())
    s.population_policy[pop_policy.name] = {"population_policy": pop_policy, "apps": [app.name]}
    
    s.run(until=200000)
    
    try:
        df = pd.read_csv(log_file)
        if os.path.exists(log_file): os.remove(log_file)
        if df.empty: return (999999, 999999, -999999)

        # 1. MAKESPAN
        makespan = df['time_out'].max()

        # 2. ENERGY MODEL (Cluster-Wide Integration)
        # E_total = E_static + E_dynamic
        
        # A. Static Energy: Cost of keeping the whole cluster ON for the duration of Makespan
        # Calculate sum of static power of all nodes (Cloud + Edge)
        cluster_static_power = sum(d.get('static_power', 0) for n, d in topo.G.nodes(data=True))
        static_energy = cluster_static_power * makespan

        # B. Dynamic Energy: Cost of processing tasks (Load-dependent)
        dynamic_energy = 0.0
        processing_events = df[df['time_out'] > df['time_in']]
        
        for _, row in processing_events.iterrows():
            duration = row['time_out'] - row['time_in']
            node_id = int(row['TOPO.dst'])
            node_attrs = topo.G.nodes[node_id]
            
            p_alpha = node_attrs.get('power_alpha', 0.0)
            ipt = node_attrs.get('IPT', 1.0)
            
            # Dynamic Power component only
            p_dynamic = (p_alpha * (ipt ** 3) * 50) 
            dynamic_energy += p_dynamic * duration

        total_energy = static_energy + dynamic_energy

        # 3. RELIABILITY
        reliability_log_sum = 0.0
        for _, row in processing_events.iterrows():
            duration = row['time_out'] - row['time_in']
            node_id = int(row['TOPO.dst'])
            node_attrs = topo.G.nodes[node_id]
            lam = node_attrs.get('failure_rate', 0.001)
            reliability_log_sum += (-lam * duration)

        return (float(makespan), float(total_energy), float(reliability_log_sum))

    except Exception:
        if os.path.exists(log_file): os.remove(log_file)
        return (999999, 999999, -999999)

# ==========================================
# 4. GA SETUP
# ==========================================
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
NUM_JOBS = len(GLOBAL_RAW_JOBS)
toolbox.register("indices", random.sample, range(NUM_JOBS), NUM_JOBS)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_schedule)
toolbox.register("select_parents", tools.selTournament, tournsize=3)
toolbox.register("select_survivors", tools.selNSGA2)

def op_balanced(offspring):
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.7: tools.cxOrdered(c1, c2)
    for m in offspring:
        if random.random() < 0.2: tools.mutShuffleIndexes(m, indpb=0.05)

def op_explore(offspring):
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.5: tools.cxPartialyMatched(c1, c2)
    for m in offspring:
        if random.random() < 0.8: tools.mutShuffleIndexes(m, indpb=0.3)

def op_exploit(offspring):
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.9: tools.cxOrdered(c1, c2)
    for m in offspring:
        if random.random() < 0.05: tools.mutShuffleIndexes(m, indpb=0.05)

RL_ACTIONS = [op_balanced, op_explore, op_exploit]

def main():
    print("--- STARTING RL-AOS-GA OPTIMIZER (FINAL FIXED RUN) ---")
    if not os.path.exists("logs"): os.makedirs("logs")
    
    log_filename = "logs/training_log.csv"
    pareto_filename = "logs/final_pareto.csv"
    
    with open(log_filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Best_Makespan", "Best_Energy", "Best_Reliability", "Action", "Reward", "State_Div", "State_Imp"])

    pop = toolbox.population(n=20)
    agent = QLearningAgent(actions=RL_ACTIONS)
    
    print("Evaluating Initial Population...")
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
    
    pop = toolbox.select_survivors(pop, len(pop))
    best_makespan_history = [min(ind.fitness.values[0] for ind in pop)]
    
    NGEN = 50 
    for g in range(NGEN):
        state = agent.get_state(pop, best_makespan_history)
        action_idx = agent.choose_action(state)
        agent.last_state = state
        agent.last_action = action_idx
        act_name = ['Balanced', 'Explore', 'Exploit'][action_idx]
        
        offspring = toolbox.select_parents(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        RL_ACTIONS[action_idx](offspring)

        for ind in offspring:
            if ind.fitness.valid: del ind.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        combined_pop = pop + offspring
        pop[:] = toolbox.select_survivors(combined_pop, len(pop))
        
        current_best_makespan = min(ind.fitness.values[0] for ind in pop)
        improvement = best_makespan_history[-1] - current_best_makespan
        
        if improvement > 0: reward = 10 + improvement
        elif improvement == 0: reward = -1
        else: reward = -5 
            
        agent.learn(state, reward)
        best_makespan_history.append(current_best_makespan)
        
        best_gen_ind = pop[0]
        # LOGGING
        with open(log_filename, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([g, current_best_makespan, best_gen_ind.fitness.values[1], best_gen_ind.fitness.values[2], act_name, reward, state[0], state[1]])
        
        print(f"Gen {g}: Act={act_name}, BestMake={current_best_makespan:.1f}, Energy={best_gen_ind.fitness.values[1]:.1f}")

    print("\nOPTIMIZATION COMPLETE.")
    with open(pareto_filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Makespan", "Energy", "Reliability", "Schedule"])
        for ind in pop:
            writer.writerow([ind.fitness.values[0], ind.fitness.values[1], ind.fitness.values[2], str(ind)])
            
    print(f"Data saved to {log_filename} and {pareto_filename}")

if __name__ == "__main__":
    main()