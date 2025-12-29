import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_training_results():
    log_path = "logs/training_log.csv"
    pareto_path = "logs/final_pareto.csv"
    
    if not os.path.exists(log_path):
        print("Log files not found. Run ga_optimizer.py first.")
        return

    # Load Data
    df_log = pd.read_csv(log_path)
    df_pareto = pd.read_csv(pareto_path)

    # Set graphical style
    sns.set_theme(style="whitegrid")

    # 1. CONVERGENCE PLOT (Makespan vs Generation)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_log, x="Generation", y="Best_Makespan", marker="o", color="b")
    plt.title("Convergence of Makespan (RL-AOS-GA)", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Makespan (Time Units)", fontsize=12)
    plt.savefig("logs/plot_convergence.png", dpi=300)
    print("Saved logs/plot_convergence.png")

    # 2. PARETO FRONT (Energy vs Makespan)
    plt.figure(figsize=(10, 6))
    # Scatter plot of the final population
    scatter = sns.scatterplot(
        data=df_pareto, 
        x="Makespan", 
        y="Energy", 
        hue="Reliability", 
        size="Reliability", 
        palette="viridis", 
        sizes=(50, 200)
    )
    plt.title("Pareto Front: Trade-off Analysis", fontsize=14)
    plt.xlabel("Makespan (Lower is Better)", fontsize=12)
    plt.ylabel("Energy Consumption (Lower is Better)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("logs/plot_pareto.png", dpi=300)
    print("Saved logs/plot_pareto.png")

    # 3. RL ACTION DISTRIBUTION
    plt.figure(figsize=(8, 5))
    # FIX: Assigned 'x' variable to 'hue' and set legend=False to fix Warning
    sns.countplot(data=df_log, x="Action", hue="Action", palette="Set2", legend=False)
    plt.title("Adaptive Operator Selection (RL Decisions)", fontsize=14)
    plt.xlabel("Genetic Operator Strategy", fontsize=12)
    plt.ylabel("Frequency Selected", fontsize=12)
    plt.savefig("logs/plot_actions.png", dpi=300)
    print("Saved logs/plot_actions.png")

if __name__ == "__main__":
    plot_training_results()