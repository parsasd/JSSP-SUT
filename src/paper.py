"""
Paper-focused experiment runner + plotting utilities.

This script is meant to generate *paper-quality* figures by:
  - Running multiple seeds for RL-AOS-GA and the static NSGA-II baseline
  - Aggregating results across seeds (median + IQR bands)
  - Producing comparative convergence, hypervolume, Pareto, and stability plots

Examples (from repo root):
  python src/paper.py run  --seeds 0-4 --instances ta01-ta03 --generations 50 --pop-size 30 --outdir paper_runs
  python src/paper.py run  --seeds 0-9 --instances ta01 --modes rl,nsga,rl_random,rl_no_hv --outdir paper_runs
  python src/paper.py plot --indir paper_runs --outdir paper_figures
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


def _parse_seeds(spec: str) -> list[int]:
    """
    Parse seeds like:
      "0,1,2"
      "0-9"
      "0-4,10,42"
    """
    seeds: list[int] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        if "-" in part:
            start_s, end_s = (x.strip() for x in part.split("-", 1))
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid seed range '{part}'")
            seeds.extend(list(range(start, end + 1)))
        else:
            seeds.append(int(part))
    # Preserve order but drop duplicates
    return list(dict.fromkeys(seeds))


def _parse_instances(spec: str) -> list[str]:
    """
    Parse instance specs like:
      "ta01,ta02"
      "ta01-ta05"
      "data/taillard_instances/ta01.txt"
    """
    instances: list[str] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        if "-" in part and part.startswith("ta"):
            left, right = (x.strip() for x in part.split("-", 1))
            if left.startswith("ta") and right.startswith("ta"):
                try:
                    left_num = int(left[2:])
                    right_num = int(right[2:])
                except ValueError:
                    instances.append(part)
                    continue
                width = max(len(left[2:]), len(right[2:]))
                for n in range(left_num, right_num + 1):
                    instances.append(f"ta{n:0{width}d}")
                continue
        instances.append(part)
    return list(dict.fromkeys(instances))


@dataclass(frozen=True)
class RunSpec:
    mode: str  # "rl", "nsga", "rl_random", "rl_no_hv", ...
    seed: int
    generations: int
    pop_size: int
    instance: str
    instance_path: str
    hv_ref_samples: int
    hv_ref_scale: float
    hv_ref_seed: int
    outdir: Path

    @property
    def run_dir(self) -> Path:
        return self.outdir / self.instance / self.mode / f"seed_{self.seed}"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run_one(spec: RunSpec, skip_existing: bool) -> None:
    spec.run_dir.mkdir(parents=True, exist_ok=True)

    # Skip if output files are present (helps when iterating on plots).
    if skip_existing:
        if spec.mode == "nsga":
            expected = [
                spec.run_dir / "baseline_training_log.csv",
                spec.run_dir / "baseline_pareto.csv",
            ]
        else:
            expected = [spec.run_dir / "training_log.csv", spec.run_dir / "final_pareto.csv"]
        if all(p.exists() for p in expected):
            return

    from ga_optimizer import run_rl_optimizer, run_nsga_baseline  # local import to keep CLI snappy
    import random

    random.seed(spec.seed)
    np.random.seed(spec.seed)

    mode_to_policy = {
        "rl": "learned",
        "rl_random": "random",
        "rl_no_hv": "no_hv",
        "rl_fixed_balanced": "fixed_balanced",
        "rl_fixed_explore": "fixed_explore",
        "rl_fixed_exploit": "fixed_exploit",
    }

    args = SimpleNamespace(
        generations=spec.generations,
        pop_size=spec.pop_size,
        seed=spec.seed,
        logdir=str(spec.run_dir),
        instance=spec.instance_path,
        policy=mode_to_policy.get(spec.mode, "learned"),
        hv_ref_samples=spec.hv_ref_samples,
        hv_ref_scale=spec.hv_ref_scale,
        hv_ref_seed=spec.hv_ref_seed,
    )

    started = time.perf_counter()
    if spec.mode == "nsga":
        run_nsga_baseline(args)
    else:
        run_rl_optimizer(args)
    elapsed_s = time.perf_counter() - started

    _write_json(
        spec.run_dir / "run_config.json",
        {
            "mode": spec.mode,
            "seed": spec.seed,
            "generations": spec.generations,
            "pop_size": spec.pop_size,
            "elapsed_s": elapsed_s,
            "instance": spec.instance,
            "instance_path": spec.instance_path,
            "policy": mode_to_policy.get(spec.mode, ""),
            "hv_ref_samples": spec.hv_ref_samples,
            "hv_ref_scale": spec.hv_ref_scale,
            "hv_ref_seed": spec.hv_ref_seed,
        },
    )


def cmd_run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    seeds = _parse_seeds(args.seeds)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    instances = _parse_instances(args.instances)

    from workload import ensure_taillard_instance

    for instance_spec in instances:
        instance_path = ensure_taillard_instance(instance_spec)
        instance_name = Path(instance_path).stem
        for mode in modes:
            for seed in seeds:
                _run_one(
                    RunSpec(
                        mode=mode,
                        seed=seed,
                        generations=args.generations,
                        pop_size=args.pop_size,
                        instance=instance_name,
                        instance_path=instance_path,
                        hv_ref_samples=args.hv_ref_samples,
                        hv_ref_scale=args.hv_ref_scale,
                        hv_ref_seed=args.hv_ref_seed,
                        outdir=outdir,
                    ),
                    skip_existing=args.skip_existing,
                )


def _collect_runs(indir: Path, mode: str) -> list[Path]:
    mode_dir = indir / mode
    if not mode_dir.exists():
        return []
    return sorted([p for p in mode_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])


def _load_training_log(run_dir: Path, mode: str) -> pd.DataFrame | None:
    if mode == "nsga":
        path = run_dir / "baseline_training_log.csv"
    else:
        path = run_dir / "training_log.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["seed"] = int(run_dir.name.replace("seed_", ""))
    df["mode"] = mode
    return df


def _load_pareto(run_dir: Path, mode: str) -> pd.DataFrame | None:
    path = run_dir / ("baseline_pareto.csv" if mode == "nsga" else "final_pareto.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["seed"] = int(run_dir.name.replace("seed_", ""))
    df["mode"] = mode
    return df


def _load_run_config(run_dir: Path) -> dict | None:
    path = run_dir / "run_config.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _cumulative_best(series: pd.Series, maximize: bool = False) -> pd.Series:
    if maximize:
        return series.cummax()
    return series.cummin()


def _aggregate_by_generation(
    runs: list[pd.DataFrame],
    value_col: str,
    maximize: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by generation with columns:
      median, q25, q75
    Uses best-so-far for minimization objectives.
    """
    if not runs:
        return pd.DataFrame()

    aligned = []
    for df in runs:
        d = df.sort_values("Generation").copy()
        d[value_col] = _cumulative_best(d[value_col], maximize=maximize)
        s = d.set_index("Generation")[value_col]
        s.name = f"seed_{int(df['seed'].iloc[0])}"
        aligned.append(s)

    mat = pd.concat(aligned, axis=1).sort_index()
    out = pd.DataFrame(index=mat.index)
    out["median"] = mat.median(axis=1)
    out["q25"] = mat.quantile(0.25, axis=1)
    out["q75"] = mat.quantile(0.75, axis=1)
    return out


def _pareto_front_2d(
    df: pd.DataFrame,
    x: str,
    y: str,
    maximize_x: bool = False,
    maximize_y: bool = False,
) -> pd.DataFrame:
    """Compute a 2D Pareto front for mixed objectives."""
    if df.empty:
        return df
    work = df[[x, y]].copy()
    if maximize_x:
        work[x] = -work[x]
    if maximize_y:
        work[y] = -work[y]
    ordered = work.sort_values([x, y], ascending=[True, True])
    best_y = np.inf
    keep_idx = []
    for idx, row in ordered.iterrows():
        if row[y] < best_y:
            keep_idx.append(idx)
            best_y = row[y]
    return df.loc[keep_idx].reset_index(drop=True)


def _pareto_front_nd(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    keep = []
    for i in range(points.shape[0]):
        dominated = False
        for j in range(points.shape[0]):
            if i == j:
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return points[keep]


def _compute_igd(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.size == 0 or candidate.size == 0:
        return float("nan")
    dists = []
    for ref in reference:
        dists.append(float(np.min(np.linalg.norm(candidate - ref, axis=1))))
    return float(np.mean(dists))


def _best_by_seed(pareto_dfs: list[pd.DataFrame], mode_label: str) -> pd.DataFrame:
    rows = []
    for df in pareto_dfs:
        seed = int(df["seed"].iloc[0])
        best_m = df.loc[df["Makespan"].idxmin()]
        best_e = df.loc[df["Energy"].idxmin()]
        best_r = df.loc[df["Reliability"].idxmax()]
        rows.append(
            {
                "mode": mode_label,
                "seed": seed,
                "best_makespan": float(best_m["Makespan"]),
                "best_energy": float(best_e["Energy"]),
                "best_reliability": float(best_r["Reliability"]),
            }
        )
    return pd.DataFrame(rows)


def _best_hv_by_seed(logs: list[pd.DataFrame], mode_label: str) -> pd.DataFrame:
    rows = []
    for df in logs:
        rows.append(
            {
                "mode": mode_label,
                "seed": int(df["seed"].iloc[0]),
                "best_hv": float(df["Hypervolume"].max()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_metrics(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        for mode in summary["mode"].unique():
            data = summary.loc[summary["mode"] == mode, metric].dropna()
            rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "n": int(data.shape[0]),
                    "mean": float(data.mean()),
                    "std": float(data.std(ddof=1)),
                    "median": float(data.median()),
                    "q25": float(data.quantile(0.25)),
                    "q75": float(data.quantile(0.75)),
                    "min": float(data.min()),
                    "max": float(data.max()),
                }
            )
    return pd.DataFrame(rows)


def _cliffs_delta(x: pd.Series, y: pd.Series) -> float:
    x = x.to_numpy()
    y = y.to_numpy()
    if x.size == 0 or y.size == 0:
        return float("nan")
    gt = 0
    lt = 0
    for xi in x:
        gt += int((xi > y).sum())
        lt += int((xi < y).sum())
    return (gt - lt) / (x.size * y.size)


def _perm_test_median_diff(
    x: pd.Series,
    y: pd.Series,
    n_perm: int = 10000,
    seed: int = 7,
) -> tuple[float, float]:
    x = x.to_numpy()
    y = y.to_numpy()
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    combined = np.concatenate([x, y])
    n_x = x.size
    observed = float(np.median(x) - np.median(y))
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        diff = float(np.median(perm[:n_x]) - np.median(perm[n_x:]))
        if abs(diff) >= abs(observed):
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return observed, p_value


def cmd_plot(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )

    default_mode_specs = {
        "rl": {"label": "RL-AOS-GA", "color": "#1f77b4"},
        "nsga": {"label": "Static NSGA-II", "color": "#ff7f0e"},
        "rl_random": {"label": "RL (Random Operator)", "color": "#2ca02c"},
        "rl_no_hv": {"label": "RL (No HV Reward)", "color": "#d62728"},
        "rl_fixed_balanced": {"label": "RL (Fixed Balanced)", "color": "#8c564b"},
        "rl_fixed_explore": {"label": "RL (Fixed Explore)", "color": "#17becf"},
        "rl_fixed_exploit": {"label": "RL (Fixed Exploit)", "color": "#bcbd22"},
    }

    def _boxplot(summary: pd.DataFrame, y_col: str, title: str, ylabel: str, filename: str, root: Path) -> None:
        plt.figure(figsize=(6.2, 3.8))
        sns.boxplot(data=summary, x="mode", y=y_col, showfliers=False)
        sns.stripplot(data=summary, x="mode", y=y_col, color="black", size=3, jitter=0.15, alpha=0.6)
        plt.title(title)
        plt.xlabel("")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(root / f"{filename}.png")
        plt.savefig(root / f"{filename}.pdf")
        plt.close()

    def _discover_instances(root: Path) -> dict[str, Path]:
        def _is_instance_dir(path: Path) -> bool:
            for child in path.iterdir():
                if not child.is_dir() or child.name.startswith("seed_"):
                    continue
                if any(grand.is_dir() and grand.name.startswith("seed_") for grand in child.iterdir()):
                    return True
            return False

        instance_dirs = {p.name: p for p in root.iterdir() if p.is_dir() and _is_instance_dir(p)}
        if instance_dirs:
            return instance_dirs
        if (root / "rl").exists() or (root / "nsga").exists():
            return {"default": root}
        return {p.name: p for p in root.iterdir() if p.is_dir()}

    def _build_mode_specs(modes: list[str]) -> dict[str, dict[str, str]]:
        palette = sns.color_palette("tab10", n_colors=max(10, len(modes)))
        specs = {}
        pal_idx = 0
        for mode in modes:
            if mode in default_mode_specs:
                specs[mode] = default_mode_specs[mode]
            else:
                specs[mode] = {
                    "label": mode.replace("_", " ").upper(),
                    "color": palette[pal_idx],
                }
                pal_idx += 1
        return specs

    def _plot_instance(instance_name: str, instance_dir: Path, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        modes = sorted([p.name for p in instance_dir.iterdir() if p.is_dir()])
        if not modes:
            return
        mode_specs = _build_mode_specs(modes)

        logs_by_mode: dict[str, list[pd.DataFrame]] = {}
        pareto_by_mode: dict[str, list[pd.DataFrame]] = {}
        run_dirs_by_mode: dict[str, list[Path]] = {}
        for mode in modes:
            run_dirs = _collect_runs(instance_dir, mode)
            run_dirs_by_mode[mode] = run_dirs
            logs_by_mode[mode] = [d for d in (_load_training_log(p, mode) for p in run_dirs) if d is not None]
            pareto_by_mode[mode] = [d for d in (_load_pareto(p, mode) for p in run_dirs) if d is not None]

        convergence_specs = [
            ("Best_Makespan", "Convergence (Best-So-Far Makespan)", "Makespan (lower is better)", False, "fig_convergence_makespan"),
            ("Best_Energy", "Convergence (Best-So-Far Energy)", "Energy (lower is better)", False, "fig_convergence_energy"),
            ("Best_Reliability", "Convergence (Best-So-Far Reliability)", "Reliability (higher is better)", True, "fig_convergence_reliability"),
            ("Hypervolume", "Convergence (Hypervolume Proxy, Best-So-Far)", "Hypervolume (higher is better)", True, "fig_convergence_hypervolume"),
        ]

        for metric, title, ylabel, maximize, fname in convergence_specs:
            plt.figure(figsize=(6.2, 3.8))
            plotted = False
            for mode in modes:
                agg = _aggregate_by_generation(logs_by_mode.get(mode, []), metric, maximize=maximize)
                if agg.empty:
                    continue
                plt.plot(agg.index, agg["median"], label=mode_specs[mode]["label"], color=mode_specs[mode]["color"])
                plt.fill_between(agg.index, agg["q25"], agg["q75"], alpha=0.2, color=mode_specs[mode]["color"])
                plotted = True
            if plotted:
                plt.title(title)
                plt.xlabel("Generation")
                plt.ylabel(ylabel)
                plt.legend()
                plt.tight_layout()
                plt.savefig(root / f"{fname}.png")
                plt.savefig(root / f"{fname}.pdf")
            plt.close()

        if "rl" in modes and "nsga" in modes:
            rl_all = pd.concat(pareto_by_mode["rl"], ignore_index=True) if pareto_by_mode["rl"] else pd.DataFrame()
            nsga_all = pd.concat(pareto_by_mode["nsga"], ignore_index=True) if pareto_by_mode["nsga"] else pd.DataFrame()

            if not rl_all.empty or not nsga_all.empty:
                plt.figure(figsize=(6.2, 4.2))
                if not rl_all.empty:
                    plt.scatter(
                        rl_all["Makespan"],
                        rl_all["Energy"],
                        s=15,
                        alpha=0.2,
                        color=mode_specs["rl"]["color"],
                        label=f"{mode_specs['rl']['label']} (all)",
                    )
                    rl_front = _pareto_front_2d(rl_all, "Makespan", "Energy")
                    plt.plot(
                        rl_front["Makespan"],
                        rl_front["Energy"],
                        color=mode_specs["rl"]["color"],
                        linewidth=2.0,
                        label=f"{mode_specs['rl']['label']} (front)",
                    )
                if not nsga_all.empty:
                    plt.scatter(
                        nsga_all["Makespan"],
                        nsga_all["Energy"],
                        s=15,
                        alpha=0.2,
                        color=mode_specs["nsga"]["color"],
                        label=f"{mode_specs['nsga']['label']} (all)",
                    )
                    nsga_front = _pareto_front_2d(nsga_all, "Makespan", "Energy")
                    plt.plot(
                        nsga_front["Makespan"],
                        nsga_front["Energy"],
                        color=mode_specs["nsga"]["color"],
                        linewidth=2.0,
                        label=f"{mode_specs['nsga']['label']} (front)",
                    )
                plt.title("Pareto Trade-off (Makespan vs Energy)")
                plt.xlabel("Makespan (lower is better)")
                plt.ylabel("Energy (lower is better)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(root / "fig_pareto_overlay.png")
                plt.savefig(root / "fig_pareto_overlay.pdf")
                plt.close()

                plt.figure(figsize=(6.2, 4.2))
                if not rl_all.empty:
                    plt.scatter(
                        rl_all["Makespan"],
                        rl_all["Reliability"],
                        s=15,
                        alpha=0.2,
                        color=mode_specs["rl"]["color"],
                        label=f"{mode_specs['rl']['label']} (all)",
                    )
                    rl_front = _pareto_front_2d(rl_all, "Makespan", "Reliability", maximize_y=True)
                    plt.plot(
                        rl_front["Makespan"],
                        rl_front["Reliability"],
                        color=mode_specs["rl"]["color"],
                        linewidth=2.0,
                        label=f"{mode_specs['rl']['label']} (front)",
                    )
                if not nsga_all.empty:
                    plt.scatter(
                        nsga_all["Makespan"],
                        nsga_all["Reliability"],
                        s=15,
                        alpha=0.2,
                        color=mode_specs["nsga"]["color"],
                        label=f"{mode_specs['nsga']['label']} (all)",
                    )
                    nsga_front = _pareto_front_2d(nsga_all, "Makespan", "Reliability", maximize_y=True)
                    plt.plot(
                        nsga_front["Makespan"],
                        nsga_front["Reliability"],
                        color=mode_specs["nsga"]["color"],
                        linewidth=2.0,
                        label=f"{mode_specs['nsga']['label']} (front)",
                    )
                plt.title("Pareto Trade-off (Makespan vs Reliability)")
                plt.xlabel("Makespan (lower is better)")
                plt.ylabel("Reliability (higher is better)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(root / "fig_pareto_makespan_reliability.png")
                plt.savefig(root / "fig_pareto_makespan_reliability.pdf")
                plt.close()

                plt.figure(figsize=(6.2, 4.2))
                if not rl_all.empty:
                    plt.scatter(
                        rl_all["Energy"],
                        rl_all["Reliability"],
                        s=15,
                        alpha=0.2,
                        color=mode_specs["rl"]["color"],
                        label=f"{mode_specs['rl']['label']} (all)",
                    )
                    rl_front = _pareto_front_2d(rl_all, "Energy", "Reliability", maximize_y=True)
                    plt.plot(
                        rl_front["Energy"],
                        rl_front["Reliability"],
                        color=mode_specs["rl"]["color"],
                        linewidth=2.0,
                        label=f"{mode_specs['rl']['label']} (front)",
                    )
                if not nsga_all.empty:
                    plt.scatter(
                        nsga_all["Energy"],
                        nsga_all["Reliability"],
                        s=15,
                        alpha=0.2,
                        color=mode_specs["nsga"]["color"],
                        label=f"{mode_specs['nsga']['label']} (all)",
                    )
                    nsga_front = _pareto_front_2d(nsga_all, "Energy", "Reliability", maximize_y=True)
                    plt.plot(
                        nsga_front["Energy"],
                        nsga_front["Reliability"],
                        color=mode_specs["nsga"]["color"],
                        linewidth=2.0,
                        label=f"{mode_specs['nsga']['label']} (front)",
                    )
                plt.title("Pareto Trade-off (Energy vs Reliability)")
                plt.xlabel("Energy (lower is better)")
                plt.ylabel("Reliability (higher is better)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(root / "fig_pareto_energy_reliability.png")
                plt.savefig(root / "fig_pareto_energy_reliability.pdf")
                plt.close()

        summary_parts = []
        for mode, paretos in pareto_by_mode.items():
            if paretos:
                summary_parts.append(_best_by_seed(paretos, mode_specs[mode]["label"]))
        summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()

        hv_parts = []
        for mode, logs in logs_by_mode.items():
            if logs:
                hv_parts.append(_best_hv_by_seed(logs, mode_specs[mode]["label"]))
        hv_summary = pd.concat(hv_parts, ignore_index=True) if hv_parts else pd.DataFrame()

        igd_rows = []
        all_points = []
        for paretos in pareto_by_mode.values():
            for df in paretos:
                pts = df[["Makespan", "Energy", "Reliability"]].copy()
                pts["Reliability"] = -pts["Reliability"]
                all_points.append(pts)
        if all_points:
            all_df = pd.concat(all_points, ignore_index=True)
            mins = all_df.min().to_numpy()
            maxs = all_df.max().to_numpy()
            denom = np.where(maxs - mins == 0, 1.0, maxs - mins)
            normalized_all = (all_df.to_numpy() - mins) / denom
            reference = _pareto_front_nd(normalized_all)
            for mode, paretos in pareto_by_mode.items():
                for df in paretos:
                    seed = int(df["seed"].iloc[0])
                    cand = df[["Makespan", "Energy", "Reliability"]].copy()
                    cand["Reliability"] = -cand["Reliability"]
                    cand_norm = (cand.to_numpy() - mins) / denom
                    igd_rows.append(
                        {
                            "mode": mode_specs[mode]["label"],
                            "seed": seed,
                            "igd": _compute_igd(reference, cand_norm),
                        }
                    )
        igd_summary = pd.DataFrame(igd_rows)

        if not summary.empty:
            if not hv_summary.empty:
                summary = summary.merge(hv_summary, on=["mode", "seed"], how="left")
            if not igd_summary.empty:
                summary = summary.merge(igd_summary, on=["mode", "seed"], how="left")
            summary = summary.sort_values(["mode", "seed"])
            summary.to_csv(root / "summary_best_by_seed.csv", index=False)

            _boxplot(summary, "best_makespan", "Best Makespan Distribution Across Seeds", "Best makespan", "fig_box_best_makespan", root)
            _boxplot(summary, "best_energy", "Best Energy Distribution Across Seeds", "Best energy", "fig_box_best_energy", root)
            _boxplot(summary, "best_reliability", "Best Reliability Distribution Across Seeds", "Best reliability", "fig_box_best_reliability", root)
            if "best_hv" in summary.columns:
                _boxplot(summary, "best_hv", "Best Hypervolume Distribution Across Seeds", "Best hypervolume", "fig_box_best_hypervolume", root)
            if "igd" in summary.columns:
                _boxplot(summary, "igd", "IGD Distribution Across Seeds", "IGD (lower is better)", "fig_box_igd", root)

            metrics = ["best_makespan", "best_energy", "best_reliability"]
            if "best_hv" in summary.columns:
                metrics.append("best_hv")
            if "igd" in summary.columns:
                metrics.append("igd")
            _summarize_metrics(summary, metrics).to_csv(root / "summary_metrics.csv", index=False)

            if summary["mode"].nunique() > 1 and "nsga" in mode_specs:
                baseline_label = mode_specs["nsga"]["label"]
                stats_rows = []
                directions = {
                    "best_makespan": "lower",
                    "best_energy": "lower",
                    "best_reliability": "higher",
                    "best_hv": "higher",
                    "igd": "lower",
                }
                for metric in metrics:
                    for mode_label in summary["mode"].unique():
                        if mode_label == baseline_label:
                            continue
                        x = summary.loc[summary["mode"] == mode_label, metric].dropna()
                        y = summary.loc[summary["mode"] == baseline_label, metric].dropna()
                        if x.empty or y.empty:
                            continue
                        median_diff, p_value = _perm_test_median_diff(x, y)
                        stats_rows.append(
                            {
                                "metric": metric,
                                "comparison": f"{mode_label} vs {baseline_label}",
                                "direction": directions.get(metric, ""),
                                "n_mode": int(x.shape[0]),
                                "n_baseline": int(y.shape[0]),
                                "median_diff_mode_minus_baseline": float(median_diff),
                                "mean_diff_mode_minus_baseline": float(x.mean() - y.mean()),
                                "cliffs_delta": float(_cliffs_delta(x, y)),
                                "p_value_perm_median": float(p_value),
                            }
                        )
                if stats_rows:
                    pd.DataFrame(stats_rows).to_csv(root / "summary_stat_tests.csv", index=False)

        if not igd_summary.empty:
            igd_summary.sort_values(["mode", "seed"]).to_csv(root / "summary_igd_by_seed.csv", index=False)

        if not hv_summary.empty:
            hv_summary.sort_values(["mode", "seed"]).to_csv(root / "summary_hypervolume_by_seed.csv", index=False)

        runtime_rows = []
        for mode, run_dirs in run_dirs_by_mode.items():
            for run_dir in run_dirs:
                cfg = _load_run_config(run_dir)
                if not cfg:
                    continue
                runtime_rows.append(
                    {
                        "mode": mode_specs[mode]["label"],
                        "seed": int(run_dir.name.replace("seed_", "")),
                        "elapsed_s": float(cfg.get("elapsed_s", 0.0)),
                        "generations": int(cfg.get("generations", 0)),
                        "pop_size": int(cfg.get("pop_size", 0)),
                        "instance": instance_name,
                    }
                )
        if runtime_rows:
            runtime_df = pd.DataFrame(runtime_rows).sort_values(["mode", "seed"])
            runtime_df.to_csv(root / "summary_runtime.csv", index=False)
            _boxplot(runtime_df, "elapsed_s", "Runtime Across Seeds", "Elapsed seconds", "fig_box_runtime", root)

        if "rl" in modes:
            rl_logs = logs_by_mode.get("rl", [])
            if rl_logs and "Action" in rl_logs[0].columns:
                actions = ["Balanced", "Explore", "Exploit"]
                per_seed = []
                for df in rl_logs:
                    counts = df.groupby(["Generation", "Action"]).size().unstack(fill_value=0)
                    for a in actions:
                        if a not in counts.columns:
                            counts[a] = 0
                    counts = counts[actions]
                    frac = counts.div(counts.sum(axis=1), axis=0)
                    per_seed.append(frac)

                avg_frac = sum(per_seed) / len(per_seed)
                avg_frac = avg_frac.fillna(0.0).sort_index()

                plt.figure(figsize=(6.2, 3.8))
                plt.stackplot(
                    avg_frac.index,
                    [avg_frac[a].values for a in actions],
                    labels=actions,
                    colors=["#2ca02c", "#d62728", "#9467bd"],
                    alpha=0.85,
                )
                plt.title("RL Adaptive Operator Selection (avg across seeds)")
                plt.xlabel("Generation")
                plt.ylabel("Selection probability")
                plt.ylim(0, 1.0)
                plt.legend(loc="upper right")
                plt.tight_layout()
                plt.savefig(root / "fig_rl_action_stack.png")
                plt.savefig(root / "fig_rl_action_stack.pdf")
                plt.close()

    instances = _discover_instances(indir)
    for instance_name, instance_dir in instances.items():
        instance_outdir = outdir if instance_name == "default" else outdir / instance_name
        _plot_instance(instance_name, instance_dir, instance_outdir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run experiments and generate paper-quality plots.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run multi-seed experiments.")
    p_run.add_argument("--outdir", type=str, default="paper_runs", help="Output directory for run artifacts.")
    p_run.add_argument("--seeds", type=str, default="0-4", help="Seed spec, e.g. '0-9' or '0-4,10,42'.")
    p_run.add_argument(
        "--instances",
        type=str,
        default="ta01",
        help="Instance spec, e.g. 'ta01,ta02' or 'ta01-ta05'.",
    )
    p_run.add_argument("--generations", type=int, default=50, help="Generations per run.")
    p_run.add_argument("--pop-size", type=int, default=30, help="Population size per run.")
    p_run.add_argument(
        "--modes",
        type=str,
        default="rl,nsga",
        help="Comma-separated modes (rl, nsga, rl_random, rl_no_hv, rl_fixed_*).",
    )
    p_run.add_argument("--hv-ref-samples", type=int, default=50, help="Samples for fixed HV reference point.")
    p_run.add_argument("--hv-ref-scale", type=float, default=1.2, help="Scale for HV reference point.")
    p_run.add_argument("--hv-ref-seed", type=int, default=1337, help="Seed for HV reference sampling.")
    p_run.add_argument("--skip-existing", action="store_true", help="Skip runs that already have output files.")
    p_run.set_defaults(func=cmd_run)

    p_plot = sub.add_parser("plot", help="Aggregate runs and generate figures.")
    p_plot.add_argument("--indir", type=str, default="paper_runs", help="Input directory produced by `run`.")
    p_plot.add_argument("--outdir", type=str, default="paper_figures", help="Directory to write figures.")
    p_plot.set_defaults(func=cmd_plot)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
