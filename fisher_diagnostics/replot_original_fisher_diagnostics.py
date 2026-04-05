#!/usr/bin/env python3
"""
replot_original_fisher_diagnostics.py

Replots the Fisher-information diagnostic figures from the original
datasets included with the repository.

This script visualizes the Fisher-information diagnostic data used in the
analysis of both the cumulative Fisher-scaling behaviour and the
diagonalization of the Fisher information matrix. The data were
generated once and stored in the repository under the accompanying
``data/`` directory. The script loads these datasets and reproduces the
diagnostic plots used in the investigation.

Unlike ``plot_fisher_diagnostics.py``, which operates on data produced by
``reproduce_fisher_diagnostics_data.py``, this script operates directly on
the archived datasets distributed with the code base.

Generated figures
-----------------

The script produces five diagnostic plots:

1. **Fisher trace scaling vs total experiment time**

   Displays the scaling of the Fisher-information trace

       Tr(I)

   as a function of the total experiment time

       T_tot = Σ_k t_k

   for different spread-state ensemble sizes R.

   Each curve corresponds to one ensemble size R and is fitted with a
   power law

       Tr(I) = A T_tot^p.

   This plot is generated for a **single selected Hamiltonian family**
   (specified via ``--family``).

2. **Scaling exponent vs spread-state ensemble size**

   Displays the fitted cumulative Fisher scaling exponent

       p

   as a function of the number of spread states

       R.

   Results from all Hamiltonian families are shown together.

3. **Scaling exponent vs scheduling parameter**

   Displays the dependence of the fitted exponent

       p(R=1)

   on the time-scheduling parameter α.

   The numerical results are compared to the theoretical prediction

       p = (α γ₀ + 1) / (α + 1)

   with γ₀ = 2, corresponding to quadratic short-time Fisher scaling.

   Data from all Hamiltonian families are shown together.

4. **Fisher trace scaling vs total experiment time for varying qubit number**

   Displays the scaling of the Fisher-information trace

       Tr(I)

   as a function of the total experiment time

       T_tot = Σ_k t_k

   for different system sizes n, while keeping the spread-state ensemble
   size fixed at R=1 and the scheduling exponent fixed at α=1.

   Each curve corresponds to one qubit number and is fitted with a power
   law

       Tr(I) = A T_tot^p.

   This plot is generated for a **single selected Hamiltonian family**
   (specified via ``--family``).

5. **Fisher-matrix diagonalization vs spread-state ensemble size**

   Displays the diagonalization measure

       η_diag = ||diag(I)||_F^2 / ||I||_F^2

   as a function of the number of spread states

       R

   for different system sizes n.

   This plot visualizes how increasing the spread-state ensemble
   suppresses off-diagonal Fisher-information couplings and makes the
   Fisher matrix progressively closer to diagonal.

Data discovery
--------------

The script searches the local ``data/`` directory for files of the form

    fisher_trace_scaling_*_all_alphas.json

and automatically determines whether they belong to:

• fixed-α / varying-R experiments  
• fixed-R=1 / varying-α experiments  
• fixed-R=1 / fixed-α / varying-qubit experiments  
• fixed-α / varying-R / varying-qubit experiments

based on the directory structure and the stored data.

Usage
-----

Run the script from the directory containing the archived datasets:

    python replot_original_fisher_diagnostics.py

Optionally specify the Hamiltonian family used for the family-specific
plots:

    python replot_original_fisher_diagnostics.py --family XYZ

Notes
-----

Figures are displayed interactively and are not written to disk.
This script allows the original diagnostic figures to be reproduced
directly from the archived datasets without regenerating the data.
"""

import os
import glob
import json
import re
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FAMILY_MARKERS = {
    "Heisenberg": "o",
    "XYZ": "o",
    "XYZ2": "s",
    "XYZ3": "D",
}

FAMILY_COLORS = {
    "Heisenberg": "red",
    "XYZ": "red",
    "XYZ2": "green",
    "XYZ3": "purple",
}

plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "figure.dpi": 100,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def discover_combined_jsons(data_dir: str) -> List[str]:
    pattern = os.path.join(data_dir, "**", "fisher_trace_scaling_*_all_alphas.json")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No combined JSON files found under {data_dir}")
    return files


def classify_results(
    json_paths: List[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    fixed_alpha_vary_R = {}
    fixed_R1_vary_alpha = {}

    for path in json_paths:
        data = load_json(path)
        family = data["family"]

        alpha_keys = sorted(float(a) for a in data["alphas"].keys())
        r_values = sorted({
            int(r)
            for a in data["alphas"].values()
            for r in a["data"].keys()
        })

        if len(alpha_keys) == 1 and len(r_values) > 1:
            fixed_alpha_vary_R[family] = data
        elif len(alpha_keys) > 1 and r_values == [1]:
            fixed_R1_vary_alpha[family] = data

    return fixed_alpha_vary_R, fixed_R1_vary_alpha


def load_qubit_sweep_results(data_dir: str) -> Dict[int, Dict[str, Any]]:
    """
    Load qubit-sweep Fisher diagnostic results.

    Expects subfolders somewhere below data_dir named like:
        2_qubits, 3_qubits, ..., 8_qubits
    each containing one combined JSON:
        fisher_trace_scaling_*_all_alphas.json
    """
    out: Dict[int, Dict[str, Any]] = {}

    for root, dirs, _ in os.walk(data_dir):
        for d in dirs:
            m = re.match(r"(\d+)_qubits$", d)
            if not m:
                continue

            n = int(m.group(1))
            subdir = os.path.join(root, d)
            candidates = sorted(
                glob.glob(os.path.join(subdir, "fisher_trace_scaling_*_all_alphas.json"))
            )
            if not candidates:
                continue

            out[n] = load_json(candidates[0])

    return out


def fit_prefactor(xs: np.ndarray, ys: np.ndarray, p: float) -> float:
    lx = np.log(xs)
    ly = np.log(ys)
    return float(np.exp(np.mean(ly - p * lx)))


def plot_trace_vs_time_for_family(all_results: Dict[str, Any]) -> None:
    family = all_results["family"]
    alpha_key = sorted(all_results["alphas"].keys(), key=float)[0]
    alpha = float(alpha_key)
    results_for_alpha = all_results["alphas"][alpha_key]

    spreading_list = sorted(int(k) for k in results_for_alpha["data"].keys())
    cmap = plt.cm.viridis(np.linspace(0, 1, len(spreading_list)))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xscale("log")
    ax.set_yscale("log")

    for i, R in enumerate(spreading_list):
        entry = results_for_alpha["data"][str(R)]
        xs = np.array(entry["T_tot"], dtype=float)
        ys = np.array(entry["trace_fisher"], dtype=float)
        p_fit = entry.get("p", np.nan)
        delta_p = entry.get("delta_p", np.nan)

        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]

        color = cmap[i]
        marker = markers[i % len(markers)]

        ax.scatter(
            xs,
            ys,
            s=80,
            color=color,
            marker=marker,
            edgecolor="black",
            zorder=3,
        )

        if np.isfinite(p_fit):
            a_fit = fit_prefactor(xs, ys, p_fit)
            x_fit = np.linspace(xs.min(), xs.max(), 300)
            y_fit = a_fit * (x_fit ** p_fit)

            if np.isfinite(delta_p):
                fit_label = fr"$R={R}$, $p={p_fit:.3f}\pm{delta_p:.1e}$"
            else:
                fit_label = fr"$R={R}$, $p={p_fit:.2f}$"

            ax.plot(
                x_fit,
                y_fit,
                "--",
                color=color,
                linewidth=2,
                label=fit_label,
            )
        else:
            ax.plot(xs, ys, "--", color=color, linewidth=2, label=fr"$R={R}$")

    ax.set_xlabel(r"Total experiment time $T_{\rm tot}$")
    ax.set_ylabel(r"Trace Fisher information $\mathrm{Tr}(I_C)$")
    ax.set_title(rf"{family}: Fisher trace vs $T_{{\rm tot}}$ at $\alpha={alpha}$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=12)
    fig.tight_layout()
    plt.show()


def plot_p_vs_R_across_families(family_results: Dict[str, Dict[str, Any]]) -> None:
    if not family_results:
        return

    families = sorted(family_results.keys())
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xscale("log")

    data_handles = []
    data_labels = []
    all_Rs = set()

    for family in families:
        data = family_results[family]
        alpha_key = sorted(data["alphas"].keys(), key=float)[0]
        alpha = float(alpha_key)
        results_for_alpha = data["alphas"][alpha_key]

        xs, ys, yerrs = [], [], []
        for R_str, entry in results_for_alpha["data"].items():
            p = entry.get("p", np.nan)
            delta_p = entry.get("delta_p", np.nan)

            if np.isfinite(p):
                xs.append(int(R_str))
                ys.append(p)
                yerrs.append(delta_p if np.isfinite(delta_p) else 0.0)
                all_Rs.add(int(R_str))

        if xs:
            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]
            yerrs = np.array(yerrs)[order]

            color = FAMILY_COLORS.get(family, "black")
            marker = FAMILY_MARKERS.get(family, "o")

            eb = ax.errorbar(
                xs,
                ys,
                yerr=yerrs,
                fmt=marker,
                linestyle="None",
                markersize=8,
                markeredgecolor="k",
                markerfacecolor=color,
                ecolor=color,
                elinewidth=1.5,
                capsize=4,
                alpha=0.9,
                label=family
            )

            ax.plot(xs, ys, "--", color=color, linewidth=2)

            data_handles.append(eb)
            data_labels.append(family)

    theory_line = ax.axhline(
        1.5,
        linestyle=":",
        linewidth=2,
        color="black",
        label=r"$p=3/2$"
    )

    data_legend = ax.legend(
        handles=data_handles,
        labels=data_labels,
        fontsize=13,
        loc="upper left",
        title="Data families"
    )
    ax.add_artist(data_legend)

    ax.legend(
        handles=[theory_line],
        fontsize=13,
        loc="lower right"
    )

    all_Rs = sorted(all_Rs)
    ax.set_xticks(all_Rs)
    ax.set_xticklabels([str(r) for r in all_Rs])

    ax.set_xlabel(r"Number of spread states $R$")
    ax.set_ylabel(r"Fitted exponent $p$")
    ax.set_title(rf"Fisher scaling exponent $p$ vs $R$ ($\alpha={alpha}$)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    plt.show()


def plot_p_r1_vs_alpha_across_families(
    family_results: Dict[str, Dict[str, Any]],
    theory_gamma0: float = 2.0,
) -> None:
    if not family_results:
        print("No fixed-R1-vary-alpha results found for p-vs-alpha plot.")
        return

    def p_theory(alpha, gamma0):
        return (alpha * gamma0 + 1.0) / (alpha + 1.0)

    families = sorted(family_results.keys())
    fig, ax = plt.subplots(figsize=(8, 7))
    all_alpha_vals = []

    data_handles = []
    data_labels = []

    for family in families:
        data = family_results[family]
        alphas = []
        p_vals = []
        p_errs = []

        for alpha_str, results_for_alpha in data["alphas"].items():
            alpha = float(alpha_str)
            entry = results_for_alpha["data"].get("1", None)
            if entry is None:
                continue

            p = entry.get("p", np.nan)
            delta_p = entry.get("delta_p", np.nan)

            if np.isfinite(p):
                alphas.append(alpha)
                p_vals.append(p)
                p_errs.append(delta_p if np.isfinite(delta_p) else 0.0)
                all_alpha_vals.append(alpha)

        if not alphas:
            continue

        alphas = np.array(alphas, dtype=float)
        p_vals = np.array(p_vals, dtype=float)
        p_errs = np.array(p_errs, dtype=float)

        order = np.argsort(alphas)
        alphas = alphas[order]
        p_vals = p_vals[order]
        p_errs = p_errs[order]

        color = FAMILY_COLORS.get(family, "black")
        marker = FAMILY_MARKERS.get(family, "o")

        eb = ax.errorbar(
            alphas,
            p_vals,
            yerr=p_errs,
            fmt=marker,
            linestyle="None",
            markersize=8,
            markeredgecolor="k",
            markerfacecolor=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            alpha=0.8,
            label=family,
        )

        ax.plot(
            alphas,
            p_vals,
            "--",
            color=color,
            linewidth=2.0,
        )

        data_handles.append(eb)
        data_labels.append(family)

    theory_main = None
    if all_alpha_vals:
        alpha_min = min(all_alpha_vals)
        alpha_max = max(all_alpha_vals)
        alpha_fine = np.linspace(alpha_min, alpha_max, 400)
        theory_vals = p_theory(alpha_fine, theory_gamma0)

        theory_main, = ax.plot(
            alpha_fine,
            theory_vals,
            "-",
            color="black",
            linewidth=2.5,
            alpha=0.9,
            label=rf"Theory: $p=(\alpha\gamma_0+1)/(\alpha+1)$, $\gamma_0={theory_gamma0:.0f}$",
        )

    sql_line = ax.axhline(
        1.0,
        color="black",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label=r"SQL: $p=1$",
    )

    data_legend = ax.legend(
        handles=data_handles,
        labels=data_labels,
        fontsize=13,
        loc="upper left",
        title="Data families",
    )
    ax.add_artist(data_legend)

    theory_handles = []
    if theory_main is not None:
        theory_handles.append(theory_main)
    theory_handles.extend([sql_line])

    ax.legend(
        handles=theory_handles,
        loc="lower right",
        fontsize=12,
        title=None,
    )

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"Fitted exponent $p$ at $R=1$")
    ax.set_title(r"Scaling exponent $p(R=1)$ vs $\alpha$ across families")
    ax.tick_params(labelsize=15)
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.7)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    plt.show()


def plot_trace_vs_time_for_varying_qubits(
    qubit_results: Dict[int, Dict[str, Any]],
    family: str = "XYZ",
) -> None:
    """
    Plot Tr(I) vs total experiment time for varying numbers of qubits.
    Assumes fixed R=1 and fixed alpha in each run.
    """
    if not qubit_results:
        print("No qubit-sweep Fisher results found.")
        return

    qubit_list = sorted(qubit_results.keys())
    #cmap = plt.cm.viridis(np.linspace(0, 1, len(qubit_list)))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(qubit_list)))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xscale("log")
    ax.set_yscale("log")

    alpha = None

    for i, n in enumerate(qubit_list):
        all_results = qubit_results[n]
        alpha_key = sorted(all_results["alphas"].keys(), key=float)[0]
        alpha = float(alpha_key)

        results_for_alpha = all_results["alphas"][alpha_key]
        entry = results_for_alpha["data"].get("1", None)

        if entry is None:
            continue

        xs = np.array(entry["T_tot"], dtype=float)
        ys = np.array(entry["trace_fisher"], dtype=float)
        p_fit = entry.get("p", np.nan)
        delta_p = entry.get("delta_p", np.nan)

        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]

        color = cmap[i]
        marker = markers[i % len(markers)]

        ax.scatter(
            xs,
            ys,
            s=80,
            color=color,
            marker=marker,
            edgecolor="black",
            zorder=3,
        )

        if np.isfinite(p_fit):
            a_fit = fit_prefactor(xs, ys, p_fit)
            x_fit = np.linspace(xs.min(), xs.max(), 300)
            y_fit = a_fit * (x_fit ** p_fit)

            if np.isfinite(delta_p):
                label = fr"$n={n}$, $p={p_fit:.3f}\pm{delta_p:.1e}$"
            else:
                label = fr"$n={n}$, $p={p_fit:.3f}$"

            ax.plot(
                x_fit,
                y_fit,
                "--",
                color=color,
                linewidth=2,
                label=label,
            )
        else:
            ax.plot(
                xs,
                ys,
                "--",
                color=color,
                linewidth=2,
                label=fr"$n={n}$",
            )

    ax.set_xlabel(r"Total experiment time $T_{\rm tot}$")
    ax.set_ylabel(r"Trace Fisher information $\mathrm{Tr}(I_C)$")
    ax.set_title(rf"Fisher trace vs $T_{{\rm tot}}$ for qubit number ($R=1$,$\alpha={alpha}$)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=12)
    fig.tight_layout()
    plt.savefig("Figure_FisherTraceScaling_vary_qubits.png", dpi=200)
    plt.show()
    
def extract_eta_vs_R(
    all_results: Dict[str, Any],
    alpha: float = 1.0,
    use_last_point: bool = True,
) -> Dict[int, float]:
    """
    Extract eta_diag as a function of R from one combined JSON result.

    If multiple eta values exist across n_steps, use either:
    - the last point (default), or
    - the mean over points.
    """
    alpha_key = str(alpha)
    if alpha_key not in all_results["alphas"]:
        alpha_key = sorted(all_results["alphas"].keys(), key=float)[0]

    results_for_alpha = all_results["alphas"][alpha_key]
    out = {}

    for R_str, entry in results_for_alpha["data"].items():
        eta_vals = entry.get("eta_diag", None)
        if eta_vals is None:
            continue

        eta_vals = np.asarray(eta_vals, dtype=float)
        eta_vals = eta_vals[np.isfinite(eta_vals)]
        if eta_vals.size == 0:
            continue

        if use_last_point:
            out[int(R_str)] = float(eta_vals[-1])
        else:
            out[int(R_str)] = float(np.mean(eta_vals))

    return out
    
def plot_eta_diag_vs_R_for_varying_qubits(
    qubit_results: Dict[int, Dict[str, Any]],
    family: str = "XYZ",
    alpha: float = 1.0,
) -> None:
    """
    Plot eta_diag vs R for different qubit numbers.
    Assumes fixed alpha and varying spread-state ensemble size R.
    """
    if not qubit_results:
        print("No qubit-sweep diagonalization results found.")
        return

    qubit_list = sorted(qubit_results.keys())
    cmap = plt.cm.tab10(np.linspace(0, 1, len(qubit_list)))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xscale("log")

    all_Rs = set()

    for i, n in enumerate(qubit_list):
        eta_by_R = extract_eta_vs_R(qubit_results[n], alpha=alpha, use_last_point=True)
        if not eta_by_R:
            continue

        Rs = np.array(sorted(eta_by_R.keys()), dtype=float)
        etas = np.array([eta_by_R[int(R)] for R in Rs], dtype=float)

        all_Rs.update(Rs.tolist())

        color = cmap[i]
        marker = markers[i % len(markers)]

        ax.scatter(
            Rs,
            etas,
            s=90,
            color=color,
            marker=marker,
            edgecolor="black",
            zorder=3,
        )
        ax.plot(
            Rs,
            etas,
            "--",
            color=color,
            linewidth=2,
            label=fr"$n={n}$",
        )

    all_Rs = sorted(set(int(r) for r in all_Rs))
    ax.set_xticks(all_Rs)
    ax.set_xticklabels([str(r) for r in all_Rs])

    ax.set_xlabel(r"Number of spread states $R$")
    ax.set_ylabel(r"Diagonalization measure $\eta_{\mathrm{diag}}$")
    ax.set_title(rf"Fisher-matrix diagonalization vs $R$ ($\alpha={alpha}$)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=12)
    fig.tight_layout()
    plt.savefig("Figure_FisherDiagonalization_vs_R.png", dpi=200)
    plt.show()

def load_qubit_sweep_results_for_family(data_dir: str, family: str) -> Dict[int, Dict[str, Any]]:
    """
    Load qubit-sweep Fisher diagnostic results for one family.

    Expects subfolders somewhere below data_dir named like:
        2_qubits, 3_qubits, ..., 8_qubits
    each containing:
        fisher_trace_scaling_<family>_all_alphas.json
    """
    out: Dict[int, Dict[str, Any]] = {}
    target_name = f"fisher_trace_scaling_{family}_all_alphas.json"

    for root, dirs, _ in os.walk(data_dir):
        for d in dirs:
            m = re.match(r"(\d+)_qubits$", d)
            if not m:
                continue

            n = int(m.group(1))
            subdir = os.path.join(root, d)
            candidate = os.path.join(subdir, target_name)

            if os.path.isfile(candidate):
                out[n] = load_json(candidate)

    return out

def plot_from_data_dir(data_dir: str, family: str) -> None:
    # -------- Experiment A --------
    expA_root = os.path.join(data_dir, "expA_fixed_alpha_vary_R")
    expA_paths = discover_combined_jsons(expA_root)
    fixed_alpha_vary_R, _ = classify_results(expA_paths)

    if family not in fixed_alpha_vary_R:
        raise ValueError(
            f"Family '{family}' not found in Experiment A results. "
            f"Available: {sorted(fixed_alpha_vary_R.keys())}"
        )

    plot_trace_vs_time_for_family(fixed_alpha_vary_R[family])
    plot_p_vs_R_across_families(fixed_alpha_vary_R)

    # -------- Experiment B --------
    expB_root = os.path.join(data_dir, "expB_fixed_R1_vary_alpha")
    expB_paths = discover_combined_jsons(expB_root)
    _, fixed_R1_vary_alpha = classify_results(expB_paths)

    plot_p_r1_vs_alpha_across_families(fixed_R1_vary_alpha)

    # -------- Experiment C --------
    expC_root = os.path.join(data_dir, "expC_vary_qubits")
    if os.path.isdir(expC_root):
        qubit_results = load_qubit_sweep_results_for_family(expC_root, family=family)
        if qubit_results:
            plot_trace_vs_time_for_varying_qubits(qubit_results, family=family)

    # -------- Experiment D --------
    expD_root = os.path.join(data_dir, "expD_diagonalization_vary_qubits_and_R")
    if os.path.isdir(expD_root):
        qubit_diag_results = load_qubit_sweep_results_for_family(expD_root, family=family)
        if qubit_diag_results:
            plot_eta_diag_vs_R_for_varying_qubits(
                qubit_diag_results,
                family=family,
                alpha=1.0,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--family", default="XYZ")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(base_dir, "original_data")

    plot_from_data_dir(
        data_dir=os.path.abspath(data_dir),
        family=args.family,
    )


if __name__ == "__main__":
    main()