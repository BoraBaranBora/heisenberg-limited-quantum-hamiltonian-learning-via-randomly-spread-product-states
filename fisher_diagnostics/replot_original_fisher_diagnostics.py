#!/usr/bin/env python3
"""
replot_original_fisher_diagnostics.py

Replots the Fisher-information diagnostic figures from the original
datasets included with the repository.

This script visualizes the Fisher-information diagnostic data used in the
analysis of the cumulative Fisher scaling behaviour. The data were
generated once and stored in the repository under the accompanying
``data/`` directory. The script loads these datasets and reproduces the
diagnostic plots used in the investigation.

Unlike ``plot_fisher_diagnostics.py``, which operates on data produced by
``reproduce_fisher_diagnostics_data.py``, this script operates directly on
the archived datasets distributed with the code base.

Generated figures
-----------------

The script produces three diagnostic plots:

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

Data discovery
--------------

The script searches the local ``data/`` directory for files of the form

    fisher_trace_scaling_*_all_alphas.json

and automatically determines whether they belong to:

• fixed-α / varying-R experiments  
• fixed-R=1 / varying-α experiments

based on the structure of the stored data.

Usage
-----

Run the script from the directory containing the archived datasets:

    python replot_original_fisher_diagnostics.py

Optionally specify the Hamiltonian family used for the first plot:

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

    # ---------- Fonts ----------
    "font.size": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,

    # ---------- Lines ----------
    "lines.linewidth": 2,
    "lines.markersize": 8,

    # ---------- Figure ----------
    #"figure.figsize": (8, 7),
    "figure.dpi": 100,

    # ---------- Tick style ----------
    "xtick.direction": "in",
    "ytick.direction": "in",

    # ---------- Legend ----------
    "legend.frameon": True,
    "legend.framealpha": 0.9,

    # ---------- Savefig ----------
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
    #plt.savefig("Figure_FisherTraceScaling_XYZ.png", dpi=200, bbox_inches="tight")
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

    # Theory reference line
    theory_line = ax.axhline(
        1.5,
        linestyle=":",
        linewidth=2,
        color="black",
        label=r"$p=3/2$"
    )

    # Data legend (upper left)
    data_legend = ax.legend(
        handles=data_handles,
        labels=data_labels,
        fontsize=13,
        loc="upper left",
        title="Data families"
    )
    ax.add_artist(data_legend)

    # Theory legend (lower right)
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
    #plt.savefig("Figure_FisherExponent_vs_R_XYZ.png", dpi=200, bbox_inches="tight")
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
    #heis_line = ax.axhline(
    #    2.0,
    #    color="gray",
    #    linestyle="--",
    #    linewidth=1.5,
    #    alpha=0.7,
    #    label=r"$p=2$",
    #)

    # Data legend in upper left
    data_legend = ax.legend(
        handles=data_handles,
        labels=data_labels,
        fontsize=13,
        loc="upper left",
        title="Data families",
    )
    ax.add_artist(data_legend)

    # Theory/reference legend in lower right
    theory_handles = []
    if theory_main is not None:
        theory_handles.append(theory_main)
    theory_handles.extend([sql_line])#, heis_line])

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
    #plt.savefig("Figure_FisherExponent_vs_Alpha_R1_XYZ.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_from_data_dir(data_dir: str, family: str) -> None:
    json_paths = discover_combined_jsons(data_dir)
    fixed_alpha_vary_R, fixed_R1_vary_alpha = classify_results(json_paths)

    if family not in fixed_alpha_vary_R:
        raise ValueError(
            f"Family '{family}' not found in fixed-alpha-vary-R results. "
            f"Available: {sorted(fixed_alpha_vary_R.keys())}"
        )

    plot_trace_vs_time_for_family(fixed_alpha_vary_R[family])
    plot_p_vs_R_across_families(fixed_alpha_vary_R)
    plot_p_r1_vs_alpha_across_families(fixed_R1_vary_alpha)


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