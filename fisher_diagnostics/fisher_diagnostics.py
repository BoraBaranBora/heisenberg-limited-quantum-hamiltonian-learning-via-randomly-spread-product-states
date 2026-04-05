#!/usr/bin/env python3
"""
fisher_diagnostics.py

Runs a Fisher-information diagnostic experiment for a single Hamiltonian
family and a specified set of experimental parameters.

This script evaluates the scaling behaviour of the classical Fisher
information obtained from measurement statistics of time-evolved quantum
states. The diagnostic directly probes how the Fisher-information trace

    Tr(I)

grows as a function of the total experiment time

    T_tot = Σ_k t_k

under a multi-time measurement protocol.

For each spread-state ensemble size R and time-schedule exponent α, the
script simulates measurement statistics, computes the classical Fisher
information using automatic differentiation, and fits the resulting
scaling trajectory to a power law

    Tr(I(T_tot)) = A T_tot^p.

The fitted exponent p characterizes the cumulative Fisher-information
scaling predicted by the theoretical analysis.

In addition to the Fisher trace, the script can also evaluate a scalar
diagonalization measure of the Fisher matrix,

    eta_diag = ||diag(I)||_F^2 / ||I||_F^2,

which quantifies how strongly the Fisher information is concentrated on
its diagonal.

This script executes **one diagnostic run** for a single Hamiltonian
family. Higher-level experiment orchestration (e.g. running multiple
families or experiment configurations) is handled by the companion
script:

    reproduce_fisher_diagnostics_data.py

Output
------

The diagnostic produces JSON files containing:

• total experiment times T_tot
• Fisher trace values Tr(I)
• diagonalization measure eta_diag
• fitted scaling exponents p
• experiment metadata

These outputs can be visualized using:

    plot_fisher_diagnostics.py

Usage
-----

Run a single diagnostic experiment from the command line:

    python fisher_diagnostics.py --family XYZ

Optional parameters allow customization of time schedules, spread-state
ensemble sizes, and measurement settings.
"""

#!/usr/bin/env python3
import os, sys
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
sys.path.insert(0, SRC_DIR)

import json
import argparse
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from datagen import DataGen
from hamiltonian_generator import generate_hamiltonian, generate_hamiltonian_parameters
from loss import Loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_times(alpha: float, N: int, delta_t: float) -> List[float]:
    return [delta_t * (k ** alpha) for k in range(1, N + 1)]


def flatten_lower_triangular_real(mat: torch.Tensor) -> torch.Tensor:
    idx = torch.tril_indices(mat.shape[0], mat.shape[1], device=mat.device)
    return mat.real[idx[0], idx[1]].to(torch.float32)


class ConstantPredictor(nn.Module):
    def __init__(self, flat_params: torch.Tensor):
        super().__init__()
        self.flat_params = nn.Parameter(flat_params.clone().detach())

    def forward(self, batch_size: int = 1) -> torch.Tensor:
        return self.flat_params.unsqueeze(0).expand(batch_size, -1)


def get_probs_for_batch(
    loss_obj: Loss,
    predictor: ConstantPredictor,
    times_t: torch.Tensor,
    init_states: torch.Tensor,
    basis_indices: torch.Tensor,
) -> torch.Tensor:
    predicted_state = loss_obj.get_state_prediction(
        predictor=predictor,
        initial_state=init_states,
        time=times_t,
    )
    predicted_state_rotated = loss_obj.apply_rotation(predicted_state, basis_indices)

    probs = torch.real(torch.diagonal(predicted_state_rotated, dim1=-2, dim2=-1))
    probs = torch.clamp(probs, min=1e-12)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs


def fisher_trace_from_probs(
    probs: torch.Tensor,
    param_vec: torch.Tensor,
    max_outcomes: int = None,
) -> float:
    """
    Compute cumulative Fisher trace

        Tr(I_C) = sum_{b,x} ||grad p_{b,x}||^2 / p_{b,x}

    summed over the batch.
    """
    probs = torch.clamp(probs, min=1e-12)

    batch_size, n_outcomes = probs.shape
    if max_outcomes is not None:
        n_outcomes = min(n_outcomes, max_outcomes)

    fisher_trace = torch.tensor(0.0, device=probs.device)

    for b in range(batch_size):
        for x in range(n_outcomes):
            p = probs[b, x]
            grad_p = torch.autograd.grad(
                p,
                param_vec,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            fisher_trace = fisher_trace + torch.dot(grad_p, grad_p) / p

    return float(fisher_trace.detach().cpu().item())

def fisher_matrix_from_probs(
    probs: torch.Tensor,
    param_vec: torch.Tensor,
    max_outcomes: int = None,
) -> torch.Tensor:
    """
    Compute cumulative classical Fisher matrix

        I_ij = sum_{b,x} (∂_i p_{b,x})(∂_j p_{b,x}) / p_{b,x}

    summed over the batch.
    """
    probs = torch.clamp(probs, min=1e-12)

    batch_size, n_outcomes = probs.shape
    n_params = param_vec.numel()

    if max_outcomes is not None:
        n_outcomes = min(n_outcomes, max_outcomes)

    fisher = torch.zeros((n_params, n_params), device=probs.device, dtype=torch.float32)

    for b in range(batch_size):
        for x in range(n_outcomes):
            p = probs[b, x]
            grad_p = torch.autograd.grad(
                p,
                param_vec,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            fisher = fisher + torch.outer(grad_p, grad_p) / p

    return fisher


def chunked_fisher_trace(
    loss_obj: Loss,
    predictor: ConstantPredictor,
    times_t: torch.Tensor,
    init_states: torch.Tensor,
    basis_indices: torch.Tensor,
    batch_size: int,
    max_outcomes: int = None,
) -> float:
    n = times_t.shape[0]
    total = 0.0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        probs = get_probs_for_batch(
            loss_obj=loss_obj,
            predictor=predictor,
            times_t=times_t[start:end],
            init_states=init_states[start:end],
            basis_indices=basis_indices[start:end],
        )

        total += fisher_trace_from_probs(
            probs=probs,
            param_vec=predictor.flat_params,
            max_outcomes=max_outcomes,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return float(total)

def eta_diag_from_fisher(fisher: torch.Tensor) -> float:
    """
    Compute diagonalization measure

        eta_diag = sum_i I_ii^2 / sum_{i,j} I_ij^2
                 = ||diag(I)||_F^2 / ||I||_F^2
    """
    diag_sq = torch.sum(torch.diagonal(fisher) ** 2)
    total_sq = torch.sum(fisher ** 2)

    if total_sq <= 0:
        return float("nan")

    return float((diag_sq / total_sq).detach().cpu().item())

def chunked_fisher_trace_and_eta(
    loss_obj: Loss,
    predictor: ConstantPredictor,
    times_t: torch.Tensor,
    init_states: torch.Tensor,
    basis_indices: torch.Tensor,
    batch_size: int,
    max_outcomes: int = None,
) -> Dict[str, float]:
    n = times_t.shape[0]
    total_trace = 0.0
    total_fisher = None

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        probs = get_probs_for_batch(
            loss_obj=loss_obj,
            predictor=predictor,
            times_t=times_t[start:end],
            init_states=init_states[start:end],
            basis_indices=basis_indices[start:end],
        )

        fisher_chunk = fisher_matrix_from_probs(
            probs=probs,
            param_vec=predictor.flat_params,
            max_outcomes=max_outcomes,
        )

        trace_chunk = float(torch.trace(fisher_chunk).detach().cpu().item())
        total_trace += trace_chunk

        if total_fisher is None:
            total_fisher = fisher_chunk
        else:
            total_fisher = total_fisher + fisher_chunk

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eta_diag = eta_diag_from_fisher(total_fisher)

    return {
        "trace_fisher": float(total_trace),
        "eta_diag": eta_diag,
    }


def fit_power_law(xs: np.ndarray, ys: np.ndarray) -> Dict[str, float]:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if np.any(xs <= 0) or np.any(ys <= 0):
        raise ValueError("Power-law fit requires positive xs and ys.")

    lx = np.log(xs)
    ly = np.log(ys)

    coeffs, cov = np.polyfit(lx, ly, 1, cov=True)
    p = float(coeffs[0])
    logA = float(coeffs[1])

    delta_p = float(np.sqrt(cov[0, 0]))
    delta_logA = float(np.sqrt(cov[1, 1]))
    
    A = float(np.exp(logA))

    y_pred = p * lx + logA
    ss_res = float(np.sum((ly - y_pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "p": p,
        "delta_p": delta_p,
        "A": A,
        "logA": logA,
        "delta_logA": delta_logA,
        "r2": r2,
    }

def safe_alpha_tag(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def make_output_dir(script_path: str, family: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(script_path))
    root = os.path.join(script_dir, "fisher_trace_diagnostics_outputs")
    os.makedirs(root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(root, f"{family}_{timestamp}")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def to_serializable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(to_serializable(obj), f, indent=2)

def run_fisher_diagnostic(
    family: str,
    num_qubits: int,
    alphas: List[float],
    delta_t: float,
    steps: List[int],
    spreadings: List[int],
    measurements: int,
    shots: int,
    seed: int,
    max_batch: int,
    max_outcomes: int = None,
    output_root: str = None,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n==================== current family = {family} ====================")

    if output_root is None:
        outdir = make_output_dir(__file__, family)
    else:
        outdir = output_root
        os.makedirs(outdir, exist_ok=True)

    print(f"Saving outputs to: {outdir}")

    ham_params = generate_hamiltonian_parameters(
        family=family,
        num_qubits=num_qubits,
        coupling_type="anisotropic_normal",
        h_field_type="random",
    )

    H_true = generate_hamiltonian(
        family=family,
        num_qubits=num_qubits,
        device=device,
        **ham_params,
    ).to(torch.complex64)

    flat_true = flatten_lower_triangular_real(H_true).to(device)
    predictor = ConstantPredictor(flat_true).to(device)
    loss_obj = Loss(num_qubits=num_qubits).to(device)

    all_results: Dict[str, Any] = {
        "family": family,
        "num_qubits": num_qubits,
        "delta_t": delta_t,
        "measurements": measurements,
        "shots": shots,
        "steps": steps,
        "spreadings": spreadings,
        "hamiltonian_params": ham_params,
        "alphas": {},
    }

    for alpha in alphas:
        print(f"\n==================== alpha = {alpha} ====================")
        results_for_alpha: Dict[str, Any] = {
            "alpha": alpha,
            "data": {},
        }

        for spreading in spreadings:
            T_vals = []
            F_vals = []
            eta_vals = []

            print(f"\n=== spreading = {spreading} ===")
            for n_steps in steps:
                times = generate_times(alpha, n_steps, delta_t)
                T_tot = float(sum(times))

                dg = DataGen(
                    num_qubits=num_qubits,
                    times=times,
                    num_measurements=measurements,
                    shots=shots,
                    spreadings=spreading,
                    initial_state_indices=[0],
                    seed=seed,
                    hamiltonian=H_true,
                )

                targets, times_t, basis, init = dg.generate_dataset()

                times_t = times_t.to(device)
                basis = basis.to(device)
                init = init.to(device)
                
                fisher_stats = chunked_fisher_trace_and_eta(
                    loss_obj=loss_obj,
                    predictor=predictor,
                    times_t=times_t,
                    init_states=init,
                    basis_indices=basis,
                    batch_size=max_batch,
                    max_outcomes=max_outcomes,
                )

                ftrace = fisher_stats["trace_fisher"]
                eta_diag = fisher_stats["eta_diag"]

                print(
                    f"steps={n_steps:>3d} | T_tot={T_tot:.6f} | "
                    f"Tr(I)={ftrace:.6e} | eta_diag={eta_diag:.6f}"
                )

                T_vals.append(T_tot)
                F_vals.append(ftrace)
                eta_vals.append(eta_diag)

#                ftrace = chunked_fisher_trace(
#                    loss_obj=loss_obj,
#                    predictor=predictor,
#                    times_t=times_t,
#                    init_states=init,
#                    basis_indices=basis,
#                    batch_size=max_batch,
#                    max_outcomes=max_outcomes,
#                )

                print(f"steps={n_steps:>3d} | T_tot={T_tot:.6f} | Val={ftrace:.6e}")

                T_arr = np.array(T_vals, dtype=float)
                F_arr = np.array(F_vals, dtype=float)

            if len(T_arr) > 1:
                fit = fit_power_law(T_arr, F_arr)
                print(f"fitted exponent p ≈ {fit['p']:.4f}")

                p_val = fit["p"]
                delta_p_val = fit["delta_p"]
                A_val = fit.get("A")
                r2_val = fit.get("r2")
            else:
                print("Skipping power-law fit: need more than one time point.")
                p_val = None
                delta_p_val = None
                A_val = None
                r2_val = None
            
            results_for_alpha["data"][str(spreading)] = {
                "n_steps": steps,
                "T_tot": T_vals,
                "trace_fisher": F_vals,
                "eta_diag": eta_vals,
                "p": p_val,
                "delta_p": delta_p_val,
                "A": A_val,
                "r2": r2_val,
            }

        all_results["alphas"][str(alpha)] = results_for_alpha

        per_alpha_json = os.path.join(
            outdir,
            f"fisher_trace_scaling_{family}_alpha{safe_alpha_tag(alpha)}.json"
        )
        save_json(results_for_alpha, per_alpha_json)

    combined_json = os.path.join(outdir, f"fisher_trace_scaling_{family}_all_alphas.json")
    save_json(all_results, combined_json)
    print(f"\nSaved combined results to {combined_json}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=str, required=True)
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument("--alphas", type=str, default="1.0")
    parser.add_argument("--delta-t", type=float, default=0.01)
    parser.add_argument("--steps", type=str, default="2,4,6,8,10")
    parser.add_argument("--spreadings", type=str, default="1,2,4,8,16")
    parser.add_argument("--measurements", type=int, default=25)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--seed", type=int, default=99901)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--max-outcomes", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)

    args = parser.parse_args()

    alpha_list = [float(x) for x in args.alphas.split(",") if x.strip()]
    step_list = [int(x) for x in args.steps.split(",") if x.strip()]
    spreading_list = [int(x) for x in args.spreadings.split(",") if x.strip()]

    run_fisher_diagnostic(
        family=args.family,
        num_qubits=args.num_qubits,
        alphas=alpha_list,
        delta_t=args.delta_t,
        steps=step_list,
        spreadings=spreading_list,
        measurements=args.measurements,
        shots=args.shots,
        seed=args.seed,
        max_batch=args.max_batch,
        max_outcomes=args.max_outcomes,
        output_root=args.output_root,
    )

if __name__ == "__main__":
    main()
