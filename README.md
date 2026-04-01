# Heisenberg-Limited Quantum Hamiltonian Learning \\ via Randomly Spread Product-States

---

This repository provides the code and data for the Hamiltonian learning strategy introduced in our publication, including both Hamiltonian recovery experiments and Fisher-information diagnostics validating the theoretical scaling predictions.

1. **Original (“publication”) data** (`replot_original_data/`)  
   Scripts under `replot_original_data/` that contain the _original_ data folders and produce all plots exactly as in the publication:  
   - **Figure 1:** Error vs ∑time for α = 1.0 (four Hamiltonian families).  
   - **Figure 2:** Error-scaling exponent β vs scheduling exponent α (with spreadings = 32).  
   - **Figure 3(a, b):** (a) Error vs ∑time for α = 1.0 with increasing number of state spreadings; (b) Fitted β vs number of spreadings.

   You can re‐plot Figures 1–3 (and the derivative comparison) without re‐training from the datasets in the same directory by using the `composite_replotting.py` script:  
   - `first_parameter_sweep_data/` → Figures 1 and 2 data (spreadings = 32; varying α; measurements = 25; steps = 1…8)  
   - `second_parameter_sweep_data/` → Figure 3 data (α = 1.0; varying number of state spreadings; measurements = 25; steps = 1…8)  

   - `composite_replotting.py` → orchestrates all re‐plotting pipelines for Figures 1–3 and the derivative comparison.

2. **Data‐(re-)generation pipelines** (`reproduce_original_data/`)  
   Scripts to re‐generate the embedding files (`.pth` weights + loss logs) for all figure data via `learn_hamiltonian.py`.  
   - The main controller is `rerun_selected_sweeps.py`, where you toggle flags to choose which sweeps (SWEEP 1 or 2) to run.  
   - Each sweep produces a set of folders under:
     ```
     first_parameter_sweep_data/     ← SWEEP 1 (α = 1.0, varying number of state spreadings; measurements = 25; steps = 1…8)
     second_parameter_sweep_data/    ← SWEEP 2 (spreadings = 32; varying α; measurements = 25; steps = 1…8)
     ```
   - Each `run_…` subfolder then contains directories of the form  
     `alpha_<…>_spreadings_<…>_measurements_<…>_shots_<…>_steps_<…>/`  
     with `config.json`, `hamiltonians.json`, `embedding_<codename>.pth`, etc.

   You can also re-plot Figures 1–2 (and the derivative comparison) from the reproduced data using the `composite_replotting.py` script.

3. **Source code** (`src/`)  
   - `learn_hamiltonian.py` – main training script  
   - `predictor.py`, `loss.py`, `hamiltonian_generator.py`, `datagen.py`, `utils.py` (helper modules)  
   - `extraction_and_evaluation.py` – loads embeddings, reconstructs density matrices, computes recovery errors, fits β across Fisher regimes  
   - `plotting_utils.py`, `plotting_pipelines.py` – collect & plot errors, compute β, draw Fisher‐regime figures  
   - `reproduction_pipelines.py` – functions to re‐generate all sweeps  


4. **Fisher‐information diagnostics** (`fisher_diagnostics/`)  
   This directory contains the scripts used to compute and visualize the Fisher-information diagnostics discussed in the paper. These diagnostics analyse the Fisher information of the measurement statistics directly, independently of the Hamiltonian reconstruction procedure.

   The purpose of these diagnostics is to verify that the scaling behaviour observed in the recovery experiments originates from the intrinsic information content of the measurement statistics rather than from numerical properties of the reconstruction algorithm.

   In particular, the Fisher diagnostics:

   - compute the classical Fisher information from simulated measurement probabilities,
   - analyse the scaling of the Fisher trace  
     \(\mathrm{Tr}\,\mathcal I(T_{\rm tot}) \propto T_{\rm tot}^{p}\),
   - verify the predicted cumulative Fisher scaling \(p=(\alpha\gamma_0+1)/(\alpha+1)\),
   - demonstrate that the quadratic short-time Fisher-information regime is already present with a single spread state when averaging over sufficiently many measurement bases.

   The scripts in this directory generate the Fisher-diagnostic figures reported in the paper, including:

   - Fisher trace vs total experiment time \(T_{\rm tot}\) for different spread-state ensemble sizes \(R\),
   - fitted Fisher scaling exponent \(p\) vs \(R\),
   - fitted Fisher scaling exponent \(p\) vs scheduling parameter \(\alpha\).

   These diagnostics complement the Hamiltonian recovery experiments by validating the theoretical Fisher-information scaling directly at the level of the measurement statistics.
   
---

## Requirements

- **Python 3.10+**  
- PyTorch (≥ 1.13 recommended)  
- NumPy, SciPy, scikit‐learn, tqdm, Matplotlib  

Install all dependencies via:

```bash
pip install -r requirements.txt
