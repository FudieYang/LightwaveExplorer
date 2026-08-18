import os
import sys
import uuid
import shutil
import json
import datetime
import numpy as np
import cma
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
PROJECT_ROOT  = "/home/fudie/LightwaveExplorer"
CLI_PATH      = os.path.join(PROJECT_ROOT, "build_cli", "LightwaveExplorer")
TMP_DIR       = "/mnt/d/lwe_tmp"

NUM_ISLANDS = 2            
TOTAL_EPOCHS = 20          
GENERATIONS_PER_EPOCH = 4  
CMA_POP_SIZE = 12          
MAX_LBFGS_ITER = 3         # Max local gradient steps for inner linear params

NUM_CELLS = 5              

SIGMA_EXPLOIT = 0.01        
SIGMA_EXPLORE = 0.05        
SIGMA_MIGRATE = 0.1        
SIGMA_DECAY   = 0.9        

# --- Constraint Injection ---
# Narrow the search space for specific parameters instead of completely freezing them.
# Format: "key": (min, max). e.g., {"g0_theta": (40.0, 42.0)}
CUSTOM_BOUNDS = {}

sys.path.append(os.path.join(PROJECT_ROOT, "Source", "Python", "src"))
import LightwaveExplorer as lwe

# =============================================================================
# ATOMIC COMPONENTS LIBRARY (Super-Structure Definitions)
# =============================================================================
ATOMIC_LIBRARY = {
    "BiaxialCrystalBlock": {
        "opt_group": "CMA", # Highly non-convex phase matching
        "params": {
            "theta": {"min": 0.0, "max": 180.0},
            # "phi": {"min": 0.0, "max": 180.0},
            "length_um": {"min": 0.0, "max": 1000.0}
        },
        "template": "rotateIntoBiaxial(d,{theta:.4f},0.0000,d)nonlinear(d,{theta:.4f},0.0000,{length_um:.4f},d)rotateFromBiaxial(d,{theta:.4f},0.0000,d)"
    },
    "NormalCrystalBlock": {
        "opt_group": "GRAD", # Bulk dispersion / SPM (smooth)
        "params": {
            "length_um": {"min": 0.0, "max": 1000.0}
        },
        "template": "nonlinear(20,0.0000,0.0000,{length_um:.4f},d)"
    },
    "LinearPropagation": {
        "opt_group": "GRAD", # Linear dispersion
        "params": {
            "length_um": {"min": 0.0, "max": 5000.0}
        },
        "template": "linear(0,0,0,{length_um:.4f},d)"
    },
    "SphericalMirror": {
        "opt_group": "GRAD", # Smooth focal adjustment
        "params": {
            "focal_m": {"min": -1.0, "max": 1.0}
        },
        "template": "sphericalMirror({focal_m:.4f})"
    },
    "RotateFrame": {
        "opt_group": "CMA", # Highly periodic (cos^2), gradient gets stuck in local minima
        "params": {
            "angle": {"min": -180.0, "max": 180.0}
        },
        "template": "rotate({angle:.4f})"
    },
    "SpectralFilter": {
        "opt_group": "GRAD", # Linear filtering
        "params": {
            "f_center_THz": {"min": 50.0, "max": 1500.0},
            "f_width_THz": {"min": 5.0, "max": 500.0},
            "inBandAmplitude": {"min": -1.0, "max": 1.0},
            "outOfBandAmplitude": {"min": 0.0, "max": 1.0}
        },
        "template": "filter({f_center_THz:.2f},{f_width_THz:.2f},4,{inBandAmplitude:.4f},{outOfBandAmplitude:.4f})"
    },
    "Aperture": {
        "opt_group": "GRAD", # Spatial filtering
        "params": {
            "d_m": {"min": 0.0, "max": 0.005},
            "activation_parameter": {"min": 1.0, "max": 100.0}
        },
        "template": "aperture({d_m:.4f},{activation_parameter:.2f})"
    }
}

class AtomicGene:
    def __init__(self, comp_type):
        self.comp_type = comp_type
        self.config = ATOMIC_LIBRARY[comp_type]

class TopologyChromosome:
    def __init__(self, genes=None):
        self.genes = genes if genes else []

def build_super_structure(n_cells=8):
    genes = []
    # Engine blocks: "Non-Intuitive Discovery" Unit
    for _ in range(n_cells):
        genes.append(AtomicGene("BiaxialCrystalBlock")) # Nonlinear generation (Angles, length)
        genes.append(AtomicGene("NormalCrystalBlock"))  # Glass dispersion (Temporal walk-off compensation)
        genes.append(AtomicGene("LinearPropagation"))   # Air dispersion (Phase-mismatch compensation)
        genes.append(AtomicGene("RotateFrame"))         # Polarization mixing
    return TopologyChromosome(genes)

SUPER_TOPO = build_super_structure(NUM_CELLS)

# Decouple specifications into CMA (outer) and GRAD (inner)
CMA_SPECS = []
GRAD_SPECS = []

for i, gene in enumerate(SUPER_TOPO.genes):
    for p_name, p_def in gene.config["params"].items():
        key = f"g{i}_{p_name}"
        lo, hi = p_def["min"], p_def["max"]
        
        # Override bounds if user specified narrow constraints
        if key in CUSTOM_BOUNDS:
            lo, hi = CUSTOM_BOUNDS[key]
        
        # Smart initialization
        if p_name == "length_um":
            x0_val = 200.0 if gene.comp_type == "BiaxialCrystalBlock" else 100.0
        elif p_name in ["focal_m", "angle"]:
            x0_val = 0.0
        elif p_name == "d_m":
            x0_val = 0.005
        else:
            x0_val = (lo + hi) / 2.0
            
        spec = (key, lo, hi, x0_val)
        if gene.config["opt_group"] == "CMA":
            CMA_SPECS.append(spec)
        else:
            GRAD_SPECS.append(spec)

# ==========================================
# 1. Physics Execution Layer
# ==========================================
def run_lwe_super_structure(cma_params, grad_params, cli_path, work_dir):
    param_dict = {}
    for j, (key, lo, hi, _) in enumerate(CMA_SPECS):
        param_dict[key] = float(np.clip(cma_params[j], lo, hi))
    for j, (key, lo, hi, _) in enumerate(GRAD_SPECS):
        param_dict[key] = float(np.clip(grad_params[j], lo, hi))

    # Enforce passive filter
    for i, gene in enumerate(SUPER_TOPO.genes):
        if gene.comp_type == "SpectralFilter":
            in_key = f"g{i}_inBandAmplitude"
            out_key = f"g{i}_outOfBandAmplitude"
            if in_key in param_dict and out_key in param_dict:
                param_dict[out_key] = 1.0 - param_dict[in_key]

    STANDARD_LENGTHS = np.array([50, 100, 150, 200, 250, 300, 400, 500, 1000])
    
    # Render string
    seq_parts = ["init()"]
    for i, gene in enumerate(SUPER_TOPO.genes):
        gene_params = {}
        for p_name in gene.config["params"].keys():
            val = param_dict[f"g{i}_{p_name}"]
            
            # Snap length for crystals to nearest standard length
            if p_name == "length_um" and gene.comp_type == "BiaxialCrystalBlock":
                idx = np.argmin(np.abs(STANDARD_LENGTHS - val))
                val = float(STANDARD_LENGTHS[idx])
                
            gene_params[p_name] = val
            
        part = gene.config["template"].format(**gene_params)
        seq_parts.append(part)
        
    sequence_str = "".join(seq_parts)

    os.makedirs(work_dir, exist_ok=True)
    runner = lwe.SimulationRunner(cli_path=cli_path, work_dir=work_dir)
    
    try:
        runner.set_params(
            sequence=sequence_str,
            # FIXED LASER PARAMETERS (No longer optimized)
            pulse_energy1=1.9e-08, frequency1=1.3e14, bandwidth1=1.5e13,
            sg_order1=2, beamwaist1=1.12e-5, delay1=-1.3e-13,
            polarization1=1.5707963267949,
            pulse_energy2=0, frequency2=4.9e14, bandwidth2=2e13,
            sg_order2=4, gdd2=1e-28, beamwaist2=9e-5,
            polarization2=1.5707963267949,
            material_index=4, crystal_thickness=0.0004, dz=1e-6,
            grid_width=2.08e-4, grid_height=2.08e-4, dx=2e-5,
            time_span=6e-13, dt=5e-16, band_gap=6, effective_mass=1, drude_gamma=5e12,
            propagation_mode=2
        )
        runner.run(verbose=False, timeout=800)

        freq = runner.result.frequencyVectorSpectrum
        spectrum = runner.result.spectrumTotal
        if spectrum.ndim > 1:
            spectrum = spectrum[-1]

        thg_center = 390e12
        mask_thg = np.abs(freq - thg_center) < 30e12
        thg_pwr = np.trapezoid(spectrum[mask_thg], freq[mask_thg])

        INPUT_ENERGY = 1.9e-08
        eff = float(thg_pwr / INPUT_ENERGY if INPUT_ENERGY > 0 else 0.0)
        
        if np.isnan(eff) or np.isinf(eff) or eff > 1.0 or eff < 0:
            eff = 0.0
            
        # Log to JSONL database
        try:
            topo_str = "___".join([gene.comp_type for gene in SUPER_TOPO.genes])
            log_data = {
                "ts": datetime.datetime.now().isoformat(),
                "topo": topo_str,
                "eff": eff,
                "seq": sequence_str,
                "n": len(param_dict),
                "params": param_dict
            }
            with open(EVAL_LOG, "a") as f:
                f.write(json.dumps(log_data) + "\n")
        except Exception:
            pass

        return eff, sequence_str
    except Exception:
        return 0.0, sequence_str
    finally:
        runner.cleanup()

# ==========================================
# 2. Inner Layer: Surrogate Gradient Optimization
# ==========================================
def gradient_assisted_eval(cma_params, init_grad_params, cli_path, work_dir):
    best_seq = ""
    best_eff = -np.inf
    
    def inner_obj(p):
        nonlocal best_seq, best_eff
        eff, seq = run_lwe_super_structure(cma_params, p, cli_path, work_dir)
        if eff > best_eff:
            best_eff = eff
            best_seq = seq
        return eff
    
    bounds_lo = np.array([s[1] for s in GRAD_SPECS])
    bounds_hi = np.array([s[2] for s in GRAD_SPECS])
    p = np.copy(init_grad_params)
    
    # 1. Base evaluation
    inner_obj(p)
    
    # 2. SPSA Gradient Optimization
    c = 2.0  # Perturbation step (must be > dz=1um to avoid grid snapping)
    for _ in range(MAX_LBFGS_ITER):
        delta = np.random.choice([-1.0, 1.0], size=len(p))
        
        p_plus = np.clip(p + c * delta, bounds_lo, bounds_hi)
        y_plus = inner_obj(p_plus)
        
        p_minus = np.clip(p - c * delta, bounds_lo, bounds_hi)
        y_minus = inner_obj(p_minus)
        
        actual_delta = (p_plus - p_minus) / 2.0
        safe_delta = np.where(np.abs(actual_delta) < 1e-6, 1.0, actual_delta)
        
        grad = (y_plus - y_minus) / safe_delta
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            # Gradient Ascent step: 20 um fixed step size in the direction of steepest ascent
            p = np.clip(p + (grad / grad_norm) * 20.0, bounds_lo, bounds_hi)
            inner_obj(p)
            
    return best_eff, p, best_seq

# ==========================================
# 3. Outer Layer: One CMA-ES Island (With Top-K Logic)
# ==========================================
def evaluate_generation(island_id, solutions, current_grad, cli_path):
    island_uuid = str(uuid.uuid4())[:6]
    work_dir = os.path.join(TMP_DIR, f"island_{island_id}_{island_uuid}")
    os.makedirs(work_dir, exist_ok=True)
    
    cheap_effs = []
    cheap_seqs = []
    for sol in solutions:
        eff, seq = run_lwe_super_structure(sol, current_grad, cli_path, work_dir)
        cheap_effs.append(eff)
        cheap_seqs.append(seq)
        
    top2_indices = np.argsort(cheap_effs)[-2:][::-1] 
    final_effs = list(cheap_effs)
    
    best_opt_eff = 0.0
    best_opt_grad = np.copy(current_grad)
    best_opt_seq = ""
    best_opt_cma = solutions[0]
    
    for idx in top2_indices:
        sol = solutions[idx]
        opt_eff, opt_grad, opt_seq = gradient_assisted_eval(sol, current_grad, cli_path, work_dir)
        final_effs[idx] = opt_eff
        cheap_seqs[idx] = opt_seq
        
        if opt_eff > best_opt_eff:
            best_opt_eff = opt_eff
            best_opt_grad = opt_grad
            best_opt_seq = opt_seq
            best_opt_cma = sol
            
    # Also check cheap evals just in case
    gen_best_idx = int(np.argmax(final_effs))
    if final_effs[gen_best_idx] > best_opt_eff:
        best_opt_eff = final_effs[gen_best_idx]
        best_opt_cma = solutions[gen_best_idx]
        best_opt_seq = cheap_seqs[gen_best_idx]
            
    shutil.rmtree(work_dir, ignore_errors=True)
    return island_id, final_effs, cheap_seqs, best_opt_eff, best_opt_grad, best_opt_seq, best_opt_cma

# ==========================================
# 4. Master Controller
# ==========================================
def main():
    if not os.path.exists(CLI_PATH):
        print(f"ERROR: LWE CLI not found at {CLI_PATH}")
        return
        
    os.makedirs(TMP_DIR, exist_ok=True)
        
    base_cma_params = np.array([s[3] for s in CMA_SPECS])
    base_grad_params = np.array([s[3] for s in GRAD_SPECS])
    
    # === ELITE INJECTION ===
    baseline_cma = np.zeros_like(base_cma_params)
    baseline_grad = np.zeros_like(base_grad_params)
    
    for k, spec in enumerate(CMA_SPECS):
        key = spec[0]
        if key == "g0_length_um": baseline_cma[k] = 400.0
        elif key == "g0_theta": baseline_cma[k] = 0.0
        elif key == "g0_phi": baseline_cma[k] = 0.0 
        elif key == "g4_theta": baseline_cma[k] = 40.2
        elif key == "g4_length_um": baseline_cma[k] = 250.0
        elif key == "g7_angle": baseline_cma[k] = 180.0
        elif key == "g8_theta": baseline_cma[k] = 139.8
        elif key == "g8_length_um": baseline_cma[k] = 250.0
        
    base_cma_params = baseline_cma
    base_grad_params = baseline_grad
    
    lower = [s[1] for s in CMA_SPECS]
    upper = [s[2] for s in CMA_SPECS]
    stds = [(u - l) for u, l in zip(upper, lower)]
    es_opts = {'popsize': CMA_POP_SIZE, 'verbose': -9, 'bounds': [lower, upper], 'CMA_stds': stds}

    es_list = []
    grad_list = []
    best_island_effs = [0.0] * NUM_ISLANDS
    best_island_seqs = [""] * NUM_ISLANDS
    
    for i in range(NUM_ISLANDS):
        s0 = SIGMA_EXPLOIT if i == 0 else SIGMA_EXPLORE
        noise = np.random.randn(len(base_cma_params)) * np.array([(s[2]-s[1]) for s in CMA_SPECS]) * s0 * 0.1
        init_pos = np.clip(base_cma_params + noise, [s[1] for s in CMA_SPECS], [s[2] for s in CMA_SPECS])
        es = cma.CMAEvolutionStrategy(init_pos, s0, es_opts)
        es_list.append(es)
        grad_list.append(np.copy(base_grad_params))

    print(f"\n{'='*60}")
    print(f"Optimizing: {len(CMA_SPECS)}D (CMA-ES) + {len(GRAD_SPECS)}D (SPSA) = {len(CMA_SPECS)+len(GRAD_SPECS)}D Total")
    print(f"Islands: {NUM_ISLANDS} | Epochs: {TOTAL_EPOCHS} | Gens/Epoch: {GENERATIONS_PER_EPOCH} | Pop: {CMA_POP_SIZE}")
    print(f"{'='*60}")

    global_best_eff = 0.0
    global_best_seq = ""

    with ProcessPoolExecutor(max_workers=NUM_ISLANDS) as executor:
        for epoch in range(TOTAL_EPOCHS):
            epoch_best_effs = [0.0] * NUM_ISLANDS
            print(f"\n--- Epoch {epoch+1}/{TOTAL_EPOCHS} ---")
            sigma_str = "".join([f"[Island {i}] sigma={getattr(es_list[i], 'sigma', 0.0):.4f}  " for i in range(NUM_ISLANDS)])
            print(sigma_str)
            for gen in range(GENERATIONS_PER_EPOCH):
                futures = []
                solutions_list = []
                for i in range(NUM_ISLANDS):
                    if not es_list[i].stop():
                        solutions = es_list[i].ask()
                        solutions_list.append(solutions)
                        f = executor.submit(evaluate_generation, i, solutions, grad_list[i], CLI_PATH)
                        futures.append(f)
                    else:
                        solutions_list.append(None)
                        futures.append(None)
                
                for i, f in enumerate(futures):
                    if f is not None:
                        island_id, final_effs, cheap_seqs, opt_eff, opt_grad, opt_seq, opt_cma = f.result()
                        cma_fitnesses = [-eff for eff in final_effs]
                        es_list[island_id].tell(solutions_list[island_id], cma_fitnesses)
                        
                        if opt_eff > best_island_effs[island_id]:
                            best_island_effs[island_id] = opt_eff
                            best_island_seqs[island_id] = opt_seq
                            grad_list[island_id] = np.copy(opt_grad)
                            
                        if opt_eff > epoch_best_effs[island_id]:
                            epoch_best_effs[island_id] = opt_eff
                            
                        if opt_eff > global_best_eff:
                            global_best_eff = opt_eff
                            global_best_seq = opt_seq
                            
                        if gen % 2 == 0:
                            gen_best_idx = int(np.argmax(final_effs))
                            print(f"  [Island {island_id}] Gen {gen}: Best = {final_effs[gen_best_idx]*100:.4f}%")
                            print(f"    Seq: {cheap_seqs[gen_best_idx]}")
                            
            # --- Epoch Migration Logic ---
            # Compare based on THIS Epoch's performance, not historical glory!
            best_idx = np.argmax(epoch_best_effs)
            worst_idx = np.argmin(epoch_best_effs)
            
            print(f"--> Epoch {epoch+1} Completed. Global Best Efficiency: {global_best_eff*100:.4f}%")
            print(f"--> Best Sequence: {global_best_seq}")
            
            if worst_idx != best_idx and NUM_ISLANDS > 1:
                print(f"--> Migrating Elite from Island {best_idx} to restart Island {worst_idx} with massive noise!")
                # Reset worst island by creating a new ES object at the best island's mean
                try:
                    best_mean = es_list[best_idx].result.xbest
                except:
                    best_mean = es_list[best_idx].mean
                es_list[worst_idx] = cma.CMAEvolutionStrategy(best_mean, SIGMA_MIGRATE, es_opts)
                grad_list[worst_idx] = np.copy(grad_list[best_idx])
                best_island_effs[worst_idx] = 0.0 # reset to allow new tracking

if __name__ == "__main__":
    main()
