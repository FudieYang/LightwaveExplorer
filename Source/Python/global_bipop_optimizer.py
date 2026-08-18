import os
import sys
import shutil

PROJECT_ROOT = "/home/fudie/LightwaveExplorer"
sys.path.append(os.path.join(PROJECT_ROOT, "Source", "Python", "src"))

import numpy as np
import cma
import uuid
import LightwaveExplorer as lwe
from concurrent.futures import ProcessPoolExecutor, as_completed

# =============================================================================
# 1. TUNABLE PARAMETERS & MACROS (Place all tunable settings here)
# =============================================================================
CLI_PATH = os.path.join(PROJECT_ROOT, "build_cli", "LightwaveExplorer")
TMP_DIR = "/mnt/d/lwe_tmp/tmp"

MAX_CELLS = 8
STANDARD_LENGTHS = np.array([50, 100, 150, 200, 250, 300, 400, 500, 1000])

# --- SPSA (Inner Gradient Search) Parameters ---
SPSA_STEP = 2.0
SPSA_LR = 20.0
MAX_LBFGS_ITER = 3

# --- BIPOP-CMA-ES (Outer Global Search) Parameters ---
BUDGET = 50000
# Initial sizes will be dynamically clamped based on minimum theoretical popsize (4 + 3*log(N))
POPSIZE_LARGE = 40
POPSIZE_SMALL = 10
SIGMA0 = 0.2 # 全局重启时的基础搜索半径（相对空间大小的比例，0.2表示20%）
MAX_WORKERS = 2 

# --- Phase Matching Target Wavelengths (um) ---
PM_WAVELENGTH_PUMP = 2.30
PM_WAVELENGTH_SHG = 1.15
PM_WAVELENGTH_SFG = 0.7667

# --- Physical Constraint Thresholds ---
MIN_CONVERSION_EFFICIENCY = 0.5   # Minimum theoretical efficiency. 
                                  # Below this, the topology is rejected without simulation.

# --- Warm Start (Baseline Injection) ---
USE_WARM_START = True


# =============================================================================
# 2. ATOMIC COMPONENTS LIBRARY (Super-Structure Definitions)
# =============================================================================
# Tags available for parameters:
# "SPSA"         -> Optimized by inner Top-2 SPSA loop
# "CMA"          -> Optimized by outer BIPOP-CMA-ES loop continuously
# "CMA_DISCRETE" -> Optimized by outer BIPOP-CMA-ES loop, but snapped to nearest STANDARD_LENGTHS
# "CMA_SKIP"     -> Special parameter. If <= 0, the entire block is skipped (Topology Search)

ATOMIC_LIBRARY = {
    "BiaxialCrystalBlock": {
        "params": {
            "type_score": {"min": -1.0, "max": 1.0, "tag": "CMA_SKIP"},
            "theta":      {"min": 0.0, "max": 180.0, "tag": "CMA"},
            # "phi":        {"min": 0.0, "max": 180.0, "tag": "CMA"},
            "length_um":  {"min": 0.0, "max": 1000.0, "tag": "CMA_DISCRETE"}
        },
        "template": "rotateIntoBiaxial(d,{theta:.4f},0.0000,d)nonlinear(d,{theta:.4f},0.0000,{length_um:.4f},d)rotateFromBiaxial(d,{theta:.4f},0.0000,d)"
    },
    "NormalCrystalBlock": {
        "params": {
            "type_score": {"min": -1.0, "max": 1.0, "tag": "CMA_SKIP"},
            "length_um":  {"min": 0.0, "max": 10000.0, "tag": "SPSA"}
        },
        "template": "nonlinear(20,0.0000,0.0000,{length_um:.4f},d)"
    },
    "LinearPropagation": {
        "params": {
            "type_score": {"min": -1.0, "max": 1.0, "tag": "CMA_SKIP"},
            "length_um": {"min": 0.0, "max": 1000.0, "tag": "SPSA"}
        },
        "template": "linear(0,0,0,{length_um:.4f},d)"
    },
    "RotateFrame": {
        "params": {
            "type_score": {"min": -1.0, "max": 1.0, "tag": "CMA_SKIP"},
            "angle": {"min": -180.0, "max": 180.0, "tag": "SPSA"}
        },
        "template": "rotate({angle:.4f})"
    },
    "SphericalMirror": {
        "params": {
            "type_score": {"min": -1.0, "max": 1.0, "tag": "CMA_SKIP"},
            "focal_m": {"min": -1.0, "max": 1.0, "tag": "SPSA"}
        },
        "template": "sphericalMirror({focal_m:.4f})"
    },
    "SpectralFilter": {
        "params": {
            "type_score": {"min": -1.0, "max": 1.0, "tag": "CMA_SKIP"},
            "f_center_THz": {"min": 50.0, "max": 1500.0, "tag": "SPSA"},
            "f_width_THz": {"min": 5.0, "max": 500.0, "tag": "SPSA"},
            "inBandAmplitude": {"min": -1.0, "max": 1.0, "tag": "SPSA"},
            "outOfBandAmplitude": {"min": 0.0, "max": 1.0, "tag": "SPSA"}
        },
        "template": "filter({f_center_THz:.2f},{f_width_THz:.2f},4,{inBandAmplitude:.4f},{outOfBandAmplitude:.4f})"
    },
    "Aperture": {
        "params": {
            "type_score": {"min": -1.0, "max": 1.0, "tag": "CMA_SKIP"},
            "d_m": {"min": 0.0, "max": 0.005, "tag": "SPSA"},
            "activation_parameter": {"min": 1.0, "max": 100.0, "tag": "SPSA"}
        },
        "template": "aperture({d_m:.4f},{activation_parameter:.2f})"
    }
}

class AtomicGene:
    def __init__(self, comp_type, global_id):
        self.comp_type = comp_type
        self.config = ATOMIC_LIBRARY[comp_type]
        self.global_id = global_id
        
    def get_specs(self, target_tag):
        specs = []
        for p_name, p_def in self.config["params"].items():
            if p_def["tag"].startswith(target_tag):
                specs.append({
                    "key": f"g{self.global_id}_{p_name}",
                    "name": p_name,
                    "min": p_def["min"],
                    "max": p_def["max"],
                    "tag": p_def["tag"]
                })
        return specs

class TopologyChromosome:
    def __init__(self, n_cells=MAX_CELLS):
        self.genes = []
        global_id = 0
        # Build Super-Structure: [BiaxialCrystalBlock, NormalCrystalBlock, LinearPropagation, RotateFrame] * n_cells
        for _ in range(n_cells):
            self.genes.append(AtomicGene("BiaxialCrystalBlock", global_id)); global_id += 1
            self.genes.append(AtomicGene("NormalCrystalBlock", global_id)); global_id += 1
            self.genes.append(AtomicGene("LinearPropagation", global_id)); global_id += 1
            self.genes.append(AtomicGene("RotateFrame", global_id)); global_id += 1
            
    def get_cma_specs(self):
        specs = []
        for gene in self.genes:
            specs.extend(gene.get_specs("CMA"))
        return specs
        
    def get_spsa_specs(self):
        specs = []
        for gene in self.genes:
            specs.extend(gene.get_specs("SPSA"))
        return specs

SUPER_TOPO = TopologyChromosome(MAX_CELLS)
CMA_SPECS = SUPER_TOPO.get_cma_specs()
SPSA_SPECS = SUPER_TOPO.get_spsa_specs()

# === ELITE INJECTION ===
BASELINE_CMA = np.zeros(len(CMA_SPECS))
BASELINE_SPSA = np.zeros(len(SPSA_SPECS))

for k, spec in enumerate(CMA_SPECS):
    key = spec["key"]
    # 必须激活对应组件的 type_score，否则会被全局跳过
    if key in ["g0_type_score", "g4_type_score", "g7_type_score", "g8_type_score"]:
        BASELINE_CMA[k] = 1.0
    elif key == "g0_length_um": BASELINE_CMA[k] = 400.0
    elif key == "g0_theta": BASELINE_CMA[k] = 0.0
    elif key == "g0_phi": BASELINE_CMA[k] = 0.0 
    elif key == "g4_theta": BASELINE_CMA[k] = 40.2
    elif key == "g4_length_um": BASELINE_CMA[k] = 250.0
    elif key == "g7_angle": BASELINE_CMA[k] = 180.0
    elif key == "g8_theta": BASELINE_CMA[k] = 139.8
    elif key == "g8_length_um": BASELINE_CMA[k] = 250.0

for k, spec in enumerate(SPSA_SPECS):
    pass


# =============================================================================
# 3. Physics Constraints Layer (Decoupled via regex parsing)
# =============================================================================
def evaluate_physics_from_seq(seq):
    """
    Evaluates physical constraints directly from the LWE sequence string.
    This completely decouples the physics evaluation from the topology generator.
    
    Returns:
        is_valid (bool): Whether the topology is worth simulating in LWE.
        penalty_multiplier (float): Soft penalty applied to the final efficiency.
        fallback_score (float): A smooth negative score guiding the optimizer if is_valid == False.
    """
    import re
    import bibo_phase_matching as bpm
    
    # The requirement for at least 2 crystals is mathematically enforced 
    # by checking both min_dk_shg and min_dk_sfg below!
    # Since we only check phase matching now, we can ignore rotations and just find all nonlinear blocks.
    pattern = r"nonlinear\(d,(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),d\)"
    matches = re.finditer(pattern, seq)
    
    min_eff_seq = 1.0
    active_crystals = 0
    
    for match in matches:
        # th is group 1, ph is group 2, L_um is group 3
        th = float(match.group(1))
        ph = float(match.group(2))
        L_um = float(match.group(3))
        L_m = L_um * 1e-6
        
        if L_um < 1.0:
            continue
            
        active_crystals += 1
        
        # Phase matching (SHG & SFG)
        dk_shg = bpm.get_min_dk(PM_WAVELENGTH_PUMP, PM_WAVELENGTH_PUMP, PM_WAVELENGTH_SHG, th, ph)
        dk_sfg = bpm.get_min_dk(PM_WAVELENGTH_PUMP, PM_WAVELENGTH_SHG, PM_WAVELENGTH_SFG, th, ph)
        
        # Conversion efficiency factor: sinc^2(dk * L / 2)
        # np.sinc(x) is defined as sin(pi*x)/(pi*x), so we need x = dk * L / (2 * pi)
        eff_shg = np.sinc(dk_shg * L_m / (2 * np.pi))**2
        eff_sfg = np.sinc(dk_sfg * L_m / (2 * np.pi))**2
        
        # The crystal is used for whichever process it's better matched for
        crystal_eff = max(eff_shg, eff_sfg)
        
        # The sequence is bottlenecked by its least efficient crystal
        if crystal_eff < min_eff_seq:
            min_eff_seq = crystal_eff
                
    total_multiplier = min_eff_seq if active_crystals > 0 else 0.0
    
    # --- Strict Limits (Pruning boundaries) ---
    is_valid = (total_multiplier >= MIN_CONVERSION_EFFICIENCY) and (active_crystals >= 1)
    
    if not is_valid:
        # Use theoretical efficiency to guide the optimizer when outside the valid region
        fallback_score = -5.0 + total_multiplier
        if active_crystals == 0:
            fallback_score -= 10.0
        return False, 0.0, fallback_score

    # --- No baseline penalty needed anymore, purely driven by efficiency ---
    return True, total_multiplier, 0.0


# =============================================================================
# 4. Simulation Execution Layer
# =============================================================================
def run_lwe_topology(cma_vector, spsa_vector):
    # Map vectors to dictionaries for easy lookup
    param_dict = {}
    bound_penalty = 0.0
    topology_drift_penalty = 0.0
    
    for spec, val in zip(CMA_SPECS, cma_vector):
        # Calculate normalized quadratic penalty for out-of-bounds to prevent the "giant plain"
        span = spec["max"] - spec["min"]
        if span > 0:
            if val < spec["min"]:
                bound_penalty += ((spec["min"] - val) / span) ** 2
            elif val > spec["max"]:
                bound_penalty += ((val - spec["max"]) / span) ** 2
                
        # Clip strictly to bounds for the physical simulation
        val = float(np.clip(val, spec["min"], spec["max"]))
        
        # Ensure we always snap to discrete float length
        if spec["tag"] == "CMA_DISCRETE":
            idx = np.argmin(np.abs(STANDARD_LENGTHS - val))
            val = float(STANDARD_LENGTHS[idx])
        param_dict[spec["key"]] = val
        
    for spec, val in zip(SPSA_SPECS, spsa_vector):
        param_dict[spec["key"]] = float(np.clip(val, spec["min"], spec["max"]))
        
    # Enforce passive filter limitation
    for gene in SUPER_TOPO.genes:
        if gene.comp_type == "SpectralFilter":
            in_key = f"g{gene.global_id}_inBandAmplitude"
            out_key = f"g{gene.global_id}_outOfBandAmplitude"
            if in_key in param_dict and out_key in param_dict:
                param_dict[out_key] = 1.0 - param_dict[in_key]
                
    # Render string
    seq_parts = ["init()"]
    for gene in SUPER_TOPO.genes:
        skip_key = f"g{gene.global_id}_type_score"
        ts = param_dict.get(skip_key, 1.0)
        
        if ts <= 0:
            # Mathematical drift: pull deactivated components slightly towards 0
            # so they don't wander infinitely into the negative plateau.
            topology_drift_penalty += (ts) ** 2 * 0.1
            continue # Component deactivated
            
        gene_params = {}
        for p_name in gene.config["params"].keys():
            if p_name != "type_score": # type_score is not in the template
                val = param_dict[f"g{gene.global_id}_{p_name}"]
                
                # --- Smooth Physical Blending ---
                # As type_score approaches 0 from above (0 to 0.2), we physically scale down 
                # the component's main parameters. This flawlessly smooths out the cliff at 0!
                if ts < 0.2:
                    scale = ts / 0.2
                    if "length_um" in p_name or "angle" in p_name or "d_m" in p_name:
                        val *= scale
                        
                gene_params[p_name] = val
                
        # --- Smooth Topology Extinction ---
        # If length or angle is driven to 0 by the optimizer, we safely prune the block
        # to avoid any PDE division-by-zero or wasted parsing
        if "length_um" in gene_params and gene_params["length_um"] <= 0.0:
            continue
        if "angle" in gene_params and abs(gene_params["angle"]) <= 0.0:
            continue
                
        seq_parts.append(gene.config["template"].format(**gene_params))
        
    seq = "".join(seq_parts)
    
    # --- PHYSICAL CONSTRAINT PRUNING ---
    is_valid, penalty_multiplier, fallback_score = evaluate_physics_from_seq(seq)
    
    if not is_valid:
        return fallback_score, seq, 0.0
    
    # Inline Evaluation
    eval_uuid = str(uuid.uuid4())[:6]
    work_dir = os.path.join(TMP_DIR, f"bipop_{eval_uuid}")
    os.makedirs(work_dir, exist_ok=True)
    
    runner = lwe.SimulationRunner(cli_path=CLI_PATH, work_dir=work_dir)
    eff = fallback_score  # Start with the micro-gradient baseline
    raw_eff = 0.0
    
    try:
        runner.set_params(
            sequence=seq,
            pulse_energy1=1.9e-08, frequency1=1.3e14, bandwidth1=1.5e13,
            sg_order1=2, beamwaist1=1.12e-5, delay1=-1.3e-13,
            polarization1=1.5707963267949,
            pulse_energy2=0, frequency2=4.9e14, bandwidth2=2e13,
            sg_order2=4, gdd2=1e-28, beamwaist2=9e-5,
            polarization2=1.5707963267949,
            material_index=26, crystal_thickness=0.0004, dz=1e-6,
            grid_width=2.08e-4, grid_height=2.08e-4, dx=2e-5,
            time_span=6e-13, dt=5e-16, band_gap=6, effective_mass=1, drude_gamma=5e12,
            propagation_mode=2
        )
        runner.run(verbose=False, timeout=800)
        
        freq = runner.result.frequencyVectorSpectrum
        spectrum = runner.result.spectrumTotal
        
        # Target THG (390 THz = 3.9e14 Hz)
        thg_center = 390e12
        mask_thg = np.abs(freq - thg_center) < 30e12
        
        if np.any(mask_thg):
            # NumPy 2.0+ uses np.trapezoid instead of np.trapz
            E_thg = np.trapezoid(spectrum[mask_thg], freq[mask_thg])
            E_total = np.trapezoid(spectrum, freq)
            
            if E_total > 0:
                raw_eff = E_thg / E_total
                # Apply Soft Penalty from evaluator AND the continuous micro-gradient
                eff = raw_eff * penalty_multiplier + fallback_score
    except Exception as e:
        print(f"[Simulation Error] {e}")
        eff = fallback_score
        raw_eff = 0.0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        
    # Apply penalties: steep wall for out-of-bounds, gentle pull for topology drift
    eff = eff - bound_penalty * 100.0 - topology_drift_penalty
    return eff, seq, raw_eff

# =============================================================================
# 4. SPSA Gradient Optimizer (Inner Loop)
# =============================================================================
def spsa_assisted_eval(cma_params, init_spsa_params):
    best_seq = ""
    best_eff = -np.inf
    best_raw_eff = 0.0
    
    def inner_obj(p):
        nonlocal best_seq, best_eff, best_raw_eff
        eff, seq, raw_eff = run_lwe_topology(cma_params, p)
        if eff > best_eff:
            best_eff = eff
            best_seq = seq
            best_raw_eff = raw_eff
        return eff
        
    p = np.copy(init_spsa_params)
    inner_obj(p) # Base eval
    
    bounds_lo = np.array([s["min"] for s in SPSA_SPECS])
    bounds_hi = np.array([s["max"] for s in SPSA_SPECS])
    
    for _ in range(MAX_LBFGS_ITER):
        delta = np.random.choice([-1.0, 1.0], size=len(p))
        
        p_plus = np.clip(p + SPSA_STEP * delta, bounds_lo, bounds_hi)
        y_plus = inner_obj(p_plus)
        
        p_minus = np.clip(p - SPSA_STEP * delta, bounds_lo, bounds_hi)
        y_minus = inner_obj(p_minus)
        
        actual_delta = (p_plus - p_minus) / 2.0
        safe_delta = np.where(np.abs(actual_delta) < 1e-6, 1.0, actual_delta)
        grad = (y_plus - y_minus) / safe_delta
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            p = np.clip(p + (grad / grad_norm) * SPSA_LR, bounds_lo, bounds_hi)
            inner_obj(p)
            
    return best_eff, p, best_seq, best_raw_eff

# =============================================================================
# 5. BIPOP-CMA-ES Evolution (Outer Loop)
# =============================================================================
def main():
    lower_bounds = [s["min"] for s in CMA_SPECS]
    upper_bounds = [s["max"] for s in CMA_SPECS]
    stds = [(u - l) for u, l in zip(upper_bounds, lower_bounds)]
    
    opts = cma.CMAOptions()
    opts.set('bounds', [lower_bounds, upper_bounds])
    
    # Looser convergence criteria. Physics simulations have numerical noise, 
    # so we prevent the optimizer from getting stuck trying to resolve tiny differences.
    # This ensures BIPOP restarts can trigger properly when stagnated.
    opts.set('tolfun', 1e-6)
    opts.set('tolx', 1e-6)
    
    opts.set('seed', 42)
    opts.set('CMA_stds', stds)
    
    min_popsize = 4 + int(3 * np.log(len(CMA_SPECS)))
    popsize_small = max(POPSIZE_SMALL, min_popsize)
    popsize_large = max(POPSIZE_LARGE, popsize_small * 2)
    evals = 0
    run_large = True
    restart_count = 0
    
    # Init SPSA vector
    if USE_WARM_START and len(BASELINE_SPSA) == len(SPSA_SPECS):
        current_spsa = np.array(BASELINE_SPSA)
    else:
        current_spsa = np.array([(s["max"] + s["min"]) / 2.0 for s in SPSA_SPECS])
    
    global_best_eff = 0.0
    global_best_raw_eff = 0.0
    global_best_seq = ""
    
    history_file = os.path.join(TMP_DIR, "bipop_history.csv")
    with open(history_file, "w") as f:
        f.write("Evals,Restart,PopSize,BestGenRawEff,GlobalBestRawEff,BestGenSequence\n")
    
    print(f"{'='*60}")
    print(f"Starting Mathematical Global Search (BIPOP-CMA-ES + SPSA)")
    print(f"CMA Dimensions: {len(CMA_SPECS)} (Min required PopSize: {min_popsize})")
    print("CMA Variables:", [s["name"] for s in CMA_SPECS[:6]], "...")
    print(f"SPSA Dimensions: {len(current_spsa)}")
    print("SPSA Variables:", [s["name"] for s in SPSA_SPECS[:6]], "...")
    print(f"Workers: {MAX_WORKERS}")
    print(f"{'='*60}")
    
    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    
    while evals < BUDGET:
        current_pop = popsize_large if run_large else popsize_small
        print(f"\n>>> Starting BIPOP Restart #{restart_count} (Large: {run_large}) | PopSize: {current_pop}")
        
        opts.set('popsize', current_pop)
        
        # Warm Start Logic
        if USE_WARM_START and restart_count == 0 and len(BASELINE_CMA) == len(CMA_SPECS):
            x0 = BASELINE_CMA
            current_sigma = 0.05  # Warm start uses a very small focused search radius (2%)
            print(">>> [INFO] Using User Baseline for Warm Start! (Sigma=0.05) <<<")
            
            print(">>> [INFO] Evaluating Pure Baseline Efficiency... <<<")
            _, baseline_seq, baseline_raw = run_lwe_topology(x0, current_spsa)
            print(f">>> [BASELINE] Raw THG: {baseline_raw * 100:.4f}% | Seq: {baseline_seq}\n")
        else:
            x0 = [(hi + lo) / 2.0 for lo, hi in zip(lower_bounds, upper_bounds)]
            current_sigma = SIGMA0 # Global restart uses larger radius (20%)
            
        es = cma.CMAEvolutionStrategy(x0, current_sigma, opts)
        gen = 0
        restart_best_eff = 0.0
        restart_best_seq = ""
        
        cov_history = []
        sigma_history = []
        mean_history = []
        fitness_history = []
        
        while not es.stop():
            solutions = es.ask()
            
            futures = []
            for sol in solutions:
                futures.append(executor.submit(run_lwe_topology, sol, current_spsa))
                
            cheap_effs = []
            cheap_raw_effs = []
            cheap_seqs = []
            
            for future in futures:
                eff, seq, raw_eff = future.result()
                cheap_effs.append(eff)
                cheap_raw_effs.append(raw_eff)
                cheap_seqs.append(seq)
                evals += 1
                
            top2_idx = np.argsort(cheap_effs)[-2:][::-1]
            final_effs = list(cheap_effs)
            
            best_gen_eff = -np.inf
            best_gen_raw_eff = 0.0
            best_gen_spsa = np.copy(current_spsa)
            best_gen_seq = ""
            
            spsa_futures = []
            for idx in top2_idx:
                spsa_futures.append((idx, executor.submit(spsa_assisted_eval, solutions[idx], current_spsa)))
                
            for idx, future in spsa_futures:
                opt_eff, opt_spsa, opt_seq, opt_raw_eff = future.result()
                final_effs[idx] = opt_eff
                evals += 3 # SPSA consumes approx 3 extra evals
                
                if opt_eff > best_gen_eff:
                    best_gen_eff = opt_eff
                    best_gen_raw_eff = opt_raw_eff
                    best_gen_spsa = opt_spsa
                    best_gen_seq = opt_seq
                    
            # Check if any non-SPSA cheap evaluation happened to be better
            cheap_best_idx = int(np.argmax(final_effs))
            if final_effs[cheap_best_idx] > best_gen_eff:
                best_gen_eff = final_effs[cheap_best_idx]
                best_gen_raw_eff = cheap_raw_effs[cheap_best_idx]
                best_gen_seq = cheap_seqs[cheap_best_idx]
                
            # Update global best tracker using the actual LWE raw efficiency!
            if best_gen_eff > global_best_eff:
                global_best_eff = best_gen_eff
                global_best_seq = best_gen_seq
                global_best_raw_eff = best_gen_raw_eff
                print(f"*** NEW GLOBAL BEST (Raw THG): {global_best_raw_eff*100:.4f}% | Evals: {evals}")
                print(global_best_seq)
                
            if best_gen_eff > restart_best_eff:
                restart_best_eff = best_gen_eff
                restart_best_seq = best_gen_seq
                
            current_spsa = np.copy(best_gen_spsa)
            es.tell(solutions, [-eff for eff in final_effs])
            
            # Record the mathematical history for this generation
            # cma uses "Lazy Update" for C matrix. Reconstruct exactly.
            if getattr(es, 'B', None) is not None and getattr(es, 'D', None) is not None:
                current_C = np.dot(es.B, (es.D ** 2)[:, None] * es.B.T)
            else:
                current_C = np.copy(es.C) 
            cov_history.append(current_C)
            sigma_history.append(es.sigma)
            mean_history.append(np.copy(es.mean))
            
            # Record best and median fitness of this generation
            median_eff = np.median(final_effs)
            fitness_history.append((best_gen_eff, median_eff))
            
            # Save history to CSV to prevent loss
            with open(history_file, "a") as f:
                f.write(f"{evals},{restart_count},{current_pop},{best_gen_raw_eff*100:.4f},{global_best_raw_eff*100:.4f},{best_gen_seq}\n")
                
            gen += 1
            if gen % 5 == 0:
                print(f"Gen {gen:03d} | Evals: {evals:05d} | Best Gen Raw Eff: {best_gen_raw_eff*100:.4f}%")
                print(f"-> Seq: {best_gen_seq}")
                
                # Dynamic save to prevent data loss
                np.savez(os.path.join(TMP_DIR, f"bipop_math_history_analyze_{restart_count}.npz"),
                         cov=np.array(cov_history),
                         sigma=np.array(sigma_history),
                         mean=np.array(mean_history),
                         fitness=np.array(fitness_history))
            
            if evals >= BUDGET:
                break
                
        print(f"\n--- [Restart #{restart_count} CONVERGED / FINISHED] ---")
        print(f"Best Efficiency in this round: {restart_best_eff*100:.4f}%")
        print(f"Best Sequence in this round: {restart_best_seq}")
        print("-" * 50)
                
        # Save final complete mathematical history
        np.savez(os.path.join(TMP_DIR, f"bipop_math_history_analyze_{restart_count}.npz"),
                 cov=np.array(cov_history),
                 sigma=np.array(sigma_history),
                 mean=np.array(mean_history),
                 fitness=np.array(fitness_history))
                
        restart_count += 1
        if run_large:
            popsize_large *= 2
        run_large = not run_large
        
    executor.shutdown()
    print("\n" + "="*50)
    print("Optimization Finished!")
    print(f"Global Best Efficiency: {global_best_eff*100:.4f}%")
    print("\nOptimal Topology Sequence:")
    print(global_best_seq)

if __name__ == "__main__":
    main()
