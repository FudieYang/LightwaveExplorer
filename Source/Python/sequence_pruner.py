import os
import sys
import uuid
import numpy as np
import shutil
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "Source", "Python", "src"))
import LightwaveExplorer as lwe

CLI_PATH = os.path.join(PROJECT_ROOT, "build_cli", "LightwaveExplorer")

BEST_SEQ = "init()rotateIntoBiaxial(d,0.0000,0.0681,d)nonlinear(d,0.0000,0.0681,329.7325,d)rotateFromBiaxial(d,0.0000,0.0681,d)nonlinear(6,0.0000,0.0000,2.0000,d)linear(0,0,0,2.0000,d)rotate(2.6495)rotateIntoBiaxial(d,39.1409,14.2471,d)nonlinear(d,39.1409,14.2471,68.2712,d)rotateFromBiaxial(d,39.1409,14.2471,d)nonlinear(6,0.0000,0.0000,68.6897,d)linear(0,0,0,2.1766,d)rotate(176.7613)rotateIntoBiaxial(d,159.3270,44.4355,d)nonlinear(d,159.3270,44.4355,426.8457,d)rotateFromBiaxial(d,159.3270,44.4355,d)nonlinear(6,0.0000,0.0000,66.0311,d)linear(0,0,0,9.4913,d)rotate(-178.6302)rotateIntoBiaxial(d,2.2797,17.4657,d)nonlinear(d,2.2797,17.4657,189.0253,d)rotateFromBiaxial(d,2.2797,17.4657,d)nonlinear(6,0.0000,0.0000,76.8911,d)linear(0,0,0,9.5057,d)rotate(12.4835)rotateIntoBiaxial(d,13.7523,1.0670,d)nonlinear(d,13.7523,1.0670,54.1040,d)rotateFromBiaxial(d,13.7523,1.0670,d)nonlinear(6,0.0000,0.0000,55.9328,d)linear(0,0,0,52.9407,d)rotate(-50.2048)"

def evaluate_sequence(seq_str, tag="baseline"):
    work_dir = os.path.join("/tmp", f"prune_{tag}_{uuid.uuid4().hex[:6]}")
    os.makedirs(work_dir, exist_ok=True)
    runner = lwe.SimulationRunner(cli_path=CLI_PATH, work_dir=work_dir)
    try:
        runner.set_params(
            sequence=seq_str,
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
        if spectrum.ndim > 1: spectrum = spectrum[-1]
        mask_thg = np.abs(freq - 390e12) < 30e12
        thg_pwr = np.trapz(spectrum[mask_thg], freq[mask_thg])
        eff = float(thg_pwr / 1.9e-08)
        if np.isnan(eff) or np.isinf(eff) or eff > 1.0 or eff < 0: return 0.0
        return eff
    except Exception:
        return 0.0
    finally:
        runner.cleanup()
        shutil.rmtree(work_dir, ignore_errors=True)

def parse_sequence(seq):
    parts = [p + ")" for p in seq.split(")") if p]
    blocks = []
    i = 0
    c_idx, g_idx, a_idx, r_idx = 0, 0, 0, 0
    while i < len(parts):
        if parts[i].startswith("rotateIntoBiaxial"):
            thick = float(parts[i+1].split(',')[3])
            blocks.append({
                "type": "Crystal", 
                "str": parts[i] + parts[i+1] + parts[i+2],
                "desc": f"Crystal {c_idx} (L={thick:.1f} um)"
            })
            c_idx += 1
            i += 3
        elif parts[i].startswith("nonlinear"):
            thick = float(parts[i].split(',')[3])
            blocks.append({
                "type": "Glass",
                "str": parts[i],
                "desc": f"Glass {g_idx} (L={thick:.1f} um)"
            })
            g_idx += 1
            i += 1
        elif parts[i].startswith("linear"):
            thick = float(parts[i].split(',')[3])
            blocks.append({
                "type": "Air",
                "str": parts[i],
                "desc": f"Air {a_idx} (L={thick:.1f} um)"
            })
            a_idx += 1
            i += 1
        elif parts[i].startswith("rotate("):
            rot = float(parts[i].split('(')[1].replace(')',''))
            blocks.append({
                "type": "Rotation",
                "str": parts[i],
                "desc": f"Rot {r_idx} ({rot:.1f} deg)"
            })
            r_idx += 1
            i += 1
        elif parts[i].startswith("init"):
            blocks.append({
                "type": "Init",
                "str": parts[i],
                "desc": "Init"
            })
            i += 1
        else:
            i += 1
    return blocks

if __name__ == "__main__":
    blocks = parse_sequence(BEST_SEQ)
    print("=== Reconstructing Baseline ===")
    baseline_seq = "".join(b["str"] for b in blocks)
    base_eff = evaluate_sequence(baseline_seq, "baseline")
    print(f"Baseline Efficiency: {base_eff*100:.4f}%\n")
    
    print("=== Ablation Study (Knock-out Analysis) ===")
    print("Muting each component one by one to test its isolated impact...\n")
    results = []
    
    results = []
    for i, b in enumerate(blocks):
        if b["type"] == "Init":
            continue
        muted_seq = "".join(bl["str"] for j, bl in enumerate(blocks) if j != i)
        
        # Ensure init() is there
        if not muted_seq.startswith("init()"):
            muted_seq = "init()" + muted_seq.replace("init()", "")
            
        eff = evaluate_sequence(muted_seq, f"mut_{i}")
        drop = base_eff - eff
        results.append({
            "index": i,
            "desc": b["desc"],
            "type": b["type"],
            "eff": eff,
            "drop": drop
        })
        print(f"Evaluated ablation of {b['desc']}: Eff = {eff*100:.4f}% (Drop: {drop*100:.4f}%)")
                
    results.sort(key=lambda x: x["drop"], reverse=True)
    
    print(f"{'Rank':<5} | {'Component':<25} | {'Muted Eff':<12} | {'Eff Drop':<12} | {'Verdict'}")
    print("-" * 80)
    for rank, r in enumerate(results):
        drop_pct = r['drop'] * 100
        eff_pct = r['eff'] * 100
        verdict = "CORE" if drop_pct > 1.0 else ("MINOR" if drop_pct > 0.05 else "USELESS")
        print(f"{rank+1:<5} | {r['desc']:<25} | {eff_pct:>8.4f}%   | {drop_pct:>8.4f}%   | {verdict}")
