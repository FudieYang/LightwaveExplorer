"""
LWE 单次仿真性能剖析脚本
用法: python3 profile_lwe.py
会跑一次完整仿真，精确拆解每个阶段的耗时。
还会测试不同 dx 和 dz 下的速度，帮你找到最佳性价比。
"""
import os, sys, time, shutil, uuid
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz  # NumPy < 2.0 compatibility

PROJECT_ROOT = "/home/fudie/LightwaveExplorer"
CLI_PATH     = os.path.join(PROJECT_ROOT, "build_cli", "LightwaveExplorer")
TMP_DIR      = "/mnt/d/lwe_tmp"

sys.path.append(os.path.join(PROJECT_ROOT, "Source", "Python", "src"))
import LightwaveExplorer as lwe

# 动态构建一个合法的 sequence（使用和优化器相同的模板）
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Source", "Python"))
from island_cmaes_optimizer import SUPER_TOPO, CMA_SPECS, GRAD_SPECS
import numpy as np

def build_test_sequence():
    """用默认初始值构建一个合法的 sequence"""
    param_dict = {}
    for key, lo, hi, x0 in CMA_SPECS:
        param_dict[key] = x0
    for key, lo, hi, x0 in GRAD_SPECS:
        param_dict[key] = x0

    parts = []
    for i, gene in enumerate(SUPER_TOPO.genes):
        kwargs = {}
        for p_name in gene.config["params"]:
            kwargs[p_name] = param_dict[f"g{i}_{p_name}"]
        parts.append(gene.config["template"].format(**kwargs))
    return "init()" + "".join(parts)

TEST_SEQ = build_test_sequence()
print(f"Test sequence: {TEST_SEQ[:200]}...")

FIXED_PARAMS = dict(
    pulse_energy1=1.9e-08, frequency1=1.3e14, bandwidth1=1.5e13,
    sg_order1=2, beamwaist1=1.12e-5, delay1=-1.3e-13,
    polarization1=1.5707963267949,
    pulse_energy2=0, frequency2=4.9e14, bandwidth2=2e13,
    sg_order2=4, gdd2=1e-28, beamwaist2=9e-5,
    polarization2=1.5707963267949,
    material_index=4, crystal_thickness=0.0004,
    grid_width=2.08e-4, grid_height=2.08e-4,
    time_span=6e-13, band_gap=6, effective_mass=1, drude_gamma=5e12,
    propagation_mode=2,
)


def run_one(dx, dz, dt, label):
    work_dir = os.path.join(TMP_DIR, f"profile_{uuid.uuid4().hex[:6]}")
    os.makedirs(work_dir, exist_ok=True)
    runner = lwe.SimulationRunner(cli_path=CLI_PATH, work_dir=work_dir)

    # Setup
    t0 = time.perf_counter()
    runner.set_params(sequence=TEST_SEQ, dx=dx, dz=dz, dt=dt, **FIXED_PARAMS)
    t_setup = time.perf_counter() - t0

    # Write settings file
    t0 = time.perf_counter()
    runner.write_settings_file(os.path.join(work_dir, "lwe_output.txt"))
    t_write = time.perf_counter() - t0

    # Run simulation (THE BOTTLENECK)
    t0 = time.perf_counter()
    try:
        runner.run(verbose=True, timeout=1200)
    except Exception as e:
        print(f"  [{label}] FAILED: {e}")
        shutil.rmtree(work_dir, ignore_errors=True)
        return None
    t_sim = time.perf_counter() - t0

    # Read result
    t0 = time.perf_counter()
    freq = runner.result.frequencyVectorSpectrum
    spectrum = runner.result.spectrumTotal
    if spectrum.ndim > 1:
        spectrum = spectrum[-1]
    thg_center = 390e12
    mask = np.abs(freq - thg_center) < 30e12
    eff = float(np.trapz(spectrum[mask], freq[mask]) / 1.9e-08)
    t_post = time.perf_counter() - t0

    # Cleanup
    t0 = time.perf_counter()
    runner.cleanup()
    shutil.rmtree(work_dir, ignore_errors=True)
    t_clean = time.perf_counter() - t0

    # Grid dimensions
    nx = int(round(2.08e-4 / dx))
    ny = nx
    nz = int(round(0.0004 / dz))
    nt = int(round(6e-13 / dt))

    print(f"\n{'='*60}")
    print(f"[{label}] dx={dx:.1e}, dz={dz:.1e}, dt={dt:.1e}")
    print(f"  Grid: {nx}x{ny} spatial | {nz} z-steps | {nt} time-steps")
    print(f"  ──────────────────────────────────────")
    print(f"  参数设置:       {t_setup*1000:8.1f} ms")
    print(f"  写入文件:       {t_write*1000:8.1f} ms")
    print(f"  ★ GPU仿真:    {t_sim:8.1f} s   ← 这个是瓶颈")
    print(f"  结果读取+分析:  {t_post*1000:8.1f} ms")
    print(f"  清理临时文件:   {t_clean*1000:8.1f} ms")
    print(f"  ──────────────────────────────────────")
    print(f"  总耗时:         {t_setup+t_write+t_sim+t_post+t_clean:8.1f} s")
    print(f"  THG效率:        {eff*100:.4f}%")
    print(f"{'='*60}")

    return {
        "label": label, "dx": dx, "dz": dz, "dt": dt,
        "nx": nx, "nz": nz, "nt": nt,
        "t_sim": t_sim, "eff": eff
    }


if __name__ == "__main__":
    print("LWE 性能剖析")
    print(f"CLI: {CLI_PATH}")
    print(f"TMP: {TMP_DIR}")
    print()

    configs = [
        # (dx, dz, dt, label)
        (8e-6,  5e-7, 5e-16, "当前精度"),
        (1.6e-5, 5e-7, 5e-16, "dx x2 (空间粗化)"),
        (8e-6,  1e-6, 5e-16, "dz x2 (传播粗化)"),
        (1.6e-5, 1e-6, 5e-16, "dx+dz 双粗化"),
        (2e-5,  1e-6, 5e-16, "dx+dz 激进粗化"),
    ]

    results = []
    for dx, dz, dt, label in configs:
        r = run_one(dx, dz, dt, label)
        if r:
            results.append(r)

    if len(results) > 1:
        print(f"\n\n{'='*70}")
        print("对比总结")
        print(f"{'='*70}")
        print(f"{'配置':<20} {'Grid':<20} {'仿真耗时':>10} {'加速比':>8} {'效率':>10}")
        print(f"{'-'*70}")
        base_t = results[0]["t_sim"]
        for r in results:
            speedup = base_t / r["t_sim"] if r["t_sim"] > 0 else 0
            grid = f"{r['nx']}x{r['nx']}x{r['nz']}x{r['nt']}"
            print(f"{r['label']:<20} {grid:<20} {r['t_sim']:>8.1f}s {speedup:>7.1f}x {r['eff']*100:>9.4f}%")
