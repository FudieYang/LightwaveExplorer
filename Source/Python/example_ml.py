"""
Example: Using LightwaveExplorer with Python for ML workflows.

This script demonstrates how to use the SimulationRunner class to:
1. Load parameters from an existing simulation file
2. Override specific parameters
3. Run simulations programmatically
4. Export results for ML training

Prerequisites:
    - Build the LWE CLI: cmake -B build -DCLI=ON && cmake --build build
    - Install the Python package: pip install -e Source/Python/
"""

import numpy as np
import LightwaveExplorer as lwe

# ============================================================
# Example 1: Basic usage — load params from file, run, get data
# ============================================================

runner = lwe.SimulationRunner(cli_path="./build/LightwaveExplorer")

# Load parameters from an existing simulation result (from GUI or previous run)
runner.load_params("path/to/existing_simulation.txt")  # or .zip

# Override just the parameters you want to change
runner.set_params(
    crystal_thickness=500e-6,
    pulse_energy1=1e-6,
)

# Run the simulation
result = runner.run()

# Access results as numpy arrays
spectrum = result.spectrumTotal
print(f"Spectrum shape: {spectrum.shape}")

# Export everything to a flat dictionary (great for ML)
data = result.to_dict()
print(f"Available keys: {list(data.keys())}")

# Save as .npz for later use
result.to_npz("my_result.npz")

# ============================================================
# Example 2: Parameter scan for ML training data generation
# ============================================================

# Scan crystal thickness from 100 to 1000 microns
thicknesses = np.linspace(100e-6, 1000e-6, 20)
results = runner.batch_run("crystal_thickness", thicknesses, output_dir="./ml_data")

# Convert to ML training format
X = np.array([r["spectrumTotal"] for r in results])       # features
y = np.array([r["crystalThickness"] for r in results])     # labels
print(f"Training data: X={X.shape}, y={y.shape}")

# Save training dataset
np.savez_compressed("training_data.npz", X=X, y=y)

# ============================================================
# Example 3: From scratch — set all parameters manually
# ============================================================

runner2 = lwe.SimulationRunner(cli_path="./build/LightwaveExplorer")
runner2.set_params(
    # Pulse 1
    pulse_energy1=1e-6,
    frequency1=375e12,
    bandwidth1=10e12,
    beamwaist1=50e-6,
    # Crystal
    material_index=4,
    crystal_thickness=500e-6,
    crystal_theta=0.5,
    # Grid
    grid_width=200e-6,
    time_span=200e-15,
    dx=1e-6,
    dt=0.2e-15,
    dz=1e-7,
)
result2 = runner2.run()
print(f"Simulation complete. Peak spectrum value: {result2.spectrumTotal.max():.2e}")

# Cleanup temp files
runner.cleanup()
runner2.cleanup()
