<p align="center"><img src="Source/BuildResources/Icon.svg" width="256" height="256"></p>

## <p align="center">Lightwave Explorer</p>
<p align="center"><b>Python Optimization by Fudie Yang</b></p>

> **Original Project Notice & Citation:**  
> This project is a fork based on **Lightwave Explorer** created by **Nick Karpowicz** (Max Planck Institute of Quantum Optics).  
> Please reference the original publication, tutorials, and documentation below:

---

### Original Publication!
- N. Karpowicz, Open source, heterogenous, nonlinear-optics simulation. [*Optics Continuum* **2**, 2244-2254 (2023).](https://opg.optica.org/optcon/fulltext.cfm?uri=optcon-2-11-2244&id=540999)

### Tutorials on YouTube!
- <a href="https://youtu.be/JY4wm2e7y_M">Explanation of the new beam modes in version 2026.1</a>
- <a href="https://youtu.be/sZ5Evkgj4Vk">Introduction, main tutorial (2026 update!)</a>
- <a href="https://youtu.be/qlcy_RBLGoU">Adding a crystal to the database</a>
- <a href="https://youtu.be/v5O0UOUdfKE">Birefringence</a>
- <a href="https://www.youtube.com/watch?v=4njswvog4bo">FDTD</a>

[Original Documentation](https://nickkarpowicz.github.io/LightwaveExplorerDocumentation)!

---

## Fork Extensions & Features

The main purpose of this fork is to provide a **more flexible Python interface**, enabling users to programmatically control, script, and interact with Lightwave Explorer directly inside Python environments.

Key capabilities added:
- **Flexible Python Interface**: Easily configure simulation parameters, execute C++ / GPU backends via `SimulationRunner`, and load binary field and spectrum results directly into NumPy/SciPy arrays for post-processing and custom workflows:
  ```python
  import LightwaveExplorer as lwe

  # Initialize a simulation runner using the compiled CLI binary (e.g., build/LightwaveExplorer)
  runner = lwe.SimulationRunner(cli_path="build/LightwaveExplorer", work_dir="/tmp/lwe_run")

  # Configure simulation parameters
  runner.set_params(
      sequence="init()rotateIntoBiaxial(d,40.2,0.0,d)nonlinear(d,40.2,0.0,250.0,d)rotateFromBiaxial(d,40.2,0.0,d)",
      pulse_energy1=1.9e-08, frequency1=1.3e14, bandwidth1=1.5e13,
      material_index=26, crystal_thickness=0.0004
  )

  # Run the simulation
  runner.run()

  # Access spectrum and electric field results directly as NumPy arrays
  freq = runner.result.frequencyVectorSpectrum
  spectrum = runner.result.spectrumTotal
  ```
- **Automated Optimization**: Includes scripts in `Source/Python` for optical sequence searching, parameter fitting, and landscape diagnostics. For detailed methodology and results, please refer to my upcoming Master's thesis on the [Attoworld website](https://attoworld.de/) (currently unreleased).

---

## Installation & Setup

### 1. Download & Install Python Package
```bash
# Clone the repository
git clone https://github.com/FudieYang/LightwaveExplorer.git
cd LightwaveExplorer

# Install the Python package
pip install Source/Python
```

### 2. Compile CLI Binary on Linux (Required for `SimulationRunner`)
To run simulations programmatically via Python, compile the C++ Command Line Interface (CLI) binary using CMake with `-DCLI=1`.

> **Note & Reference**: Below are quick Linux build commands used in this fork. For full compilation details, advanced CMake options, and other operating systems (Windows, Mac, SLURM Clusters), please refer to the original [Lightwave Explorer Repository](https://github.com/NickKarpowicz/LightwaveExplorer) and [Documentation](https://nickkarpowicz.github.io/LightwaveExplorerDocumentation).

#### Prerequisites
System development libraries required on Linux (e.g., Fedora/dnf or Ubuntu/apt equivalents): `fmt-devel`, `qt6-qtbase-devel`, `cairo-devel`, `tbb-devel`.

#### Basic CPU Build
```bash
mkdir build && cd build
cmake -DCLI=1 ..
cmake --build . --config Release
```

#### NVIDIA CUDA GPU Acceleration
> **Note**: Ensure the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) is installed beforehand.
```bash
mkdir build && cd build
cmake -DCLI=1 -DUSE_CUDA=1 -DCMAKE_CUDA_HOST_COMPILER=g++ -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86 ..
cmake --build . --config Release
```

#### SYCL GPU Acceleration (Intel / AMD)
> **Note**: Ensure Intel OneAPI Base Toolkit (`icpx`) or DPC++ compiler for AMD ROCm is installed beforehand.
```bash
# Intel GPU
cmake -DCLI=1 -DUSE_SYCL=1 -DCMAKE_CXX_COMPILER=icpx ..
cmake --build . --config Release

# AMD GPU (ROCm)
cmake -DCLI=1 -DUSE_SYCL=1 -DBACKEND_ROCM=gfx906 -DROCM_LIB_PATH=/usr/lib/clang/18/amdgcn/bitcode -DCMAKE_CXX_COMPILER=clang++ ..
cmake --build . --config Release
```

---

## Overview of Added Files (`Source/Python`)

This fork introduces several key Python scripts and modules in `Source/Python`:

- **`global_bipop_optimizer.py`**: Dual-loop optimizer combining BIPOP-CMA-ES (for non-convex global sequence and orientation search) and SPSA (for fast local parameter tuning).
- **`landscape_analyzer.py`**: Diagnostic framework for analyzing phase-matching maps, Morris elementary effects sensitivity ($\mu^*, \sigma$), and generating 3D interactive surfaces.
- **`bibo_phase_matching.py`**: Sellmeier dispersion solver, Fresnel equation solver, and phase mismatch ($\Delta k$) engine for BiBO crystals.
- **`sequence_pruner.py`**: Knock-out ablation tool to mute sequence elements one-by-one and extract minimal core configurations.
- **`island_cmaes_optimizer.py`**: Multi-island CMA-ES optimizer with dynamic elite migration.
- **`geometry_search.py`**: Structural and geometrical search framework *(early version, un-tuned)*. Contains initial implementations of Optuna parameter searching, and a PyTorch-based Transformer surrogate ranking model.



