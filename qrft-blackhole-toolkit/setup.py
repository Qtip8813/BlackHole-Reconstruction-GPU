"""
QRFT Black Hole Toolkit — Setup & Dependency Installer
========================================================
Detects your hardware, installs all required libraries,
and verifies everything works.

Run with:
    python setup.py

What it does:
    1. Checks Python version
    2. Installs core dependencies (numpy, scipy, matplotlib)
    3. Detects NVIDIA GPU and CUDA version
    4. Installs the correct CuPy package for your GPU
    5. Optionally installs ehtim + astropy for real EHT data
    6. Runs a quick verification test

Created by: Rodney Lee Arnold Jr. ∞ 0425
"""

import subprocess
import sys
import os
import platform
import shutil


def run_cmd(cmd, capture=True, check=False):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture,
            text=True, timeout=300
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout", 1
    except Exception as e:
        return "", str(e), 1


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_ok(msg):
    print(f"  ✓ {msg}")


def print_warn(msg):
    print(f"  ⚠ {msg}")


def print_fail(msg):
    print(f"  ✗ {msg}")


def pip_install(package, extra_args=""):
    """Install a package via pip."""
    cmd = f"{sys.executable} -m pip install {package} {extra_args}"
    print(f"  Installing {package}...")
    stdout, stderr, code = run_cmd(cmd)
    if code == 0:
        print_ok(f"{package} installed successfully")
        return True
    else:
        print_fail(f"Failed to install {package}")
        if stderr:
            # Show just the last few lines of error
            error_lines = stderr.strip().split('\n')[-3:]
            for line in error_lines:
                print(f"    {line}")
        return False


def check_python():
    """Verify Python version."""
    print_header("Step 1: Python Version")
    
    ver = sys.version_info
    print(f"  Python {ver.major}.{ver.minor}.{ver.micro}")
    print(f"  Path: {sys.executable}")
    print(f"  Platform: {platform.system()} {platform.machine()}")

    if ver.major < 3 or (ver.major == 3 and ver.minor < 9):
        print_fail("Python 3.9+ required. Please upgrade.")
        return False

    print_ok(f"Python {ver.major}.{ver.minor} is supported")
    return True


def install_core_dependencies():
    """Install numpy, scipy, matplotlib, Pillow."""
    print_header("Step 2: Core Dependencies")

    packages = [
        ("numpy", "numpy>=1.24.0"),
        ("scipy", "scipy>=1.10.0"),
        ("matplotlib", "matplotlib>=3.7.0"),
        ("Pillow", "Pillow>=9.0.0"),
    ]

    all_ok = True
    for name, spec in packages:
        try:
            mod = __import__(name if name != "Pillow" else "PIL")
            version = getattr(mod, '__version__', 'unknown')
            print_ok(f"{name} {version} already installed")
        except ImportError:
            if not pip_install(spec):
                all_ok = False

    return all_ok


def detect_gpu():
    """Detect NVIDIA GPU and CUDA version."""
    print_header("Step 3: GPU Detection")

    gpu_info = {
        'has_nvidia': False,
        'gpu_name': None,
        'cuda_version': None,
        'cuda_major': None,
        'nvcc_available': False,
        'nvidia_smi_available': False,
    }

    # Check nvidia-smi
    nvidia_smi = shutil.which('nvidia-smi')
    if nvidia_smi:
        gpu_info['nvidia_smi_available'] = True
        stdout, _, code = run_cmd('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader')
        if code == 0 and stdout:
            gpu_info['has_nvidia'] = True
            parts = stdout.split(',')
            gpu_info['gpu_name'] = parts[0].strip() if parts else 'Unknown'
            print_ok(f"NVIDIA GPU detected: {gpu_info['gpu_name']}")
            if len(parts) > 1:
                print(f"    Driver: {parts[1].strip()}")
            if len(parts) > 2:
                print(f"    Memory: {parts[2].strip()}")

        # Get CUDA version from nvidia-smi
        stdout, _, code = run_cmd('nvidia-smi --query-gpu=driver_version --format=csv,noheader')
        
        # Better: parse CUDA version from nvidia-smi header
        stdout_full, _, _ = run_cmd('nvidia-smi')
        if 'CUDA Version' in stdout_full:
            for line in stdout_full.split('\n'):
                if 'CUDA Version' in line:
                    # Extract version number
                    import re
                    match = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', line)
                    if match:
                        major = int(match.group(1))
                        minor = int(match.group(2))
                        gpu_info['cuda_version'] = f"{major}.{minor}"
                        gpu_info['cuda_major'] = major
                        print_ok(f"CUDA Version: {gpu_info['cuda_version']}")
                    break
    else:
        print_warn("nvidia-smi not found")

    # Check nvcc
    nvcc = shutil.which('nvcc')
    if nvcc:
        gpu_info['nvcc_available'] = True
        stdout, _, code = run_cmd('nvcc --version')
        if code == 0:
            import re
            match = re.search(r'release (\d+)\.(\d+)', stdout)
            if match:
                major = int(match.group(1))
                if gpu_info['cuda_major'] is None:
                    gpu_info['cuda_version'] = f"{major}.{int(match.group(2))}"
                    gpu_info['cuda_major'] = major
                print_ok(f"NVCC compiler available (CUDA {match.group(1)}.{match.group(2)})")

    if not gpu_info['has_nvidia']:
        print_warn("No NVIDIA GPU detected. Toolkit will run on CPU (numpy fallback).")
        print(f"    This is fine for testing. GPU just makes it faster.")

    return gpu_info


def install_cupy(gpu_info):
    """Install the correct CuPy version for the detected CUDA."""
    print_header("Step 4: GPU Acceleration (CuPy)")

    if not gpu_info['has_nvidia']:
        print_warn("Skipping CuPy (no NVIDIA GPU detected)")
        print(f"    If you have a GPU but it wasn't detected, make sure")
        print(f"    NVIDIA drivers are installed and nvidia-smi works.")
        return False

    # Check if CuPy is already installed
    try:
        import cupy
        print_ok(f"CuPy {cupy.__version__} already installed")

        # Quick GPU test
        try:
            a = cupy.array([1, 2, 3])
            _ = cupy.sum(a)
            device = cupy.cuda.Device()
            print_ok(f"CuPy GPU test passed (Device: {device.id})")
            return True
        except Exception as e:
            print_warn(f"CuPy installed but GPU test failed: {e}")
            print(f"    Will attempt reinstall...")
    except ImportError:
        pass

    # Determine correct CuPy package
    cuda_major = gpu_info.get('cuda_major')

    if cuda_major is None:
        print_warn("Could not detect CUDA version. Trying cupy-cuda12x (most common)...")
        cupy_package = "cupy-cuda12x"
    elif cuda_major >= 12:
        cupy_package = "cupy-cuda12x"
    elif cuda_major == 11:
        cupy_package = "cupy-cuda11x"
    else:
        print_warn(f"CUDA {cuda_major} is very old. Trying cupy-cuda11x...")
        cupy_package = "cupy-cuda11x"

    print(f"  Selected package: {cupy_package}")
    print(f"  (This may take a few minutes on first install...)")

    if pip_install(cupy_package):
        # Verify
        try:
            import importlib
            import cupy
            importlib.reload(cupy)
            a = cupy.array([1, 2, 3])
            _ = cupy.sum(a)
            print_ok("CuPy GPU verification passed!")
            return True
        except Exception as e:
            print_warn(f"CuPy installed but verification failed: {e}")
            print(f"    The toolkit will fall back to CPU (numpy).")
            print(f"    Try: pip install {cupy_package} --force-reinstall")
            return False
    else:
        # Try alternative package
        alt_package = "cupy-cuda11x" if "12" in cupy_package else "cupy-cuda12x"
        print(f"  Trying alternative: {alt_package}...")
        if pip_install(alt_package):
            return True

        print_warn("CuPy installation failed. Toolkit will use CPU fallback.")
        print(f"    Manual install options:")
        print(f"      pip install cupy-cuda12x    (for CUDA 12+)")
        print(f"      pip install cupy-cuda11x    (for CUDA 11)")
        print(f"      pip install cupy            (builds from source, slow)")
        return False


def install_optional():
    """Offer to install optional dependencies."""
    print_header("Step 5: Optional Dependencies")

    print(f"  These are optional and not required for the toolkit to work:")
    print(f"    - ehtim:   Load real EHT .uvfits observation data")
    print(f"    - astropy: Load FITS image files")
    print()

    try:
        response = input("  Install optional packages? (y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        response = 'n'

    if response == 'y':
        pip_install("astropy>=5.0")
        pip_install("ehtim")
    else:
        print("  Skipped optional packages.")


def verify_installation():
    """Run a quick sanity check of the full pipeline."""
    print_header("Step 6: Verification")

    print("  Running quick pipeline test...")

    # Test core imports
    all_ok = True

    for name in ['numpy', 'scipy', 'matplotlib']:
        try:
            mod = __import__(name)
            ver = getattr(mod, '__version__', '?')
            print_ok(f"{name} {ver}")
        except ImportError:
            print_fail(f"{name} not found")
            all_ok = False

    # Test CuPy
    try:
        import cupy
        a = cupy.zeros(10)
        print_ok(f"CuPy {cupy.__version__} — GPU active")
    except ImportError:
        print_warn("CuPy not available — using CPU fallback (this is fine)")
    except Exception as e:
        print_warn(f"CuPy error: {e}")

    # Test toolkit imports
    try:
        # Ensure toolkit is importable
        toolkit_dir = os.path.dirname(os.path.abspath(__file__))
        if toolkit_dir not in sys.path:
            sys.path.insert(0, toolkit_dir)

        from core import Q4StokesEncoder, Q4PSSynchronizer, Base60Bridge, FBAIInterpolator
        from pipeline import EHTLoader
        from entropy import CoherenceGate
        from gpu.evpa_kernel import EVPAKernel, GPU_AVAILABLE

        print_ok(f"Toolkit imports OK — GPU={GPU_AVAILABLE}")
    except Exception as e:
        print_fail(f"Toolkit import failed: {e}")
        all_ok = False

    # Quick functional test
    try:
        import numpy as np
        from pipeline.eht_loader import EHTLoader as _EHTLoader
        from core.q4_encoder import Q4StokesEncoder as _Enc

        loader = _EHTLoader()
        data = loader.synthetic_blackhole(npix=32, seed=425)
        enc = _Enc()
        encoded = enc.encode_image(data['stokes'])
        decoded = enc.decode_image(encoded)
        nxc = float(np.corrcoef(data['stokes'][..., 0].flat, decoded[..., 0].flat)[0, 1])
        print_ok(f"Pipeline test: NxCorr={nxc:.4f} (should be >0.99)")
    except Exception as e:
        print_fail(f"Pipeline test failed: {e}")
        all_ok = False

    return all_ok


def main():
    print()
    print("  ∞ 0425 — QRFT Black Hole Toolkit Setup")
    print("  Rod's AI Consulting LLC")
    print("  ========================================")

    # Step 1: Python
    if not check_python():
        sys.exit(1)

    # Step 2: Core
    if not install_core_dependencies():
        print_fail("Core dependency installation failed. Cannot continue.")
        sys.exit(1)

    # Step 3: GPU Detection
    gpu_info = detect_gpu()

    # Step 4: CuPy
    cupy_ok = install_cupy(gpu_info)

    # Step 5: Optional
    install_optional()

    # Step 6: Verify
    all_ok = verify_installation()

    # Summary
    print_header("SETUP COMPLETE")

    print(f"""
  System:
    Python:     {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}
    Platform:   {platform.system()} {platform.machine()}
    GPU:        {gpu_info.get('gpu_name', 'None detected')}
    CUDA:       {gpu_info.get('cuda_version', 'N/A')}
    CuPy:       {'Installed' if cupy_ok else 'Not available (CPU fallback)'}

  Run the toolkit:
    python -m tests.test_synthetic          Phase 1+2 (encoder + coherence)
    python -m tests.test_phase3             Phase 3 (FBAI interpolation)
    python -m tests.test_phase4             Phase 4 (GPU + video)
    python -m tests.test_phase5_benchmark   Phase 5 (full benchmark)
    python -m viz.render_blackhole          Render images

  Images will be saved to: output/
""")

    if gpu_info['has_nvidia'] and not cupy_ok:
        print(f"  ⚠ GPU detected but CuPy failed to install.")
        print(f"    Try manually: pip install cupy-cuda12x")
        print(f"    The toolkit works fine on CPU — GPU just makes it faster.")
        print()

    if all_ok:
        print("  All checks passed. You're ready to go! ✓")
    else:
        print("  Some checks had warnings. The toolkit may still work.")
        print("  Run: python -m tests.test_synthetic  to verify.")

    print()
    print("  ∞ 0425")
    print()


if __name__ == '__main__':
    main()
