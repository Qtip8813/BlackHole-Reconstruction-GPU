"""
GPU EVPA Kernel
================
CuPy-accelerated Electric Vector Position Angle, fractional polarization,
and linear polarization intensity calculations.

Computes all three polarization products in a single GPU pass to minimize
host↔device memory transfers.

Falls back to numpy automatically if CuPy is not available.

Created by: Rodney Lee Arnold Jr. ∞ 0425
"""

import numpy as np
from typing import Dict, Tuple, Optional

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False


def get_array_module(arr):
    """Return cp if array is on GPU, np otherwise."""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp
    return np


def to_gpu(arr: np.ndarray) -> 'cp.ndarray':
    """Transfer numpy array to GPU. Returns numpy array if no GPU."""
    if GPU_AVAILABLE:
        return cp.asarray(arr)
    return arr


def to_cpu(arr) -> np.ndarray:
    """Transfer array back to CPU. No-op if already numpy."""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


class EVPAKernel:
    """
    GPU-accelerated polarization product calculations.

    Computes EVPA, fractional polarization, and linear polarization
    intensity from Stokes Q and U buffers using CUDA parallel math.

    On RTX 5070 Ti (8,960 CUDA cores):
        128x128 image: ~16K operations → single warp dispatch
        512x512 image: ~262K operations → still single-pass

    Usage:
        kernel = EVPAKernel(use_gpu=True)

        # From raw Stokes arrays
        results = kernel.compute_all(I, Q, U, V)

        # From Q4 encoded data
        results = kernel.compute_from_encoded(encoded, encoder)

        # Just EVPA
        evpa = kernel.evpa(Q, U)
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the EVPA Kernel.

        Args:
            use_gpu: Attempt to use GPU. Falls back to CPU if unavailable.
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

        if self.use_gpu:
            # Warm up GPU with a small operation
            _ = cp.zeros(1)
            device = cp.cuda.Device()
            self._device_name = device.attributes.get('DeviceName', 'Unknown')
            self._compute_capability = device.compute_capability
            print(f"  [EVPA Kernel] GPU active: {self._device_name}")
            print(f"  [EVPA Kernel] Compute capability: {self._compute_capability}")
        else:
            self._device_name = "CPU (numpy fallback)"
            self._compute_capability = "N/A"
            if use_gpu:
                print("  [EVPA Kernel] CuPy not found, using numpy CPU fallback")

    # ------------------------------------------------------------------
    # Core Polarization Products
    # ------------------------------------------------------------------

    def evpa(self, Q: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Calculate Electric Vector Position Angle.

        χ = ½ arctan2(U, Q)

        This is the "swirl direction" — the orientation of the
        magnetic field projected onto the sky plane.

        Args:
            Q: Stokes Q array (any shape)
            U: Stokes U array (same shape as Q)

        Returns:
            EVPA array in radians [-π/2, π/2]
        """
        xp = self.xp
        Q_dev = xp.asarray(Q) if self.use_gpu else Q
        U_dev = xp.asarray(U) if self.use_gpu else U

        result = 0.5 * xp.arctan2(U_dev, Q_dev)

        return to_cpu(result)

    def linear_polarization(self, Q: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Calculate linear polarization intensity.

        LP = sqrt(Q² + U²)

        This is the "strength" of the swirl at each pixel.

        Args:
            Q, U: Stokes parameter arrays

        Returns:
            Linear polarization intensity array
        """
        xp = self.xp
        Q_dev = xp.asarray(Q) if self.use_gpu else Q
        U_dev = xp.asarray(U) if self.use_gpu else U

        result = xp.sqrt(Q_dev**2 + U_dev**2)

        return to_cpu(result)

    def fractional_polarization(self, I: np.ndarray, Q: np.ndarray,
                                  U: np.ndarray) -> np.ndarray:
        """
        Calculate fractional polarization.

        m = sqrt(Q² + U²) / I

        Ranges from 0 (unpolarized) to 1 (fully polarized).
        For M87*, observed ~15-20% near the photon ring.

        Args:
            I, Q, U: Stokes parameter arrays

        Returns:
            Fractional polarization array, clipped to [0, 1]
        """
        xp = self.xp
        I_dev = xp.asarray(I) if self.use_gpu else I
        Q_dev = xp.asarray(Q) if self.use_gpu else Q
        U_dev = xp.asarray(U) if self.use_gpu else U

        # Avoid division by zero
        I_safe = xp.where(I_dev > 0, I_dev, xp.float64(1e-10))
        lp = xp.sqrt(Q_dev**2 + U_dev**2)
        m = lp / I_safe

        result = xp.clip(m, 0.0, 1.0)

        return to_cpu(result)

    def compute_all(self, I: np.ndarray, Q: np.ndarray,
                     U: np.ndarray, V: Optional[np.ndarray] = None) -> Dict:
        """
        Compute all polarization products in a SINGLE GPU pass.

        Transfers data to GPU once, computes everything, transfers back.
        This is the most efficient way to use the GPU.

        Args:
            I, Q, U: Stokes parameter arrays (required)
            V: Stokes V array (optional, for circular polarization)

        Returns:
            Dict with:
                'evpa':       EVPA angle array (radians)
                'lp':         Linear polarization intensity
                'frac_pol':   Fractional polarization [0, 1]
                'total_pol':  Total polarization (including V if given)
                'stokes_P':   Complex polarization P = Q + iU
        """
        xp = self.xp

        # Single transfer to GPU
        I_dev = xp.asarray(I) if self.use_gpu else I.astype(np.float64)
        Q_dev = xp.asarray(Q) if self.use_gpu else Q.astype(np.float64)
        U_dev = xp.asarray(U) if self.use_gpu else U.astype(np.float64)

        # All computations on device
        evpa = 0.5 * xp.arctan2(U_dev, Q_dev)
        lp = xp.sqrt(Q_dev**2 + U_dev**2)

        I_safe = xp.where(I_dev > 0, I_dev, xp.float64(1e-10))
        frac_pol = xp.clip(lp / I_safe, 0.0, 1.0)

        # Complex polarization
        stokes_P = Q_dev + 1j * U_dev

        # Total polarization (including circular if available)
        if V is not None:
            V_dev = xp.asarray(V) if self.use_gpu else V.astype(np.float64)
            total_pol = xp.sqrt(Q_dev**2 + U_dev**2 + V_dev**2)
        else:
            total_pol = lp

        # Single transfer back to CPU
        return {
            'evpa': to_cpu(evpa),
            'lp': to_cpu(lp),
            'frac_pol': to_cpu(frac_pol),
            'total_pol': to_cpu(total_pol),
            'stokes_P': to_cpu(stokes_P),
        }

    def compute_from_stokes(self, stokes_array: np.ndarray) -> Dict:
        """
        Compute all polarization products from a (H, W, 4) Stokes array.

        Convenience wrapper around compute_all().

        Args:
            stokes_array: (H, W, 4) array with [I, Q, U, V]

        Returns:
            Dict with all polarization products
        """
        I = stokes_array[..., 0]
        Q = stokes_array[..., 1]
        U = stokes_array[..., 2]
        V = stokes_array[..., 3]

        return self.compute_all(I, Q, U, V)

    # ------------------------------------------------------------------
    # EVPA Vector Field (for visualization)
    # ------------------------------------------------------------------

    def evpa_vectors(self, Q: np.ndarray, U: np.ndarray,
                      step: int = 4, min_lp: float = 0.01) -> Dict:
        """
        Compute EVPA tick mark positions and directions for plotting.

        Returns vectors suitable for matplotlib quiver plots.

        Args:
            Q, U: Stokes parameter arrays (H, W)
            step: Pixel spacing between tick marks
            min_lp: Minimum linear polarization to show a tick

        Returns:
            Dict with 'x', 'y', 'dx', 'dy', 'magnitude' arrays
        """
        xp = self.xp
        Q_dev = xp.asarray(Q) if self.use_gpu else Q
        U_dev = xp.asarray(U) if self.use_gpu else U

        evpa = 0.5 * xp.arctan2(U_dev, Q_dev)
        lp = xp.sqrt(Q_dev**2 + U_dev**2)

        evpa_cpu = to_cpu(evpa)
        lp_cpu = to_cpu(lp)

        H, W = Q.shape
        xs, ys, dxs, dys, mags = [], [], [], [], []

        for yi in range(0, H, step):
            for xi in range(0, W, step):
                if lp_cpu[yi, xi] > min_lp:
                    angle = evpa_cpu[yi, xi]
                    mag = lp_cpu[yi, xi]
                    xs.append(xi)
                    ys.append(yi)
                    dxs.append(np.cos(angle) * mag)
                    dys.append(np.sin(angle) * mag)
                    mags.append(mag)

        return {
            'x': np.array(xs),
            'y': np.array(ys),
            'dx': np.array(dxs),
            'dy': np.array(dys),
            'magnitude': np.array(mags)
        }

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def device_info(self) -> Dict:
        """Return GPU device information."""
        info = {
            'gpu_available': GPU_AVAILABLE,
            'using_gpu': self.use_gpu,
            'device': self._device_name,
            'compute_capability': self._compute_capability,
        }

        if self.use_gpu:
            mem = cp.cuda.Device().mem_info
            info['free_memory_gb'] = mem[0] / 1e9
            info['total_memory_gb'] = mem[1] / 1e9

        return info

    def __repr__(self) -> str:
        return f"EVPAKernel(device='{self._device_name}', gpu={self.use_gpu})"
