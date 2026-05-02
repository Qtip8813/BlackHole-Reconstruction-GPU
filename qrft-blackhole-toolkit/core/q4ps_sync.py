"""
Q4PS Synchronizer
==================
Implements the Quantum Four-Polarization Synchronization (Q4PS) Condition
for validating physical coherence in polarimetric image data.

Theory (from Q4PS White Paper, Section 2.0):
    The four Stokes parameters at each pixel are treated as four
    coupled quantum oscillators (analogous to the Kuramoto model).
    
    Phase angles phi_i are derived from each Stokes channel.
    The Coherence Order Parameter Z = R * exp(i*Psi) measures
    how synchronized the four channels are.
    
    R(t) >= R_c  →  Physically coherent (stable)
    R(t) <  R_c  →  Noise-dominated or unphysical (flag for FBAI)

    The Coherence Probability Function:
        P_coh = 0.5 * [1 + tanh(alpha * (R - R_c))]
    
    maps the continuous coherence state onto a probability,
    which serves as the Fractional Bit value in the FBAI system.

Created by: Rodney Lee Arnold Jr. ∞ 0425
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CoherenceMap:
    """
    Full coherence analysis result for an image.
    
    Attributes:
        R:          (H, W) Coherence radius, 0 <= R <= 1
        Psi:        (H, W) Average phase angle
        P_coh:      (H, W) Coherence probability (fractional bit value)
        mask_valid:  (H, W) Boolean mask: True where R >= R_c
        mask_flag:   (H, W) Boolean mask: True where pixel needs FBAI
        stats:       Summary statistics dict
    """
    R: np.ndarray
    Psi: np.ndarray
    P_coh: np.ndarray
    mask_valid: np.ndarray
    mask_flag: np.ndarray
    stats: Dict


class Q4PSSynchronizer:
    """
    Evaluates the Q4PS Synchronization Condition across a polarimetric
    image to determine which pixels are physically coherent.

    The synchronizer:
        1. Converts Stokes (I, Q, U, V) into phase angles
        2. Computes the Coherence Order Parameter Z per pixel
        3. Extracts the Coherence Radius R = |Z|
        4. Evaluates the Coherence Probability P_coh
        5. Produces validity masks for downstream processing

    Pixels that fail the coherence condition are candidates for
    FBAI re-prediction (gap filling / denoising).

    Usage:
        sync = Q4PSSynchronizer(R_c=0.6, alpha=10.0)
        
        # From raw Stokes array
        result = sync.analyze(stokes_array)
        
        # From encoded Q4 data
        result = sync.analyze_encoded(encoded_dict, encoder)
        
        # Check coherence at one pixel
        R, Psi, P = sync.pixel_coherence(I, Q, U, V)
    """

    def __init__(self, R_c: float = 0.6, alpha: float = 10.0,
                 K: float = 1.0, noise_sigma: float = 0.05):
        """
        Initialize the Q4PS Synchronizer.

        Args:
            R_c:   Critical coherence threshold. Pixels with R < R_c
                   are flagged as incoherent. Default 0.6 is calibrated
                   for typical EHT noise levels.
            alpha: Sensitivity constant for the tanh probability function.
                   Higher alpha = sharper transition at R_c.
            K:     Coupling strength constant (strong nuclear force analog).
                   Used in the phase dynamics simulation.
            noise_sigma: Standard deviation of quantum noise term xi_i(t).
        """
        if not 0.0 < R_c < 1.0:
            raise ValueError(f"R_c must be in (0, 1), got {R_c}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")

        self.R_c = R_c
        self.alpha = alpha
        self.K = K
        self.noise_sigma = noise_sigma

    # ------------------------------------------------------------------
    # Phase Mapping: Stokes → Phase Angles
    # ------------------------------------------------------------------

    def _stokes_to_phases(self, stokes: np.ndarray) -> np.ndarray:
        """
        Map Stokes parameters to phase angles on the unit circle.

        Each Stokes channel is normalized to [0, 1] and then mapped
        to a phase angle phi_i in [0, 2*pi).

        The mapping preserves the relative relationships:
            - I (total intensity) → reference phase
            - Q (horizontal/vertical) → 0 or pi offset from I
            - U (diagonal) → pi/4 offset
            - V (circular) → independent circular phase

        Args:
            stokes: Array of shape (..., 4) with [I, Q, U, V]

        Returns:
            Array of shape (..., 4) with phase angles in [0, 2*pi)
        """
        # Normalize each channel to [0, 1] independently
        flat = stokes.reshape(-1, 4).astype(np.float64)
        
        ch_min = flat.min(axis=0)
        ch_max = flat.max(axis=0)
        ch_range = np.where((ch_max - ch_min) == 0, 1.0, ch_max - ch_min)
        
        # Broadcast normalization
        normalized = (stokes - ch_min) / ch_range

        # Map to [0, 2*pi)
        phases = normalized * 2 * np.pi

        return phases

    # ------------------------------------------------------------------
    # Coherence Order Parameter
    # ------------------------------------------------------------------

    def _compute_order_parameter(self, phases: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the macroscopic Coherence Order Parameter Z.

        Z = (1/4) * sum_{j=1}^{4} exp(i * phi_j)
        R = |Z|     (coherence radius)
        Psi = arg(Z) (average phase)

        Args:
            phases: Array of shape (..., 4) with phase angles

        Returns:
            R:   Array of shape (...) with coherence radius [0, 1]
            Psi: Array of shape (...) with average phase [-pi, pi]
        """
        # Z = mean of unit vectors
        Z = np.mean(np.exp(1j * phases), axis=-1)
        
        R = np.abs(Z)
        Psi = np.angle(Z)

        return R, Psi

    # ------------------------------------------------------------------
    # Coherence Probability Function
    # ------------------------------------------------------------------

    def coherence_probability(self, R: np.ndarray) -> np.ndarray:
        """
        Compute the Coherence Probability P_coh.

        P_coh = 0.5 * [1 + tanh(alpha * (R - R_c))]

        This is the Fractional Bit value:
            P_coh ≈ 1  → Stable, coherent pixel
            P_coh ≈ 0  → Incoherent, needs FBAI reconstruction

        Args:
            R: Coherence radius array, values in [0, 1]

        Returns:
            P_coh array, values in [0, 1]
        """
        return 0.5 * (1.0 + np.tanh(self.alpha * (R - self.R_c)))

    # ------------------------------------------------------------------
    # Single Pixel Analysis
    # ------------------------------------------------------------------

    def pixel_coherence(self, I: float, Q: float, U: float,
                        V: float) -> Tuple[float, float, float]:
        """
        Compute coherence for a single pixel.

        Args:
            I, Q, U, V: Stokes parameter values

        Returns:
            (R, Psi, P_coh) tuple
        """
        stokes = np.array([[I, Q, U, V]])
        phases = self._stokes_to_phases(stokes)
        R, Psi = self._compute_order_parameter(phases)

        R_val = float(R[0])
        Psi_val = float(Psi[0])
        P_val = float(self.coherence_probability(R)[0])

        return R_val, Psi_val, P_val

    # ------------------------------------------------------------------
    # Full Image Analysis
    # ------------------------------------------------------------------

    def analyze(self, stokes_array: np.ndarray) -> CoherenceMap:
        """
        Run full Q4PS synchronization analysis on a Stokes image.

        Args:
            stokes_array: (H, W, 4) array with [I, Q, U, V] channels

        Returns:
            CoherenceMap with R, Psi, P_coh, masks, and stats
        """
        if stokes_array.ndim != 3 or stokes_array.shape[2] != 4:
            raise ValueError(
                f"Expected shape (H, W, 4), got {stokes_array.shape}")

        # Step 1: Map to phases
        phases = self._stokes_to_phases(stokes_array)

        # Step 2: Compute order parameter
        R, Psi = self._compute_order_parameter(phases)

        # Step 3: Coherence probability
        P_coh = self.coherence_probability(R)

        # Step 4: Masks
        mask_valid = R >= self.R_c
        mask_flag = ~mask_valid

        # Step 5: Summary statistics
        total_pixels = R.size
        n_valid = int(mask_valid.sum())
        n_flagged = int(mask_flag.sum())

        stats = {
            'total_pixels': total_pixels,
            'valid_pixels': n_valid,
            'flagged_pixels': n_flagged,
            'valid_fraction': n_valid / total_pixels,
            'flagged_fraction': n_flagged / total_pixels,
            'R_mean': float(R.mean()),
            'R_std': float(R.std()),
            'R_min': float(R.min()),
            'R_max': float(R.max()),
            'R_c': self.R_c,
            'P_coh_mean': float(P_coh.mean()),
            'alpha': self.alpha,
            'K': self.K
        }

        return CoherenceMap(
            R=R,
            Psi=Psi,
            P_coh=P_coh,
            mask_valid=mask_valid,
            mask_flag=mask_flag,
            stats=stats
        )

    def analyze_encoded(self, encoded: Dict, encoder) -> CoherenceMap:
        """
        Run Q4PS analysis on Q4-encoded data.

        Convenience method that decodes first, then analyzes.

        Args:
            encoded: Dict from Q4StokesEncoder.encode_image()
            encoder: The Q4StokesEncoder instance used to encode

        Returns:
            CoherenceMap
        """
        stokes = encoder.decode_image(encoded)
        return self.analyze(stokes)

    # ------------------------------------------------------------------
    # Phase Dynamics Simulation (Kuramoto)
    # ------------------------------------------------------------------

    def simulate_phase_evolution(self, stokes_pixel: np.ndarray,
                                 n_steps: int = 100,
                                 dt: float = 0.01) -> Dict:
        """
        Simulate the Kuramoto-like phase dynamics for a single pixel
        to observe how its coherence evolves over time.

        This implements the Q4PS phase equation:
            d(phi_i)/dt = omega_i + (K/N) * sum_j sin(phi_j - phi_i) + xi_i(t)

        Useful for understanding stability and predicting decay.

        Args:
            stokes_pixel: (4,) array with [I, Q, U, V]
            n_steps: Number of time steps to simulate
            dt: Time step size

        Returns:
            Dict with:
                'phases': (n_steps, 4) phase trajectory
                'R_trajectory': (n_steps,) coherence radius over time
                'P_trajectory': (n_steps,) coherence probability over time
                'converged': bool, whether system reached stable sync
        """
        N = 4

        # Initial phases from Stokes values
        phases_init = self._stokes_to_phases(
            stokes_pixel.reshape(1, 1, 4)
        ).flatten()

        # Natural frequencies (derived from Stokes magnitudes)
        omega = np.abs(stokes_pixel)
        omega = omega / (omega.max() + 1e-10) * np.pi  # Normalize to [0, pi]

        # Storage
        phases_traj = np.zeros((n_steps, N))
        R_traj = np.zeros(n_steps)
        P_traj = np.zeros(n_steps)

        phi = phases_init.copy()

        for t in range(n_steps):
            # Record
            phases_traj[t] = phi
            Z = np.mean(np.exp(1j * phi))
            R_traj[t] = np.abs(Z)
            P_traj[t] = float(self.coherence_probability(np.array([R_traj[t]]))[0])

            # Kuramoto coupling
            coupling = np.zeros(N)
            for i in range(N):
                for j in range(N):
                    coupling[i] += np.sin(phi[j] - phi[i])
                coupling[i] *= self.K / N

            # Noise
            noise = np.random.normal(0, self.noise_sigma, N)

            # Update
            dphi = omega + coupling + noise
            phi = (phi + dphi * dt) % (2 * np.pi)

        # Check convergence: R stayed above R_c for last 20% of steps
        tail = R_traj[int(0.8 * n_steps):]
        converged = bool(np.all(tail >= self.R_c))

        return {
            'phases': phases_traj,
            'R_trajectory': R_traj,
            'P_trajectory': P_traj,
            'converged': converged,
            'final_R': float(R_traj[-1]),
            'final_P': float(P_traj[-1])
        }

    # ------------------------------------------------------------------
    # Entropy Bridge
    # ------------------------------------------------------------------

    def coherence_entropy(self, R: np.ndarray) -> np.ndarray:
        """
        Compute the coherence-weighted entropy for the EVPA field.

        When R is high (ordered), entropy is low.
        When R is low (chaotic), entropy is high.

        S_coh = -R * log(R) - (1-R) * log(1-R)

        This is the binary entropy function applied to R, connecting
        the Q4PS synchronization state to the CSP entropy toolkit.

        Args:
            R: Coherence radius array, values in (0, 1)

        Returns:
            Entropy array, values in [0, log(2)]
        """
        # Clamp to avoid log(0)
        R_safe = np.clip(R, 1e-10, 1.0 - 1e-10)
        return -(R_safe * np.log(R_safe) + (1 - R_safe) * np.log(1 - R_safe))

    def __repr__(self) -> str:
        return (
            f"Q4PSSynchronizer(R_c={self.R_c}, alpha={self.alpha}, "
            f"K={self.K}, noise_sigma={self.noise_sigma})"
        )
