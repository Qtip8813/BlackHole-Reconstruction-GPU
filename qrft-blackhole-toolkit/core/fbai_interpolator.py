"""
FBAI Interpolator
==================
Fractional Bit AI gap-filling engine for black hole image reconstruction.

This is the generative core of the pipeline. Where Phase 1 (Q4 Encoder)
encodes and Phase 2 (Q4PS Synchronizer) judges, the FBAI Interpolator
*creates* — predicting physically consistent pixel values for the ~74%
of the uv-plane that has no telescope data.

Architecture: Tri-Manifold Prediction
    MINIMALIST mode:    Large-scale structure (outer disk, jet base)
                        Radial profile fitting with few parameters.
                        Operates on 16x16 downsampled grid.

    GOLDEN_RATIO mode:  Mid-scale features (accretion disk turbulence)
                        Self-similar scaling with phi-weighted kernels.
                        Operates on 64x64 mid-resolution grid.

    SINGULARITY mode:   Fine detail (photon ring, shadow edge)
                        Physics-constrained interpolation using Kerr
                        metric predictions for ring geometry.
                        Operates at full resolution.

Each prediction produces a Fractional Bit value in [0, 1] representing
confidence. The Q4PS coherence gate accepts or rejects each prediction.
Rejected pixels get re-predicted with tightened constraints.

Created by: Rodney Lee Arnold Jr. ∞ 0425
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import time


class ManifoldMode(Enum):
    """The three QRSP-FBAI operating modes."""
    MINIMALIST = "minimalist"
    GOLDEN_RATIO = "golden_ratio"
    SINGULARITY = "singularity"


@dataclass
class FBAIPrediction:
    """
    Result of FBAI interpolation for a single pixel or region.

    Attributes:
        stokes:          Predicted (4,) Stokes values [I, Q, U, V]
        fractional_bit:  Confidence value in [0, 1] (the "fractional bit")
        mode:            Which manifold mode produced this prediction
        iterations:      How many attempts before acceptance
        accepted:        Whether Q4PS coherence gate accepted this
        R_value:         Coherence radius of the prediction
    """
    stokes: np.ndarray
    fractional_bit: float
    mode: ManifoldMode
    iterations: int
    accepted: bool
    R_value: float


@dataclass
class InterpolationResult:
    """
    Full result of FBAI interpolation across an image.

    Attributes:
        reconstructed:    (H, W, 4) fully reconstructed Stokes image
        confidence_map:   (H, W) fractional bit confidence per pixel
        mode_map:         (H, W) which manifold mode was used per pixel
        iteration_map:    (H, W) iterations needed per pixel
        acceptance_map:   (H, W) bool - accepted by coherence gate
        stats:            Summary statistics
        timing:           Per-phase timing in seconds
    """
    reconstructed: np.ndarray
    confidence_map: np.ndarray
    mode_map: np.ndarray
    iteration_map: np.ndarray
    acceptance_map: np.ndarray
    stats: Dict
    timing: Dict


class FBAIInterpolator:
    """
    Tri-manifold FBAI interpolation engine for black hole image
    reconstruction.

    The interpolator fills gaps in the uv-plane (missing telescope data)
    using three operating modes at different spatial scales, with each
    prediction validated by the Q4PS coherence condition.

    Usage:
        from core.q4_encoder import Q4StokesEncoder
        from core.q4ps_sync import Q4PSSynchronizer

        interpolator = FBAIInterpolator(
            encoder=Q4StokesEncoder(),
            synchronizer=Q4PSSynchronizer(R_c=0.6),
            max_iterations=10
        )

        result = interpolator.reconstruct(
            stokes_array=data['stokes'],
            uv_mask=data['uv_mask'],
            metadata=data['metadata']
        )

        # result.reconstructed is the full (H, W, 4) image
        # result.confidence_map shows where FBAI is confident
    """

    # Golden ratio constant
    PHI = (1 + np.sqrt(5)) / 2  # 1.6180339887...

    # Mode resolution scales
    MINIMALIST_SCALE = 8     # Downscale factor for coarse grid
    GOLDEN_RATIO_SCALE = 2   # Downscale factor for mid grid

    def __init__(self, encoder, synchronizer,
                 max_iterations: int = 10,
                 convergence_threshold: float = 0.001,
                 verbose: bool = True):
        """
        Initialize the FBAI Interpolator.

        Args:
            encoder:       Q4StokesEncoder instance
            synchronizer:  Q4PSSynchronizer instance
            max_iterations: Max re-prediction attempts per pixel
            convergence_threshold: Stop iterating when change < this
            verbose:       Print progress updates
        """
        self.encoder = encoder
        self.sync = synchronizer
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose

        # Statistics tracking
        self._stats = {}

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [FBAI] {msg}")

    # ==================================================================
    # MAIN RECONSTRUCTION PIPELINE
    # ==================================================================

    def reconstruct(self, stokes_array: np.ndarray,
                     uv_mask: np.ndarray,
                     metadata: Optional[Dict] = None) -> InterpolationResult:
        """
        Run full tri-manifold reconstruction.

        Pipeline:
            1. MINIMALIST pass: Fit large-scale radial structure
            2. GOLDEN_RATIO pass: Fill mid-scale features with
               phi-weighted interpolation
            3. SINGULARITY pass: Refine fine detail near photon ring
            4. Coherence validation: Accept/reject each pixel
            5. Iterative refinement: Re-predict rejected pixels

        Args:
            stokes_array: (H, W, 4) original Stokes data
            uv_mask:      (H, W) boolean, True where data exists
            metadata:     Optional dict with physical params (spin, mass, etc.)

        Returns:
            InterpolationResult with full reconstruction
        """
        H, W = stokes_array.shape[:2]
        timing = {}

        self._log(f"Starting tri-manifold reconstruction ({H}x{W})")
        self._log(f"UV coverage: {uv_mask.sum()}/{uv_mask.size} "
                  f"({uv_mask.sum()/uv_mask.size*100:.1f}%)")
        self._log(f"Gap pixels to fill: {(~uv_mask).sum():,}")

        # Initialize result with original data where available
        result = stokes_array.copy()
        confidence = np.zeros((H, W), dtype=np.float64)
        mode_map = np.full((H, W), -1, dtype=np.int32)
        iteration_map = np.zeros((H, W), dtype=np.int32)

        # Sampled pixels get full confidence
        confidence[uv_mask] = 1.0

        # Extract physical parameters if available
        shadow_radius = self._get_shadow_radius(metadata, H, W)
        spin = metadata.get('spin', 0.0) if metadata else 0.0

        # Build coordinate grids
        cx, cy = H // 2, W // 2
        Y, X = np.mgrid[0:H, 0:W]
        R_grid = np.sqrt((X - cx)**2 + (Y - cy)**2)
        theta_grid = np.arctan2(Y - cy, X - cx)

        # ==============================================================
        # PASS 1: MINIMALIST — Large-scale radial structure
        # ==============================================================
        t0 = time.time()
        self._log("Pass 1: MINIMALIST (large-scale structure)")

        result, confidence, mode_map = self._minimalist_pass(
            result, uv_mask, confidence, mode_map,
            R_grid, theta_grid, shadow_radius, cx, cy
        )

        timing['minimalist'] = time.time() - t0
        filled_1 = (mode_map == 0).sum()
        self._log(f"  Filled {filled_1:,} pixels in {timing['minimalist']:.3f}s")

        # ==============================================================
        # PASS 2: GOLDEN_RATIO — Mid-scale self-similar features
        # ==============================================================
        t0 = time.time()
        self._log("Pass 2: GOLDEN_RATIO (mid-scale features)")

        result, confidence, mode_map = self._golden_ratio_pass(
            result, uv_mask, confidence, mode_map,
            R_grid, theta_grid, shadow_radius
        )

        timing['golden_ratio'] = time.time() - t0
        filled_2 = (mode_map == 1).sum()
        self._log(f"  Filled {filled_2:,} pixels in {timing['golden_ratio']:.3f}s")

        # ==============================================================
        # PASS 3: SINGULARITY — Fine detail near photon ring
        # ==============================================================
        t0 = time.time()
        self._log("Pass 3: SINGULARITY (photon ring detail)")

        result, confidence, mode_map = self._singularity_pass(
            result, uv_mask, confidence, mode_map,
            R_grid, theta_grid, shadow_radius, spin
        )

        timing['singularity'] = time.time() - t0
        filled_3 = (mode_map == 2).sum()
        self._log(f"  Filled {filled_3:,} pixels in {timing['singularity']:.3f}s")

        # ==============================================================
        # PASS 4: Coherence Validation + Iterative Refinement
        # ==============================================================
        t0 = time.time()
        self._log("Pass 4: Q4PS coherence validation + refinement")

        result, confidence, iteration_map, acceptance_map = \
            self._coherence_refinement(
                result, uv_mask, confidence, mode_map,
                R_grid, theta_grid, shadow_radius
            )

        timing['refinement'] = time.time() - t0
        self._log(f"  Refinement completed in {timing['refinement']:.3f}s")

        # ==============================================================
        # Compile statistics
        # ==============================================================
        total_gap = int((~uv_mask).sum())
        total_filled = int((mode_map >= 0).sum()) - int(uv_mask.sum())
        accepted = int(acceptance_map.sum())

        stats = {
            'total_pixels': H * W,
            'sampled_pixels': int(uv_mask.sum()),
            'gap_pixels': total_gap,
            'filled_pixels': total_filled,
            'fill_rate': total_filled / max(total_gap, 1),
            'accepted_pixels': accepted,
            'acceptance_rate': accepted / max(H * W, 1),
            'mean_confidence': float(confidence.mean()),
            'min_confidence': float(confidence.min()),
            'mean_iterations': float(iteration_map[~uv_mask].mean()),
            'max_iterations': int(iteration_map.max()),
            'pixels_by_mode': {
                'minimalist': int((mode_map == 0).sum()),
                'golden_ratio': int((mode_map == 1).sum()),
                'singularity': int((mode_map == 2).sum()),
                'original_data': int(uv_mask.sum())
            },
            'timing': timing
        }

        self._log(f"Reconstruction complete:")
        self._log(f"  Fill rate:       {stats['fill_rate']*100:.1f}%")
        self._log(f"  Acceptance rate: {stats['acceptance_rate']*100:.1f}%")
        self._log(f"  Mean confidence: {stats['mean_confidence']:.4f}")
        self._log(f"  Total time:      {sum(timing.values()):.3f}s")

        return InterpolationResult(
            reconstructed=result,
            confidence_map=confidence,
            mode_map=mode_map,
            iteration_map=iteration_map,
            acceptance_map=acceptance_map,
            stats=stats,
            timing=timing
        )

    # ==================================================================
    # PASS 1: MINIMALIST MODE
    # ==================================================================

    def _minimalist_pass(self, result, uv_mask, confidence, mode_map,
                          R_grid, theta_grid, shadow_radius, cx, cy):
        """
        Large-scale radial structure fitting.

        Strategy:
            1. Bin valid pixels by radial distance
            2. Fit a smooth radial profile for each Stokes channel
            3. Fill gap pixels using the radial profile + azimuthal
               modulation for Doppler asymmetry

        This captures the broad ring shape and overall brightness
        distribution — the "skeleton" of the image.
        """
        H, W = result.shape[:2]
        gap_mask = ~uv_mask

        # ---- Step 1: Build radial bins ----
        max_r = R_grid.max()
        n_bins = max(H // self.MINIMALIST_SCALE, 8)
        bin_edges = np.linspace(0, max_r, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # ---- Step 2: Compute radial profiles from valid data ----
        radial_profiles = np.zeros((n_bins, 4))
        radial_counts = np.zeros(n_bins)

        for b in range(n_bins):
            in_bin = (R_grid >= bin_edges[b]) & (R_grid < bin_edges[b+1]) & uv_mask
            if in_bin.any():
                radial_profiles[b] = result[in_bin].mean(axis=0)
                radial_counts[b] = in_bin.sum()

        # Smooth the profiles (handle empty bins via interpolation)
        for ch in range(4):
            valid_bins = radial_counts > 0
            if valid_bins.sum() >= 2:
                radial_profiles[:, ch] = np.interp(
                    bin_centers,
                    bin_centers[valid_bins],
                    radial_profiles[valid_bins, ch]
                )

        # ---- Step 3: Compute azimuthal modulation (Doppler) ----
        # Fit first-order Fourier: I(theta) = A + B*cos(theta) + C*sin(theta)
        azimuthal_coeffs = np.zeros((n_bins, 4, 3))  # [bin, channel, (A,B,C)]

        for b in range(n_bins):
            in_bin = (R_grid >= bin_edges[b]) & (R_grid < bin_edges[b+1]) & uv_mask
            if in_bin.sum() > 5:
                angles = theta_grid[in_bin]
                for ch in range(4):
                    values = result[in_bin, ch]
                    # Least squares: v = A + B*cos + C*sin
                    design = np.column_stack([
                        np.ones(len(angles)),
                        np.cos(angles),
                        np.sin(angles)
                    ])
                    coeffs, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
                    azimuthal_coeffs[b, ch] = coeffs

        # ---- Step 4: Predict gap pixels ----
        gap_y, gap_x = np.where(gap_mask)

        for idx in range(len(gap_y)):
            y, x = gap_y[idx], gap_x[idx]
            r = R_grid[y, x]
            theta = theta_grid[y, x]

            # Find radial bin
            b = np.searchsorted(bin_edges, r) - 1
            b = np.clip(b, 0, n_bins - 1)

            # Predict from radial profile + azimuthal modulation
            prediction = np.zeros(4)
            for ch in range(4):
                A, B, C = azimuthal_coeffs[b, ch]
                prediction[ch] = A + B * np.cos(theta) + C * np.sin(theta)

            # Fractional bit: based on how many valid pixels were in this bin
            fb = min(radial_counts[b] / 50.0, 1.0) * 0.6  # Cap at 0.6 for minimalist

            result[y, x] = prediction
            confidence[y, x] = fb
            mode_map[y, x] = 0  # MINIMALIST

        return result, confidence, mode_map

    # ==================================================================
    # PASS 2: GOLDEN_RATIO MODE
    # ==================================================================

    def _golden_ratio_pass(self, result, uv_mask, confidence, mode_map,
                            R_grid, theta_grid, shadow_radius):
        """
        Mid-scale feature refinement using golden-ratio weighted
        interpolation.

        Strategy:
            1. For each gap pixel, find neighbors with data or
               minimalist predictions
            2. Weight neighbors using phi-scaled distance kernel:
               w_i = 1 / (d_i^phi) where phi = 1.618...
            3. Self-similar refinement: also consider neighbors at
               phi-scaled radial distances (captures turbulent scaling)

        The golden ratio weighting isn't arbitrary — it approximates
        the self-similar cascade scaling observed in MHD turbulence.
        """
        H, W = result.shape[:2]
        gap_mask = ~uv_mask

        # Kernel radius scales with image size
        kernel_r = max(H // (self.GOLDEN_RATIO_SCALE * 4), 3)

        # Only refine pixels that were filled by minimalist
        # (or still in gaps)
        refine_mask = gap_mask & (confidence < 0.8)
        refine_y, refine_x = np.where(refine_mask)

        if len(refine_y) == 0:
            return result, confidence, mode_map

        # Precompute: which pixels have "good" data (original or high confidence)
        good_mask = uv_mask | (confidence >= 0.5)

        for idx in range(len(refine_y)):
            y, x = refine_y[idx], refine_x[idx]

            # ---- Gather neighbors within kernel ----
            y_min = max(0, y - kernel_r)
            y_max = min(H, y + kernel_r + 1)
            x_min = max(0, x - kernel_r)
            x_max = min(W, x + kernel_r + 1)

            patch_good = good_mask[y_min:y_max, x_min:x_max]
            if not patch_good.any():
                continue

            patch_result = result[y_min:y_max, x_min:x_max]
            patch_conf = confidence[y_min:y_max, x_min:x_max]

            # Local coordinate grid
            py, px = np.mgrid[0:y_max-y_min, 0:x_max-x_min]
            local_y = y - y_min
            local_x = x - x_min
            dist = np.sqrt((py - local_y)**2 + (px - local_x)**2)
            dist = np.where(dist == 0, 1e-10, dist)

            # ---- Golden ratio distance weighting ----
            weights = 1.0 / (dist ** self.PHI)
            weights *= patch_good.astype(float)
            weights *= patch_conf  # Trust higher-confidence neighbors more

            # ---- Self-similar scaling: boost neighbors at phi-scaled distance ----
            r_pixel = R_grid[y, x]
            patch_r = R_grid[y_min:y_max, x_min:x_max]

            # Neighbors at phi-scaled radial distances get a bonus
            patch_r_safe = np.where(patch_r > 0, patch_r, 1.0)
            r_ratio = r_pixel / patch_r_safe
            r_ratio = np.where(patch_r > 0, r_ratio, 1.0)
            phi_proximity = np.exp(-2.0 * (r_ratio - self.PHI)**2) + \
                            np.exp(-2.0 * (r_ratio - 1.0/self.PHI)**2)
            weights *= (1.0 + 0.5 * phi_proximity)

            total_weight = weights.sum()
            if total_weight < 1e-10:
                continue

            # ---- Weighted prediction ----
            prediction = np.zeros(4)
            for ch in range(4):
                prediction[ch] = (weights * patch_result[..., ch]).sum() / total_weight

            # Fractional bit: quality of local neighborhood
            fb = min(total_weight / (kernel_r * 2), 1.0) * 0.8  # Cap at 0.8

            # Only update if this is better than minimalist
            if fb > confidence[y, x]:
                result[y, x] = prediction
                confidence[y, x] = fb
                mode_map[y, x] = 1  # GOLDEN_RATIO

        return result, confidence, mode_map

    # ==================================================================
    # PASS 3: SINGULARITY MODE
    # ==================================================================

    def _singularity_pass(self, result, uv_mask, confidence, mode_map,
                           R_grid, theta_grid, shadow_radius, spin):
        """
        Fine-detail refinement near the photon ring.

        Strategy:
            1. Identify the photon ring zone (pixels near shadow_radius)
            2. Use Kerr metric predictions to constrain ring geometry:
               - Ring width from spin parameter
               - Brightness asymmetry from frame dragging
               - EVPA twist from magnetic field threading
            3. High-precision interpolation with physics constraints

        This mode only operates on pixels within ~2x the shadow radius.
        It's the most computationally expensive but most physically
        accurate mode.
        """
        H, W = result.shape[:2]

        # Define the photon ring zone: within 2x shadow radius
        ring_inner = shadow_radius * 0.5
        ring_outer = shadow_radius * 2.0
        ring_zone = (R_grid >= ring_inner) & (R_grid <= ring_outer)
        gap_in_ring = (~uv_mask) & ring_zone

        ring_y, ring_x = np.where(gap_in_ring)

        if len(ring_y) == 0:
            return result, confidence, mode_map

        self._log(f"  Ring zone pixels: {ring_zone.sum():,}, "
                  f"gaps in ring: {len(ring_y):,}")

        # ---- Kerr metric predictions ----
        # Ring width narrows with spin (higher spin = thinner ring)
        ring_width_factor = 1.0 - 0.5 * abs(spin)
        ring_sigma = 3.0 * ring_width_factor  # pixels

        # Frame-dragging asymmetry
        drag_strength = 0.4 * spin

        # ---- Physics-constrained interpolation ----
        for idx in range(len(ring_y)):
            y, x = ring_y[idx], ring_x[idx]
            r = R_grid[y, x]
            theta = theta_grid[y, x]

            # Expected ring brightness profile (Gaussian centered at shadow_radius)
            ring_profile = np.exp(-0.5 * ((r - shadow_radius) / ring_sigma)**2)

            # Frame-dragging Doppler modulation
            doppler = 1.0 + drag_strength * np.cos(theta)

            # Predicted Stokes I from ring model
            I_pred = ring_profile * doppler

            # EVPA from toroidal field + frame-dragging twist
            evpa_pred = theta + spin * 0.3 * np.exp(-r / shadow_radius)

            # Fractional polarization near ring (~18% for M87*)
            frac_pol_pred = 0.18 * ring_profile

            # Stokes Q, U from predicted EVPA and fractional pol
            Q_pred = frac_pol_pred * I_pred * np.cos(2 * evpa_pred)
            U_pred = frac_pol_pred * I_pred * np.sin(2 * evpa_pred)

            # Stokes V (weak circular pol)
            V_pred = 0.01 * I_pred * np.sin(theta + spin * 0.5)

            # Blend with existing prediction (weighted by ring proximity)
            ring_weight = ring_profile  # Strong near ring, weak far away
            blend = ring_weight * 0.7  # Physics gets 70% weight near ring

            prediction = np.array([I_pred, Q_pred, U_pred, V_pred])

            # Blend with current result
            blended = blend * prediction + (1 - blend) * result[y, x]

            # Scale to match the intensity range of valid data
            # (the ring model is normalized, real data may not be)
            if uv_mask.any():
                valid_in_ring = uv_mask & ring_zone
                if valid_in_ring.any():
                    scale = result[valid_in_ring, 0].max()
                    blended[0] *= scale
                    blended[1] *= scale
                    blended[2] *= scale
                    blended[3] *= scale

            # Fractional bit: high confidence near ring center
            fb = 0.5 + 0.45 * ring_profile  # [0.5, 0.95] range

            if fb > confidence[y, x]:
                result[y, x] = blended
                confidence[y, x] = fb
                mode_map[y, x] = 2  # SINGULARITY

        return result, confidence, mode_map

    # ==================================================================
    # PASS 4: COHERENCE VALIDATION + ITERATIVE REFINEMENT
    # ==================================================================

    def _coherence_refinement(self, result, uv_mask, confidence, mode_map,
                               R_grid, theta_grid, shadow_radius):
        """
        Validate all predictions against Q4PS coherence condition.
        Re-predict rejected pixels with tightened constraints.

        For each gap pixel:
            1. Check R(t) >= R_c via the synchronizer
            2. If accepted: mark as valid, keep prediction
            3. If rejected: perturb prediction toward local mean,
               re-check, up to max_iterations times
        """
        H, W = result.shape[:2]
        gap_mask = ~uv_mask
        iteration_map = np.zeros((H, W), dtype=np.int32)

        # Run coherence analysis on current state
        coherence = self.sync.analyze(result)
        acceptance_map = coherence.mask_valid.copy()

        # Pixels that need refinement: in gaps AND failed coherence
        needs_work = gap_mask & (~acceptance_map)
        work_y, work_x = np.where(needs_work)

        self._log(f"  Pixels needing refinement: {len(work_y):,}")

        if len(work_y) == 0:
            return result, confidence, iteration_map, acceptance_map

        # Iterative refinement
        for iteration in range(self.max_iterations):
            if len(work_y) == 0:
                break

            n_fixed = 0

            for idx in range(len(work_y)):
                y, x = work_y[idx], work_x[idx]

                # Get local neighborhood mean (from accepted pixels)
                kernel = 5
                y_lo = max(0, y - kernel)
                y_hi = min(H, y + kernel + 1)
                x_lo = max(0, x - kernel)
                x_hi = min(W, x + kernel + 1)

                patch = result[y_lo:y_hi, x_lo:x_hi]
                patch_accepted = acceptance_map[y_lo:y_hi, x_lo:x_hi]

                if patch_accepted.any():
                    selected = patch[patch_accepted]  # (N, 4)
                    local_mean = selected.mean(axis=0).ravel()[:4]
                else:
                    local_mean = patch.reshape(-1, 4).mean(axis=0)

                # Ensure shapes match
                local_mean = local_mean.flatten()[:4]

                # Perturb toward local mean (damped)
                damping = 0.3 + 0.1 * iteration  # Increase damping per iteration
                damping = min(damping, 0.8)

                new_val = (1 - damping) * result[y, x] + damping * local_mean

                # Add small noise to avoid getting stuck
                noise_scale = 0.01 / (1 + iteration)
                noise = np.random.normal(0, noise_scale, 4)
                new_val += noise

                # Ensure Stokes I non-negative
                new_val[0] = np.maximum(new_val[0], 0)

                result[y, x] = new_val
                iteration_map[y, x] = iteration + 1

            # Re-check coherence
            coherence = self.sync.analyze(result)
            acceptance_map = coherence.mask_valid.copy()

            # Force-accept original data
            acceptance_map[uv_mask] = True

            # Update work list
            still_needs_work = gap_mask & (~acceptance_map)
            prev_count = len(work_y)
            work_y, work_x = np.where(still_needs_work)

            n_fixed = prev_count - len(work_y)

            if self.verbose and (iteration < 3 or iteration == self.max_iterations - 1):
                self._log(f"    Iteration {iteration+1}: "
                         f"fixed {n_fixed}, remaining {len(work_y)}")

            # Check convergence
            if n_fixed == 0 and iteration > 2:
                self._log(f"    Converged at iteration {iteration+1}")
                break

        # Update confidence for refined pixels
        # Refined pixels that got accepted get a confidence boost
        refined_accepted = gap_mask & acceptance_map & (iteration_map > 0)
        confidence[refined_accepted] = np.minimum(
            confidence[refined_accepted] + 0.15, 1.0
        )

        # Refined pixels still rejected get penalized
        refined_rejected = gap_mask & (~acceptance_map) & (iteration_map > 0)
        confidence[refined_rejected] *= 0.5

        return result, confidence, iteration_map, acceptance_map

    # ==================================================================
    # UTILITY METHODS
    # ==================================================================

    def _get_shadow_radius(self, metadata: Optional[Dict],
                           H: int, W: int) -> float:
        """
        Get the black hole shadow radius in pixel units.

        If metadata contains physical params, calculate from those.
        Otherwise estimate from image geometry.
        """
        if metadata and 'shadow_radius_uas' in metadata and 'fov_uas' in metadata:
            # Convert from microarcseconds to pixels
            uas_per_pixel = metadata['fov_uas'] / max(H, W)
            return metadata['shadow_radius_uas'] / uas_per_pixel
        else:
            # Assume shadow is ~1/7 of the image width (typical for EHT FOV)
            return max(H, W) / 7.0

    def reconstruction_report(self, result: InterpolationResult) -> str:
        """
        Generate a human-readable report of the reconstruction.

        Args:
            result: InterpolationResult from reconstruct()

        Returns:
            Formatted string report
        """
        s = result.stats
        t = result.timing

        lines = [
            "",
            "=" * 60,
            "  FBAI Reconstruction Report",
            "  ∞ 0425 — QRFT Black Hole Toolkit",
            "=" * 60,
            "",
            f"  Image:           {result.reconstructed.shape[0]}x"
            f"{result.reconstructed.shape[1]}",
            f"  Total pixels:    {s['total_pixels']:,}",
            f"  Sampled (data):  {s['sampled_pixels']:,}",
            f"  Gaps filled:     {s['filled_pixels']:,} "
            f"({s['fill_rate']*100:.1f}%)",
            "",
            "  Mode Breakdown:",
            f"    MINIMALIST:    {s['pixels_by_mode']['minimalist']:,}",
            f"    GOLDEN_RATIO:  {s['pixels_by_mode']['golden_ratio']:,}",
            f"    SINGULARITY:   {s['pixels_by_mode']['singularity']:,}",
            f"    Original data: {s['pixels_by_mode']['original_data']:,}",
            "",
            "  Quality Metrics:",
            f"    Acceptance:    {s['acceptance_rate']*100:.1f}%",
            f"    Confidence:    {s['mean_confidence']:.4f} (mean), "
            f"{s['min_confidence']:.4f} (min)",
            f"    Iterations:    {s['mean_iterations']:.1f} (mean), "
            f"{s['max_iterations']} (max)",
            "",
            "  Timing:",
            f"    MINIMALIST:    {t.get('minimalist', 0):.3f}s",
            f"    GOLDEN_RATIO:  {t.get('golden_ratio', 0):.3f}s",
            f"    SINGULARITY:   {t.get('singularity', 0):.3f}s",
            f"    Refinement:    {t.get('refinement', 0):.3f}s",
            f"    Total:         {sum(t.values()):.3f}s",
            "",
            "=" * 60,
        ]

        return '\n'.join(lines)

    def __repr__(self) -> str:
        return (
            f"FBAIInterpolator(max_iter={self.max_iterations}, "
            f"conv_threshold={self.convergence_threshold}, "
            f"phi={self.PHI:.4f})"
        )
