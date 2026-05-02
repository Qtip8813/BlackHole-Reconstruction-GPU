"""
QRFT Phase 3: Noise Extraction & Removal
==========================================
Direct approach:
  1. Extract the NOISE from the image
  2. REMOVE the noise (subtraction)
  3. What remains = CLEAN SIGNAL
  4. Reconstruct from clean signal ONLY

"Find the noise, remove it, keep the signal"
∞ 0425 — Rod's AI Consulting LLC

Run:
    python noise_extraction_reconstruction.py
"""

import sys
import time
import numpy as np

# GPU imports
try:
    import cupy as cp
    from cupyx.scipy.ndimage import convolve, gaussian_filter, median_filter
    GPU_AVAILABLE = True
    print("✓ CuPy detected - GPU acceleration enabled")
except ImportError:
    import numpy as cp
    from scipy.ndimage import convolve, gaussian_filter, median_filter
    GPU_AVAILABLE = False
    print("⚠ CuPy not found - using CPU fallback")


# ============================================================
# NOISE EXTRACTOR
# ============================================================

class NoiseExtractor:
    """
    Extract noise from black hole image

    Strategy:
      1. Smooth the image (signal is smooth, noise is sharp)
      2. Residual = Original - Smooth = NOISE
      3. Identify high-residual pixels as noise
      4. Remove them completely
    """

    def __init__(
        self,
        noise_percentile=85.0,
        smooth_sigma=2.0,
        verbose=True
    ):
        self.noise_percentile = noise_percentile
        self.smooth_sigma = smooth_sigma
        self.verbose = verbose
        self.gpu_enabled = GPU_AVAILABLE

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  NOISE EXTRACTOR")
            print(f"  'Find it, extract it, remove it'")
            print(f"  ∞ 0425")
            print(f"{'='*60}")
            print(f"  Noise percentile: {noise_percentile:.0f}th")
            print(f"  Smoothing sigma:  {smooth_sigma:.1f}")

    def extract_noise(self, stokes, uv_mask):
        """
        Extract noise from image

        Returns:
            noise_map: [H, W, 4] - the NOISE extracted
            noise_mask: [H, W] - True = noise pixel, False = clean
            clean_stokes: [H, W, 4] - signal with noise REMOVED
        """
        H, W = stokes.shape[:2]

        # Move to GPU
        stokes_gpu = cp.array(stokes, dtype=cp.float32)

        if self.verbose:
            print(f"\n  Extracting noise from {H}x{W} image...")

        # Step 1: Smooth the image (preserves signal, removes sharp noise)
        smooth = cp.zeros_like(stokes_gpu)
        for ch in range(4):
            smooth[..., ch] = gaussian_filter(stokes_gpu[..., ch], sigma=self.smooth_sigma)

        # Step 2: Residual = Original - Smooth = NOISE
        residual = stokes_gpu - smooth

        # Step 3: Compute residual magnitude (per pixel)
        residual_mag = cp.sqrt(cp.sum(residual ** 2, axis=-1))

        # Step 4: Identify noise pixels (high residual)
        # Use percentile-based threshold on absolute residual
        # Top X% of residuals = noise

        uv_gpu = cp.array(uv_mask, dtype=cp.bool_)
        uv_residuals = residual_mag[uv_gpu]

        if len(uv_residuals) > 0:
            # Find threshold: pixels above Xth percentile = noise
            threshold_val = float(cp.percentile(uv_residuals, self.noise_percentile))

            if self.verbose:
                print(f"  Residual threshold ({self.noise_percentile:.0f}th percentile): {threshold_val:.4f}")

            # Noise mask: high absolute residual in UV-sampled region
            noise_mask_gpu = (residual_mag > threshold_val) & uv_gpu
        else:
            noise_mask_gpu = cp.zeros((H, W), dtype=cp.bool_)

        # Step 5: Extract the noise
        noise_map_gpu = cp.where(noise_mask_gpu[..., None], residual, 0.0)

        # Step 6: Remove noise from original
        clean_stokes_gpu = stokes_gpu - noise_map_gpu

        # Clip intensity to non-negative
        clean_stokes_gpu[..., 0] = cp.clip(clean_stokes_gpu[..., 0], 0.0, None)

        # Copy back to CPU
        if GPU_AVAILABLE:
            noise_map = cp.asnumpy(noise_map_gpu)
            noise_mask = cp.asnumpy(noise_mask_gpu)
            clean_stokes = cp.asnumpy(clean_stokes_gpu)
        else:
            noise_map = np.array(noise_map_gpu)
            noise_mask = np.array(noise_mask_gpu)
            clean_stokes = np.array(clean_stokes_gpu)

        if self.verbose:
            noise_count = noise_mask.sum()
            uv_count = uv_mask.sum()
            print(f"  Noise pixels detected: {noise_count:,} / {uv_count:,} UV sampled")
            print(f"  Noise fraction:        {noise_count/uv_count*100:.1f}%")
            print(f"  Clean pixels:          {uv_count - noise_count:,} ({(uv_count-noise_count)/uv_count*100:.1f}%)")

        return noise_map, noise_mask, clean_stokes


# ============================================================
# CLEAN SIGNAL RECONSTRUCTOR
# ============================================================

class CleanSignalReconstructor:
    """
    Reconstruct from CLEAN signal only (noise removed)

    Input: Image with noise EXTRACTED and REMOVED
    Process: Fill gaps using ONLY clean pixels
    Output: Reconstructed image from pure signal
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.gpu_enabled = GPU_AVAILABLE

    def reconstruct(
        self,
        clean_stokes: np.ndarray,
        clean_mask: np.ndarray,
        steps: int = 15
    ):
        """
        Reconstruct from clean signal

        Args:
            clean_stokes: [H, W, 4] - signal with noise removed
            clean_mask: [H, W] - True = clean pixel, False = gap or noise
            steps: iteration count

        Returns:
            reconstructed: [H, W, 4]
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  CLEAN SIGNAL RECONSTRUCTION")
            print(f"{'='*60}")
            clean_count = clean_mask.sum()
            total = clean_mask.size
            print(f"  Clean pixels: {clean_count:,} ({clean_count/total*100:.1f}%)")
            print(f"  Iterations:   {steps}")

        t0 = time.time()

        # Move to GPU
        state = cp.array(clean_stokes, dtype=cp.float32)
        clean_gpu = cp.array(clean_mask, dtype=cp.bool_)

        # Neighbor kernel (trust only clean neighbors)
        kernel = cp.array([
            [0.6/8, 0.6/8, 0.6/8],
            [0.6/8,   0.0, 0.6/8],
            [0.6/8, 0.6/8, 0.6/8]
        ], dtype=cp.float32)

        # Iterative fill
        for step in range(steps):
            # Compute neighbor field from CLEAN pixels only
            neighbor = cp.zeros_like(state)
            for ch in range(4):
                # Mask out non-clean pixels before convolution
                clean_channel = cp.where(clean_gpu, state[..., ch], 0.0)
                neighbor[..., ch] = convolve(clean_channel, kernel, mode='reflect')

            # Global mean (clean pixels only)
            clean_states = state[clean_gpu]
            if len(clean_states) > 0:
                global_mean = cp.mean(clean_states, axis=0)
            else:
                global_mean = cp.zeros(4, dtype=cp.float32)

            # Blend
            shared = 0.85 * neighbor + 0.15 * global_mean

            # Update gaps only (preserve clean pixels exactly)
            gap = ~clean_gpu
            lr = 0.20

            new_val = cp.tanh((1 - lr) * state + lr * shared)
            state = cp.where(gap[..., None], new_val, state)

            if self.verbose and (step + 1) % 5 == 0:
                # Compute gap coherence
                gap_states = state[gap]
                if len(gap_states) > 0:
                    gap_mean = cp.mean(gap_states, axis=0)
                    sims = cp.sum(gap_states * gap_mean, axis=1) / (
                        cp.linalg.norm(gap_states, axis=1) * cp.linalg.norm(gap_mean) + 1e-12
                    )
                    coh = float(cp.mean(sims))
                else:
                    coh = 1.0
                print(f"  Step {step+1}/{steps}: Gap coherence={coh:.4f}")

        # Copy back
        if GPU_AVAILABLE:
            recon = cp.asnumpy(state)
        else:
            recon = np.array(state)

        dt = time.time() - t0

        if self.verbose:
            print(f"\n  Reconstruction complete in {dt:.3f}s")
            print(f"{'='*60}\n")

        return recon, dt


# ============================================================
# FULL PIPELINE
# ============================================================

def noise_extraction_pipeline(
    stokes_obs: np.ndarray,
    uv_mask: np.ndarray,
    noise_percentile: float = 85.0,
    smooth_sigma: float = 2.0,
    recon_steps: int = 15
):
    """
    Complete noise extraction → clean reconstruction pipeline

    Phase 1: Extract noise (top X% of residuals)
    Phase 2: Remove noise
    Phase 3: Reconstruct from clean signal
    """

    print("\n" + "="*70)
    print("  NOISE EXTRACTION & REMOVAL PIPELINE")
    print("  ∞ 0425 — Rod's AI Consulting LLC")
    print("="*70)

    # Phase 1: Extract noise
    extractor = NoiseExtractor(
        noise_percentile=noise_percentile,
        smooth_sigma=smooth_sigma,
        verbose=True
    )

    noise_map, noise_mask, clean_stokes = extractor.extract_noise(stokes_obs, uv_mask)

    # Clean mask = UV sampled AND not noisy
    clean_mask = uv_mask & ~noise_mask

    # Phase 2: Reconstruct from clean signal
    reconstructor = CleanSignalReconstructor(verbose=True)
    recon, recon_time = reconstructor.reconstruct(clean_stokes, clean_mask, steps=recon_steps)

    return {
        'reconstructed': recon,
        'noise_map': noise_map,
        'noise_mask': noise_mask,
        'clean_stokes': clean_stokes,
        'clean_mask': clean_mask,
        'time_s': recon_time
    }


# ============================================================
# TEST
# ============================================================

def generate_test_data(npix=128, uv_coverage=0.4, noise_level=0.10, seed=425):
    """Generate synthetic black hole with noise"""
    np.random.seed(seed)

    cy, cx = npix // 2, npix // 2
    Y, X = np.mgrid[0:npix, 0:npix]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    Theta = np.arctan2(Y - cy, X - cx)

    # Photon ring
    ring_r = 35.0
    ring_w = 8.0
    I = np.exp(-0.5 * ((R - ring_r) / ring_w)**2)

    # Polarization
    Q = 0.3 * I * np.cos(2 * Theta)
    U = 0.3 * I * np.sin(2 * Theta)
    V = 0.05 * I

    stokes_clean = np.stack([I, Q, U, V], axis=-1).astype(np.float32)

    # UV coverage
    uv_mask = np.random.random((npix, npix)) < uv_coverage

    # Add noise to observed regions
    stokes_noisy = stokes_clean.copy()
    noise = np.random.normal(0, noise_level, stokes_clean.shape).astype(np.float32)
    stokes_noisy[uv_mask] += noise[uv_mask]

    # Zero out unobserved
    stokes_obs = stokes_noisy.copy()
    stokes_obs[~uv_mask] = 0.0

    return stokes_clean, stokes_obs, uv_mask


def test_noise_extraction():
    """Test noise extraction pipeline"""

    # Generate data
    print("\n[Data] Generating synthetic black hole with noise...")
    stokes_gt, stokes_obs, uv_mask = generate_test_data(
        npix=128, uv_coverage=0.4, noise_level=0.10, seed=425
    )

    print(f"  Image: {stokes_gt.shape}")
    print(f"  UV coverage: {uv_mask.sum()/uv_mask.size*100:.1f}%")
    print(f"  Noise level: 10%")

    # Run pipeline
    result = noise_extraction_pipeline(
        stokes_obs, uv_mask,
        noise_percentile=85.0,  # Top 15% = noise
        smooth_sigma=2.0,
        recon_steps=15
    )

    # Validation
    print("\n" + "="*70)
    print("  VALIDATION")
    print("="*70)

    recon = result['reconstructed']

    def nxcorr(a, b):
        return np.corrcoef(a.flatten(), b.flatten())[0, 1]

    nc_I = nxcorr(stokes_gt[..., 0], recon[..., 0])
    nc_Q = nxcorr(stokes_gt[..., 1], recon[..., 1])

    rmse = np.sqrt(np.mean((stokes_gt - recon)**2))

    print(f"\n  Reconstruction Quality:")
    print(f"    NxCorr (Stokes I):   {nc_I:.4f}")
    print(f"    NxCorr (Stokes Q):   {nc_Q:.4f}")
    print(f"    RMSE:                {rmse:.4f}")
    print(f"    Time:                {result['time_s']:.3f}s")

    print(f"\n  Noise Statistics:")
    print(f"    Noise pixels:        {result['noise_mask'].sum():,}")
    print(f"    Clean pixels:        {result['clean_mask'].sum():,}")
    print(f"    Removed fraction:    {result['noise_mask'].sum()/uv_mask.sum()*100:.1f}%")

    # Visualization
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))

        # Row 1: Stokes I
        axes[0,0].imshow(stokes_gt[..., 0], cmap='hot', origin='lower')
        axes[0,0].set_title('Ground Truth\nStokes I', fontweight='bold')
        axes[0,0].axis('off')

        axes[0,1].imshow(stokes_obs[..., 0], cmap='hot', origin='lower')
        axes[0,1].set_title('Observed\n(40% UV + 10% Noise)', fontweight='bold')
        axes[0,1].axis('off')

        axes[0,2].imshow(result['clean_stokes'][..., 0], cmap='hot', origin='lower')
        axes[0,2].set_title('After Noise Removal\n(Clean Signal)', fontweight='bold')
        axes[0,2].axis('off')

        axes[0,3].imshow(recon[..., 0], cmap='hot', origin='lower')
        axes[0,3].set_title(f'Reconstructed\nNxCorr={nc_I:.3f}', fontweight='bold')
        axes[0,3].axis('off')

        # Row 2: Masks and noise
        axes[1,0].imshow(uv_mask, cmap='gray', origin='lower')
        axes[1,0].set_title(f'UV Mask\n{uv_mask.sum()/uv_mask.size*100:.1f}%', fontweight='bold')
        axes[1,0].axis('off')

        axes[1,1].imshow(result['noise_mask'], cmap='Reds', origin='lower')
        axes[1,1].set_title(f'Noise Mask\n{result["noise_mask"].sum():,} pixels', fontweight='bold')
        axes[1,1].axis('off')

        axes[1,2].imshow(result['clean_mask'], cmap='Greens', origin='lower')
        axes[1,2].set_title(f'Clean Mask\n{result["clean_mask"].sum():,} pixels', fontweight='bold')
        axes[1,2].axis('off')

        noise_mag = np.sqrt(np.sum(result['noise_map']**2, axis=-1))
        axes[1,3].imshow(noise_mag, cmap='plasma', origin='lower')
        axes[1,3].set_title('Extracted Noise\n(Magnitude)', fontweight='bold')
        axes[1,3].axis('off')

        fig.suptitle(
            'Noise Extraction & Removal Pipeline\n∞ 0425 — "Find it, extract it, remove it"',
            fontsize=14, fontweight='bold', y=0.98
        )

        plt.tight_layout()
        plt.savefig('noise_extraction_result.png', dpi=150, bbox_inches='tight')
        print(f"\n  ✓ Saved: noise_extraction_result.png")
        plt.close()

    except ImportError:
        print("\n  ⚠ matplotlib not available - skipping visualization")

    print("\n" + "="*70)
    print("  NOISE EXTRACTION PIPELINE: COMPLETE")
    print("="*70)

    return nc_I > 0.7


if __name__ == '__main__':
    success = test_noise_extraction()
    sys.exit(0 if success else 1)
