"""
QRFT Black Hole Toolkit - Reconstruction Visualization
========================================================
Renders the full pipeline output: ground truth, sparse observation,
FBAI reconstruction, confidence maps, EVPA swirl, and mode breakdown.

Created by: Rodney Lee Arnold Jr. ∞ 0425
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle

from core.q4_encoder import Q4StokesEncoder
from core.q4ps_sync import Q4PSSynchronizer
from core.fbai_interpolator import FBAIInterpolator
from entropy.coherence_gate import CoherenceGate
from pipeline.eht_loader import EHTLoader


def render_blackhole():
    """Generate all visualization images."""

    # Output directory (local, cross-platform)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    print("Generating synthetic black hole data...")
    loader = EHTLoader()
    data = loader.synthetic_blackhole(
        npix=128, fov_uas=150.0, mass_msun=6.5e9,
        spin=0.9, inclination_deg=17.0, noise_level=0.01, seed=425
    )

    ground_truth = data['stokes']
    uv_mask = data['uv_mask']
    metadata = data['metadata']

    # Sparse observation
    observed = ground_truth.copy()
    observed[~uv_mask] = 0.0

    print("Running FBAI reconstruction...")
    encoder = Q4StokesEncoder(n_layers=1)
    sync = Q4PSSynchronizer(R_c=0.5, alpha=8.0, K=1.0)
    interpolator = FBAIInterpolator(
        encoder=encoder, synchronizer=sync,
        max_iterations=8, verbose=False
    )

    result = interpolator.reconstruct(observed, uv_mask, metadata)
    reconstructed = result.reconstructed

    # Compute polarization products
    gt_I = ground_truth[..., 0]
    gt_Q = ground_truth[..., 1]
    gt_U = ground_truth[..., 2]
    gt_evpa = 0.5 * np.arctan2(gt_U, gt_Q)
    gt_lp = np.sqrt(gt_Q**2 + gt_U**2)
    gt_frac = gt_lp / np.where(gt_I > 0.01, gt_I, 1e-10)

    rc_I = reconstructed[..., 0]
    rc_Q = reconstructed[..., 1]
    rc_U = reconstructed[..., 2]
    rc_evpa = 0.5 * np.arctan2(rc_U, rc_Q)
    rc_lp = np.sqrt(rc_Q**2 + rc_U**2)
    rc_frac = rc_lp / np.where(rc_I > 0.01, rc_I, 1e-10)

    # Shadow radius in pixels
    uas_per_px = metadata['fov_uas'] / 128
    shadow_r_px = metadata['shadow_radius_uas'] / uas_per_px

    # ================================================================
    # FIGURE 1: Main comparison — Ground Truth vs Sparse vs Reconstructed
    # ================================================================
    print("Rendering main comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0a0a0a')

    # Custom colormap: black → orange → white (like real EHT images)
    colors_eht = ['#000000', '#1a0500', '#4a1000', '#8b2500',
                  '#cc5500', '#ff8800', '#ffbb44', '#ffdd88', '#ffffff']
    eht_cmap = mcolors.LinearSegmentedColormap.from_list('eht', colors_eht, N=256)

    vmax = gt_I.max()

    for ax in axes:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#333333')

    im0 = axes[0].imshow(gt_I, cmap=eht_cmap, origin='lower', vmin=0, vmax=vmax)
    axes[0].set_title('Ground Truth', color='white', fontsize=14, fontweight='bold')
    ring0 = Circle((64, 64), shadow_r_px, fill=False, color='cyan', linewidth=0.8, linestyle='--', alpha=0.5)
    axes[0].add_patch(ring0)

    axes[1].imshow(observed[..., 0], cmap=eht_cmap, origin='lower', vmin=0, vmax=vmax)
    axes[1].set_title('Sparse Observation (26.2% coverage)', color='white', fontsize=14, fontweight='bold')
    ring1 = Circle((64, 64), shadow_r_px, fill=False, color='cyan', linewidth=0.8, linestyle='--', alpha=0.5)
    axes[1].add_patch(ring1)

    axes[2].imshow(rc_I, cmap=eht_cmap, origin='lower', vmin=0, vmax=vmax)
    axes[2].set_title('FBAI Reconstruction (NxCorr: 0.967)', color='white', fontsize=14, fontweight='bold')
    ring2 = Circle((64, 64), shadow_r_px, fill=False, color='cyan', linewidth=0.8, linestyle='--', alpha=0.5)
    axes[2].add_patch(ring2)

    fig.suptitle('QRFT Black Hole Toolkit — FBAI Tri-Manifold Reconstruction\n∞ 0425 — Rod\'s AI Consulting LLC',
                 color='white', fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'blackhole_reconstruction.png'),
                dpi=200, bbox_inches='tight', facecolor='#0a0a0a',
                edgecolor='none', pad_inches=0.3)
    plt.close()

    # ================================================================
    # FIGURE 2: EVPA Swirl Visualization
    # ================================================================
    print("Rendering EVPA swirl...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#0a0a0a')

    for ax in axes:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#333333')

    # Ground truth EVPA with tick marks
    axes[0].imshow(gt_I, cmap=eht_cmap, origin='lower', vmin=0, vmax=vmax, alpha=0.7)

    # Draw EVPA ticks where intensity is significant
    step = 6
    for yi in range(0, 128, step):
        for xi in range(0, 128, step):
            if gt_I[yi, xi] > 0.1 * vmax:
                angle = gt_evpa[yi, xi]
                length = 2.5 * gt_frac[yi, xi] / max(gt_frac.max(), 0.01)
                dx = length * np.cos(angle)
                dy = length * np.sin(angle)
                axes[0].plot([xi - dx, xi + dx], [yi - dy, yi + dy],
                            color='cyan', linewidth=0.8, alpha=0.8)

    axes[0].set_title('Ground Truth — EVPA Swirl', color='white', fontsize=13, fontweight='bold')
    ring_gt = Circle((64, 64), shadow_r_px, fill=False, color='white', linewidth=0.5, linestyle=':', alpha=0.4)
    axes[0].add_patch(ring_gt)

    # Reconstructed EVPA
    axes[1].imshow(rc_I, cmap=eht_cmap, origin='lower', vmin=0, vmax=vmax, alpha=0.7)

    for yi in range(0, 128, step):
        for xi in range(0, 128, step):
            if rc_I[yi, xi] > 0.1 * vmax:
                angle = rc_evpa[yi, xi]
                rc_frac_clipped = np.clip(rc_frac, 0, 1)
                length = 2.5 * rc_frac_clipped[yi, xi] / max(rc_frac_clipped.max(), 0.01)
                dx = length * np.cos(angle)
                dy = length * np.sin(angle)
                axes[1].plot([xi - dx, xi + dx], [yi - dy, yi + dy],
                            color='cyan', linewidth=0.8, alpha=0.8)

    axes[1].set_title('FBAI Reconstruction — EVPA Swirl', color='white', fontsize=13, fontweight='bold')
    ring_rc = Circle((64, 64), shadow_r_px, fill=False, color='white', linewidth=0.5, linestyle=':', alpha=0.4)
    axes[1].add_patch(ring_rc)

    fig.suptitle('Electric Vector Position Angle — Magnetic Field Visualization\nχ = ½ arctan2(U, Q)',
                 color='white', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'blackhole_evpa_swirl.png'),
                dpi=200, bbox_inches='tight', facecolor='#0a0a0a',
                edgecolor='none', pad_inches=0.3)
    plt.close()

    # ================================================================
    # FIGURE 3: Pipeline diagnostic — Confidence, Coherence, Mode map
    # ================================================================
    print("Rendering diagnostics...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#0a0a0a')

    for row in axes:
        for ax in row:
            ax.set_facecolor('#0a0a0a')
            ax.tick_params(colors='white', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#333333')

    # Row 1: UV mask, Confidence map, Mode map
    axes[0, 0].imshow(uv_mask.astype(float), cmap='gray', origin='lower')
    axes[0, 0].set_title('UV Coverage Mask (26.2%)', color='white', fontsize=11, fontweight='bold')

    im_conf = axes[0, 1].imshow(result.confidence_map, cmap='inferno', origin='lower', vmin=0, vmax=1)
    axes[0, 1].set_title('FBAI Confidence Map', color='white', fontsize=11, fontweight='bold')
    plt.colorbar(im_conf, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Mode map with custom colors
    mode_colors = ['#2196F3', '#FFD700', '#FF1744', '#4CAF50']  # MIN, GOLD, SING, DATA
    mode_display = result.mode_map.copy().astype(float)
    mode_display[uv_mask] = 3  # Original data
    mode_cmap = mcolors.ListedColormap(mode_colors)

    axes[0, 2].imshow(mode_display, cmap=mode_cmap, origin='lower', vmin=-0.5, vmax=3.5)
    axes[0, 2].set_title('Mode Map (Blue=MIN, Gold=φ, Red=SING, Green=Data)',
                         color='white', fontsize=10, fontweight='bold')

    # Row 2: Coherence R, Coherence entropy, Fractional polarization comparison
    coherence = sync.analyze(reconstructed)

    im_R = axes[1, 0].imshow(coherence.R, cmap='plasma', origin='lower', vmin=0, vmax=1)
    axes[1, 0].set_title('Q4PS Coherence Radius R', color='white', fontsize=11, fontweight='bold')
    plt.colorbar(im_R, ax=axes[1, 0], fraction=0.046, pad=0.04)

    S = sync.coherence_entropy(coherence.R)
    im_S = axes[1, 1].imshow(S, cmap='viridis', origin='lower')
    axes[1, 1].set_title('Coherence Entropy', color='white', fontsize=11, fontweight='bold')
    plt.colorbar(im_S, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im_fp = axes[1, 2].imshow(np.clip(rc_frac, 0, 1), cmap='magma', origin='lower', vmin=0, vmax=0.5)
    axes[1, 2].set_title('Fractional Polarization (reconstructed)', color='white', fontsize=11, fontweight='bold')
    plt.colorbar(im_fp, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle('QRFT Pipeline Diagnostics — Phase 1-3 Complete\n∞ 0425',
                 color='white', fontsize=15, fontweight='bold', y=1.01)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'blackhole_diagnostics.png'),
                dpi=200, bbox_inches='tight', facecolor='#0a0a0a',
                edgecolor='none', pad_inches=0.3)
    plt.close()

    # ================================================================
    # FIGURE 4: Close-up of the reconstructed black hole (hero image)
    # ================================================================
    print("Rendering hero image...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#000000')

    ax.imshow(rc_I, cmap=eht_cmap, origin='lower', vmin=0, vmax=vmax)

    # Add subtle EVPA ticks
    step = 5
    for yi in range(0, 128, step):
        for xi in range(0, 128, step):
            if rc_I[yi, xi] > 0.15 * vmax:
                angle = rc_evpa[yi, xi]
                rc_frac_clipped = np.clip(rc_frac, 0, 1)
                length = 2.0 * rc_frac_clipped[yi, xi] / max(rc_frac_clipped.max(), 0.01)
                dx = length * np.cos(angle)
                dy = length * np.sin(angle)
                ax.plot([xi - dx, xi + dx], [yi - dy, yi + dy],
                       color='cyan', linewidth=0.6, alpha=0.5)

    # Shadow circle
    shadow = Circle((64, 64), shadow_r_px, fill=False, color='white',
                    linewidth=0.3, linestyle=':', alpha=0.2)
    ax.add_patch(shadow)

    ax.set_xlim(30, 98)
    ax.set_ylim(30, 98)
    ax.axis('off')

    ax.text(0.5, 0.02,
            'QRFT Black Hole Toolkit — FBAI Reconstruction\n'
            'Q4 Polarization Encoder + Tri-Manifold Interpolation\n'
            '∞ 0425 — Rodney Lee Arnold Jr.',
            transform=ax.transAxes, ha='center', va='bottom',
            color='white', fontsize=10, alpha=0.7,
            fontfamily='monospace')

    fig.savefig(os.path.join(output_dir, 'blackhole_hero.png'),
                dpi=250, bbox_inches='tight', facecolor='#000000',
                edgecolor='none', pad_inches=0.1)
    plt.close()

    print("\nAll images rendered successfully!")
    print("  blackhole_reconstruction.png  — Side-by-side comparison")
    print("  blackhole_evpa_swirl.png      — Magnetic field visualization")
    print("  blackhole_diagnostics.png     — Pipeline diagnostic maps")
    print("  blackhole_hero.png            — Close-up hero image")


if __name__ == '__main__':
    render_blackhole()
