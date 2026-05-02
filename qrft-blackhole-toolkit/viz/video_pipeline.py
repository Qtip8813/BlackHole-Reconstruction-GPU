"""
Video Reconstruction Pipeline
===============================
Generates time-evolving black hole video by reconstructing
sequential frames with temporal coherence constraints.

The accretion disk plasma orbits the black hole at relativistic
speeds. This pipeline:
    1. Generates/loads sequential time snapshots
    2. Reconstructs each frame using FBAI
    3. Enforces temporal Q4PS coherence between frames
       (frame N+1 can't wildly differ from frame N)
    4. Outputs frame sequence for video encoding

Temporal coherence uses the same Kuramoto coupling model as
spatial coherence — the coupling constant K now connects
adjacent frames instead of adjacent pixels.

Created by: Rodney Lee Arnold Jr. ∞ 0425
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
import time


class VideoReconstructor:
    """
    Temporal black hole video reconstruction.

    Usage:
        from core import Q4StokesEncoder, Q4PSSynchronizer
        from core.fbai_interpolator import FBAIInterpolator

        vid = VideoReconstructor(
            encoder=Q4StokesEncoder(),
            synchronizer=Q4PSSynchronizer(),
            interpolator=FBAIInterpolator(encoder, sync, verbose=False)
        )

        # Generate and reconstruct a synthetic rotation video
        result = vid.reconstruct_synthetic(
            n_frames=30, npix=128, spin=0.9, fps=10
        )

        # Save frames
        vid.save_frames(result, output_dir='frames/')
    """

    def __init__(self, encoder, synchronizer, interpolator,
                 temporal_K: float = 0.5,
                 temporal_damping: float = 0.3,
                 verbose: bool = True):
        """
        Args:
            encoder:           Q4StokesEncoder instance
            synchronizer:      Q4PSSynchronizer instance
            interpolator:      FBAIInterpolator instance
            temporal_K:        Temporal coupling strength (higher = smoother
                               transitions between frames, lower = more
                               independent frames)
            temporal_damping:  How much the previous frame influences the next
                               reconstruction. 0 = no influence, 1 = identical.
            verbose:           Print progress
        """
        self.encoder = encoder
        self.sync = synchronizer
        self.interpolator = interpolator
        self.temporal_K = temporal_K
        self.temporal_damping = temporal_damping
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"  [Video] {msg}")

    # ------------------------------------------------------------------
    # Synthetic Video Generation
    # ------------------------------------------------------------------

    def generate_synthetic_frames(self, n_frames: int = 30,
                                    npix: int = 128,
                                    fov_uas: float = 150.0,
                                    spin: float = 0.9,
                                    inclination_deg: float = 17.0,
                                    orbital_period_frames: int = 60,
                                    noise_level: float = 0.02,
                                    seed: int = 425) -> Dict:
        """
        Generate synthetic time-evolving black hole frames.

        The accretion disk rotates over time, creating:
            - Changing Doppler asymmetry direction
            - Evolving EVPA spiral pattern
            - Turbulent brightness fluctuations

        Args:
            n_frames:              Number of frames to generate
            npix:                  Image size
            fov_uas:               Field of view (microarcseconds)
            spin:                  Black hole spin [0, 1)
            inclination_deg:       Observer inclination
            orbital_period_frames: Frames per full orbit
            noise_level:           Per-frame noise level
            seed:                  Random seed

        Returns:
            Dict with 'frames', 'uv_masks', 'metadata'
        """
        np.random.seed(seed)

        half_fov = fov_uas / 2.0
        shadow_radius = 21.0  # uas for M87*
        inc = np.radians(inclination_deg)

        x = np.linspace(-half_fov, half_fov, npix)
        y = np.linspace(-half_fov, half_fov, npix)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)

        ring_width = 5.0

        frames = []
        uv_masks = []

        for frame_idx in range(n_frames):
            # Time phase: orbital rotation
            t = frame_idx / orbital_period_frames * 2 * np.pi

            # Ring (constant structure)
            ring = np.exp(-0.5 * ((R - shadow_radius) / ring_width)**2)

            # Rotating Doppler pattern
            doppler = 1.0 + 0.4 * spin * np.sin(inc) * np.cos(theta - t)
            I = ring * doppler
            I = I / (I.max() + 1e-10)

            # Rotating EVPA + turbulent perturbation
            turbulence = 0.1 * np.sin(3 * theta + 2 * t) * ring
            evpa_angle = theta - t + spin * 0.3 * np.exp(-R / shadow_radius) + turbulence

            frac_pol = 0.18 * ring

            Q = frac_pol * I * np.cos(2 * evpa_angle)
            U = frac_pol * I * np.sin(2 * evpa_angle)
            V = 0.01 * I * np.sin(theta - t + spin * 0.5)

            # Per-frame noise
            noise_std = noise_level
            I += np.random.normal(0, noise_std, I.shape)
            Q += np.random.normal(0, noise_std * 0.5, Q.shape)
            U += np.random.normal(0, noise_std * 0.5, U.shape)
            V += np.random.normal(0, noise_std * 0.1, V.shape)
            I = np.maximum(I, 0)

            stokes = np.stack([I, Q, U, V], axis=-1)
            frames.append(stokes)

            # Each frame gets slightly different uv-coverage
            # (Earth rotation changes baseline projections)
            uv_mask = self._generate_rotated_uv_mask(npix, frame_idx, n_frames)
            uv_masks.append(uv_mask)

        metadata = {
            'n_frames': n_frames,
            'npix': npix,
            'fov_uas': fov_uas,
            'spin': spin,
            'inclination_deg': inclination_deg,
            'shadow_radius_uas': shadow_radius,
            'orbital_period_frames': orbital_period_frames,
            'noise_level': noise_level,
            'synthetic': True
        }

        return {
            'frames': np.stack(frames),       # (N, H, W, 4)
            'uv_masks': np.stack(uv_masks),   # (N, H, W)
            'metadata': metadata
        }

    def _generate_rotated_uv_mask(self, npix: int,
                                    frame_idx: int,
                                    n_frames: int) -> np.ndarray:
        """Generate UV mask that rotates with Earth rotation."""
        mask = np.zeros((npix, npix), dtype=bool)
        u = np.linspace(-1, 1, npix)
        v = np.linspace(-1, 1, npix)
        U, V = np.meshgrid(u, v)

        # Baseline rotation angle (Earth rotation over observation)
        rotation = frame_idx / n_frames * np.pi * 0.3  # ~54° total rotation

        n_baselines = 28
        np.random.seed(425 + frame_idx)

        for _ in range(n_baselines):
            r0 = np.random.uniform(0.1, 0.95)
            angle0 = np.random.uniform(0, 2 * np.pi) + rotation
            arc_length = np.random.uniform(0.3, 1.2)
            width = 0.03

            for t in np.linspace(0, arc_length, 50):
                cx = r0 * np.cos(angle0 + t)
                cy = r0 * np.sin(angle0 + t)
                dist = np.sqrt((U - cx)**2 + (V - cy)**2)
                mask |= (dist < width)

        mask |= mask[::-1, ::-1]
        return mask

    # ------------------------------------------------------------------
    # Video Reconstruction
    # ------------------------------------------------------------------

    def reconstruct_synthetic(self, n_frames: int = 30,
                                npix: int = 128,
                                spin: float = 0.9,
                                **kwargs) -> Dict:
        """
        Generate and reconstruct a synthetic black hole video.

        Args:
            n_frames: Number of frames
            npix: Image size
            spin: Black hole spin
            **kwargs: Additional args for generate_synthetic_frames()

        Returns:
            Dict with 'ground_truth', 'reconstructed', 'confidence',
                      'metadata', 'timing'
        """
        self._log(f"Generating {n_frames} synthetic frames ({npix}x{npix})")

        data = self.generate_synthetic_frames(
            n_frames=n_frames, npix=npix, spin=spin, **kwargs
        )

        return self.reconstruct_frames(
            frames=data['frames'],
            uv_masks=data['uv_masks'],
            metadata=data['metadata'],
            ground_truth=data['frames']
        )

    def reconstruct_frames(self, frames: np.ndarray,
                             uv_masks: np.ndarray,
                             metadata: Dict,
                             ground_truth: Optional[np.ndarray] = None) -> Dict:
        """
        Reconstruct a sequence of frames with temporal coherence.

        Args:
            frames:       (N, H, W, 4) observed Stokes frames
            uv_masks:     (N, H, W) boolean coverage masks
            metadata:     Physical parameters
            ground_truth: Optional (N, H, W, 4) for quality comparison

        Returns:
            Dict with reconstructed frames, confidence, metrics
        """
        N, H, W, _ = frames.shape
        self._log(f"Reconstructing {N} frames with temporal coupling K={self.temporal_K}")

        reconstructed = np.zeros_like(frames)
        confidence = np.zeros((N, H, W))
        per_frame_metrics = []

        prev_frame = None
        total_time = 0

        for i in range(N):
            t0 = time.time()

            # Sparse observation (zero out gaps)
            observed = frames[i].copy()
            observed[~uv_masks[i]] = 0.0

            # If we have a previous frame, use it as temporal prior
            if prev_frame is not None and self.temporal_damping > 0:
                # Blend previous reconstruction into gap regions
                # as initial guess (temporal coherence)
                gap = ~uv_masks[i]
                observed[gap] = self.temporal_damping * prev_frame[gap]

            # Run FBAI reconstruction
            result = self.interpolator.reconstruct(
                stokes_array=observed,
                uv_mask=uv_masks[i],
                metadata=metadata
            )

            # Temporal coherence enforcement
            if prev_frame is not None:
                result_stokes = result.reconstructed
                # Smooth transition: blend with previous where confidence is low
                low_conf = result.confidence_map < 0.5
                temporal_blend = self.temporal_K * prev_frame + \
                                 (1 - self.temporal_K) * result_stokes
                for ch in range(4):
                    result_stokes[..., ch] = np.where(
                        low_conf,
                        temporal_blend[..., ch],
                        result_stokes[..., ch]
                    )
                reconstructed[i] = result_stokes
            else:
                reconstructed[i] = result.reconstructed

            confidence[i] = result.confidence_map
            prev_frame = reconstructed[i].copy()

            frame_time = time.time() - t0
            total_time += frame_time

            # Per-frame metrics
            metrics = {
                'frame': i,
                'time_s': frame_time,
                'mean_confidence': float(result.confidence_map.mean()),
                'acceptance_rate': result.stats['acceptance_rate'],
            }

            if ground_truth is not None:
                nxc = float(np.corrcoef(
                    ground_truth[i, ..., 0].flatten(),
                    reconstructed[i, ..., 0].flatten()
                )[0, 1])
                metrics['nxcorr_I'] = nxc

            per_frame_metrics.append(metrics)

            if self.verbose and (i < 3 or i % 5 == 0 or i == N - 1):
                nxc_str = f", NxCorr={metrics.get('nxcorr_I', 'N/A'):.4f}" \
                    if 'nxcorr_I' in metrics else ""
                self._log(f"  Frame {i+1}/{N}: {frame_time:.2f}s, "
                         f"conf={metrics['mean_confidence']:.3f}"
                         f"{nxc_str}")

        # Summary
        avg_fps = N / total_time
        summary = {
            'n_frames': N,
            'total_time_s': total_time,
            'avg_fps': avg_fps,
            'avg_frame_time_s': total_time / N,
            'mean_confidence': float(confidence.mean()),
            'temporal_K': self.temporal_K,
            'temporal_damping': self.temporal_damping,
            'per_frame': per_frame_metrics
        }

        if ground_truth is not None:
            nxcorrs = [m['nxcorr_I'] for m in per_frame_metrics if 'nxcorr_I' in m]
            summary['mean_nxcorr'] = float(np.mean(nxcorrs))
            summary['min_nxcorr'] = float(np.min(nxcorrs))

        self._log(f"Video reconstruction complete:")
        self._log(f"  Total time:  {total_time:.1f}s")
        self._log(f"  Average FPS: {avg_fps:.2f}")
        if 'mean_nxcorr' in summary:
            self._log(f"  Mean NxCorr: {summary['mean_nxcorr']:.4f}")

        return {
            'ground_truth': ground_truth,
            'reconstructed': reconstructed,
            'confidence': confidence,
            'uv_masks': uv_masks,
            'metadata': metadata,
            'stats': summary
        }

    # ------------------------------------------------------------------
    # Frame Export
    # ------------------------------------------------------------------

    def save_frames(self, result: Dict, output_dir: str = 'frames/'):
        """
        Save reconstructed frames as individual numpy files and
        a metadata JSON.

        Args:
            result: Dict from reconstruct_frames()
            output_dir: Output directory
        """
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        N = result['reconstructed'].shape[0]

        for i in range(N):
            np.save(os.path.join(output_dir, f'frame_{i:04d}.npy'),
                    result['reconstructed'][i])

        # Save confidence maps
        for i in range(N):
            np.save(os.path.join(output_dir, f'confidence_{i:04d}.npy'),
                    result['confidence'][i])

        # Metadata
        stats = result['stats'].copy()
        # Convert per_frame metrics for JSON serialization
        stats['per_frame'] = [
            {k: float(v) if isinstance(v, (np.floating, float)) else v
             for k, v in m.items()}
            for m in stats['per_frame']
        ]

        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        self._log(f"Saved {N} frames to {output_dir}")

    def render_video_matplotlib(self, result: Dict,
                                  output_path: str = 'blackhole_video.mp4',
                                  fps: int = 10,
                                  dpi: int = 150) -> str:
        """
        Render video using matplotlib animation.

        Args:
            result: Dict from reconstruct_frames()
            output_path: Output video file path
            fps: Frames per second
            dpi: Resolution

        Returns:
            Path to saved video
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.animation import FuncAnimation, FFMpegWriter

        frames = result['reconstructed']
        N, H, W, _ = frames.shape

        # EHT-style colormap
        colors_eht = ['#000000', '#1a0500', '#4a1000', '#8b2500',
                       '#cc5500', '#ff8800', '#ffbb44', '#ffdd88', '#ffffff']
        eht_cmap = mcolors.LinearSegmentedColormap.from_list('eht', colors_eht, N=256)

        vmax = frames[..., 0].max()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.patch.set_facecolor('#000000')
        ax.set_facecolor('#000000')
        ax.axis('off')

        im = ax.imshow(frames[0, ..., 0], cmap=eht_cmap, origin='lower',
                       vmin=0, vmax=vmax)

        title = ax.set_title('', color='white', fontsize=12, fontfamily='monospace')

        def update(frame_idx):
            im.set_data(frames[frame_idx, ..., 0])

            per_frame = result['stats']['per_frame']
            nxc = per_frame[frame_idx].get('nxcorr_I', 0)
            conf = per_frame[frame_idx].get('mean_confidence', 0)

            title.set_text(
                f'QRFT Black Hole Reconstruction  ∞ 0425\n'
                f'Frame {frame_idx+1}/{N}  |  '
                f'NxCorr: {nxc:.3f}  |  Confidence: {conf:.3f}'
            )
            return [im, title]

        anim = FuncAnimation(fig, update, frames=N, interval=1000//fps, blit=False)

        # Try FFmpeg first, fall back to pillow
        try:
            writer = FFMpegWriter(fps=fps, metadata={'title': 'QRFT Black Hole'})
            anim.save(output_path, writer=writer, dpi=dpi)
        except Exception:
            # Fall back to GIF
            gif_path = output_path.replace('.mp4', '.gif')
            anim.save(gif_path, writer='pillow', fps=fps, dpi=dpi)
            output_path = gif_path

        plt.close()
        self._log(f"Video saved to {output_path}")
        return output_path

    def __repr__(self) -> str:
        return (f"VideoReconstructor(temporal_K={self.temporal_K}, "
                f"damping={self.temporal_damping})")
