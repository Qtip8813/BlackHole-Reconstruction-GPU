"""
QRFT Black Hole Toolkit — Phase 6.1 Temporal Coherence Benchmark
================================================================
Binary inspiral / merger benchmark with QRFT peak-lock, residual-style
temporal seeding, centroid-guided stabilization, and temporal metrics.

Run with:
    python -m tests.test_phase6

Created by: Rodney Lee Arnold Jr. ∞ 0425
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from typing import Dict, List, Tuple, Optional

from core.q4_encoder import Q4StokesEncoder
from core.q4ps_sync import Q4PSSynchronizer
from core.fbai_interpolator import FBAIInterpolator
from entropy.coherence_gate import CoherenceGate
from gpu.parallel_refiner import ParallelRefiner
from pipeline.eht_loader import EHTLoader


# ============================================================
# Metrics
# ============================================================

def nxcorr(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.flatten(), b.flatten()
    if af.std() == 0 or bf.std() == 0:
        return 0.0
    return float(np.corrcoef(af, bf)[0, 1])


def center_of_mass_I(stokes: np.ndarray) -> Tuple[float, float]:
    I = np.maximum(stokes[..., 0], 0.0)
    total = I.sum()
    h, w = I.shape
    if total <= 1e-12:
        return h / 2.0, w / 2.0
    y, x = np.indices(I.shape)
    return float((y * I).sum() / total), float((x * I).sum() / total)


def peak_location_I(stokes: np.ndarray) -> Tuple[int, int]:
    idx = np.unravel_index(np.argmax(stokes[..., 0]), stokes[..., 0].shape)
    return int(idx[0]), int(idx[1])


def centroid_error(gt: np.ndarray, rc: np.ndarray) -> float:
    gy, gx = center_of_mass_I(gt)
    ry, rx = center_of_mass_I(rc)
    return float(np.sqrt((gy - ry) ** 2 + (gx - rx) ** 2))


def peak_error(gt: np.ndarray, rc: np.ndarray) -> float:
    gy, gx = peak_location_I(gt)
    ry, rx = peak_location_I(rc)
    return float(np.sqrt((gy - ry) ** 2 + (gx - rx) ** 2))


def peak_intensity_error(gt: np.ndarray, rc: np.ndarray) -> float:
    return float(abs(float(gt[..., 0].max()) - float(rc[..., 0].max())))


def temporal_nxcorr(a: np.ndarray, b: np.ndarray) -> float:
    return nxcorr(a[..., 0], b[..., 0])


def evpa_map(stokes: np.ndarray) -> np.ndarray:
    return 0.5 * np.arctan2(stokes[..., 2], stokes[..., 1])


def evpa_temporal_rmse(prev_frame: np.ndarray, next_frame: np.ndarray) -> float:
    e1 = evpa_map(prev_frame)
    e2 = evpa_map(next_frame)
    diff = e2 - e1
    diff = np.arctan2(np.sin(2 * diff), np.cos(2 * diff)) / 2.0
    return float(np.degrees(np.sqrt(np.mean(diff ** 2))))


# ============================================================
# Inspiral Generator
# ============================================================

def generate_binary_inspiral_sequence(
    loader: EHTLoader,
    num_frames: int = 20,
    npix: int = 128,
    noise_level: float = 0.03,
    max_offset: int = 20,
    seed: int = 425,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)

    base_a = loader.synthetic_blackhole(
        npix=npix, fov_uas=150.0, mass_msun=6.5e9,
        spin=0.9, inclination_deg=17.0, noise_level=0.0, seed=seed
    )["stokes"]

    base_b = loader.synthetic_blackhole(
        npix=npix, fov_uas=150.0, mass_msun=6.5e9,
        spin=0.9, inclination_deg=17.0, noise_level=0.0, seed=seed + 1
    )["stokes"]

    sequence = []

    for frame in range(num_frames):
        frac = frame / max(num_frames - 1, 1)
        offset = int(round(max_offset * (1.0 - frac)))

        stokes_a = np.roll(base_a, -offset, axis=1)
        stokes_b = np.roll(base_b, +offset, axis=1)

        mod_a = 1.0 + 0.08 * np.cos(2 * np.pi * frac)
        mod_b = 1.0 + 0.08 * np.sin(2 * np.pi * frac)

        combined = mod_a * stokes_a + mod_b * stokes_b
        combined += rng.normal(0.0, noise_level, combined.shape)

        combined[..., 0] = np.clip(combined[..., 0], 0.0, None)
        sequence.append(combined.astype(np.float32))

    return sequence


# ============================================================
# Baselines
# ============================================================

class TemporalBaselines:
    @staticmethod
    def zeros_fill(stokes: np.ndarray, uv_mask: np.ndarray) -> np.ndarray:
        result = stokes.copy()
        result[~uv_mask] = 0.0
        return result

    @staticmethod
    def radial_fill(stokes: np.ndarray, uv_mask: np.ndarray, n_bins: int = 32) -> np.ndarray:
        h, w = stokes.shape[:2]
        cy, cx = h // 2, w // 2
        y, x = np.mgrid[0:h, 0:w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        theta = np.arctan2(y - cy, x - cx)

        bin_edges = np.linspace(0, r.max(), n_bins + 1)
        result = stokes.copy()

        for ch in range(4):
            profile = np.zeros(n_bins)
            counts = np.zeros(n_bins)

            for b in range(n_bins):
                in_bin = (r >= bin_edges[b]) & (r < bin_edges[b + 1]) & uv_mask
                if in_bin.any():
                    profile[b] = stokes[in_bin, ch].mean()
                    counts[b] = in_bin.sum()

            valid = counts > 0
            if valid.sum() >= 2:
                centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                profile = np.interp(centers, centers[valid], profile[valid])

            for b in range(n_bins):
                gap = (r >= bin_edges[b]) & (r < bin_edges[b + 1]) & (~uv_mask)
                valid_bin = (r >= bin_edges[b]) & (r < bin_edges[b + 1]) & uv_mask

                if gap.any():
                    if valid_bin.sum() > 3:
                        angles = theta[valid_bin]
                        vals = stokes[valid_bin, ch]
                        A = vals.mean()
                        B = np.mean(vals * np.cos(angles)) / max(np.mean(np.cos(angles) ** 2), 1e-10)
                        C = np.mean(vals * np.sin(angles)) / max(np.mean(np.sin(angles) ** 2), 1e-10)
                        result[gap, ch] = A + 0.3 * B * np.cos(theta[gap]) + 0.3 * C * np.sin(theta[gap])
                    else:
                        result[gap, ch] = profile[b]

        result[..., 0] = np.clip(result[..., 0], 0.0, None)
        return result

    @staticmethod
    def smooth_interpolate(stokes: np.ndarray, uv_mask: np.ndarray, iterations: int = 20) -> np.ndarray:
        result = TemporalBaselines.radial_fill(stokes, uv_mask)
        gap = ~uv_mask
        h, w = stokes.shape[:2]

        for _ in range(iterations):
            padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="reflect")
            neighbor_sum = np.zeros_like(result)
            count = np.zeros((h, w, 1), dtype=np.float32)

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    neighbor_sum += padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                    count += 1.0

            avg = neighbor_sum / np.maximum(count, 1.0)

            for ch in range(4):
                result[..., ch] = np.where(gap, avg[..., ch], result[..., ch])

            result[..., 0] = np.clip(result[..., 0], 0.0, None)

        return result


# ============================================================
# QRFT 6.1 Stabilizers
# ============================================================

def residual_temporal_seed(
    observed: np.ndarray,
    uv_mask: np.ndarray,
    prev_recon: Optional[np.ndarray],
) -> np.ndarray:
    seeded = observed.copy()

    if prev_recon is None:
        return seeded

    gap = ~uv_mask
    delta_I = np.abs(observed[..., 0] - prev_recon[..., 0])
    adaptive = delta_I / max(float(delta_I.max()), 1e-8)
    adaptive = np.clip(adaptive, 0.0, 1.0)

    for ch in range(seeded.shape[-1]):
        seeded[..., ch] = np.where(
            gap,
            (1.0 - adaptive) * prev_recon[..., ch] + adaptive * seeded[..., ch],
            seeded[..., ch],
        )

    seeded[..., 0] = np.clip(seeded[..., 0], 0.0, None)
    return seeded


def peak_lock_blend(
    recon: np.ndarray,
    observed: np.ndarray,
    uv_mask: np.ndarray,
    lock_radius: int = 4,
    lock_strength: float = 0.65,
) -> np.ndarray:
    out = recon.copy()

    obs_I = observed[..., 0]
    py, px = np.unravel_index(np.argmax(obs_I), obs_I.shape)

    h, w = obs_I.shape
    y, x = np.indices((h, w))
    peak_region = ((x - px) ** 2 + (y - py) ** 2) <= lock_radius ** 2
    trusted = peak_region & uv_mask

    for ch in range(recon.shape[-1]):
        out[..., ch] = np.where(
            trusted,
            lock_strength * observed[..., ch] + (1.0 - lock_strength) * recon[..., ch],
            out[..., ch],
        )

    out[..., 0] = np.clip(out[..., 0], 0.0, None)
    return out


def centroid_guided_blend(
    recon: np.ndarray,
    prev_recon: Optional[np.ndarray],
    strength: float = 0.20,
) -> np.ndarray:
    if prev_recon is None:
        return recon

    rc_y, rc_x = center_of_mass_I(recon)
    pr_y, pr_x = center_of_mass_I(prev_recon)

    dy = int(round(pr_y - rc_y))
    dx = int(round(pr_x - rc_x))

    shifted = np.roll(np.roll(recon, dy, axis=0), dx, axis=1)

    out = (1.0 - strength) * recon + strength * shifted
    out[..., 0] = np.clip(out[..., 0], 0.0, None)
    return out


# ============================================================
# QRFT Reconstruction Wrapper
# ============================================================

def reconstruct_qrft_frame(
    observed: np.ndarray,
    uv_mask: np.ndarray,
    metadata: Dict,
    prev_recon: Optional[np.ndarray] = None,
    use_refiner: bool = False,
):
    encoder = Q4StokesEncoder(n_layers=1)
    sync = Q4PSSynchronizer(R_c=0.5, alpha=8.0, K=1.0)

    interpolator = FBAIInterpolator(
        encoder=encoder,
        synchronizer=sync,
        max_iterations=6 if prev_recon is None else 3,
        verbose=False,
    )

    seeded = residual_temporal_seed(observed, uv_mask, prev_recon)

    t0 = time.time()
    result = interpolator.reconstruct(seeded, uv_mask, metadata)
    recon = result.reconstructed.copy()

    # Confidence-gated temporal blend
    if prev_recon is not None:
        low_conf = result.confidence_map < 0.4
        for ch in range(recon.shape[-1]):
            recon[..., ch] = np.where(
                low_conf,
                0.30 * prev_recon[..., ch] + 0.70 * recon[..., ch],
                recon[..., ch],
            )

    # Phase 6.1 stabilization
    recon = peak_lock_blend(recon, observed, uv_mask, lock_radius=4, lock_strength=0.65)
    recon = centroid_guided_blend(recon, prev_recon, strength=0.20)

    refine_stats = None
    if use_refiner:
        refiner = ParallelRefiner(sync, use_gpu=True)
        recon, refine_stats = refiner.refine(
            recon,
            uv_mask,
            max_iterations=20,
            global_kernel=True,
            verbose=False,
        )

    dt = time.time() - t0

    return {
        "recon": recon,
        "time_s": dt,
        "result": result,
        "refine_stats": refine_stats,
    }


# ============================================================
# Main Benchmark
# ============================================================

def run_phase6():
    print("\n" + "=" * 72)
    print("  QRFT BLACK HOLE TOOLKIT — PHASE 6.1 TEMPORAL COHERENCE BENCHMARK")
    print("  Binary Inspiral / Merger Sequence")
    print("  Peak Lock + Residual Temporal Seeding + Centroid Stabilization")
    print("  ∞ 0425 — Rod's AI Consulting LLC")
    print("=" * 72)

    loader = EHTLoader()

    test_configs = [
        {"name": "Medium Noise", "noise": 0.02, "seed": 425},
        {"name": "High Noise", "noise": 0.05, "seed": 425},
    ]

    for config in test_configs:
        print(f"\n{'=' * 72}")
        print(f"  TEST: {config['name']}")
        print(f"{'=' * 72}")

        sequence = generate_binary_inspiral_sequence(
            loader=loader,
            num_frames=20,
            npix=128,
            noise_level=config["noise"],
            seed=config["seed"],
        )

        ref = loader.synthetic_blackhole(
            npix=128,
            fov_uas=150.0,
            mass_msun=6.5e9,
            spin=0.9,
            inclination_deg=17.0,
            noise_level=0.0,
            seed=config["seed"],
        )

        uv_mask = ref["uv_mask"]
        metadata = ref["metadata"]

        methods = {
            "Zeros": {"frames": [], "times": []},
            "Radial": {"frames": [], "times": []},
            "Smooth": {"frames": [], "times": []},
            "QRFT": {"frames": [], "times": [], "coh_valid": [], "gate_quality": []},
            "QRFT+Refiner": {"frames": [], "times": [], "coh_valid": [], "gate_quality": []},
        }

        prev_qrft = None
        prev_qrft_ref = None

        for gt in sequence:
            observed = gt.copy()
            observed[~uv_mask] = 0.0

            t0 = time.time()
            methods["Zeros"]["frames"].append(TemporalBaselines.zeros_fill(gt, uv_mask))
            methods["Zeros"]["times"].append(time.time() - t0)

            t0 = time.time()
            methods["Radial"]["frames"].append(TemporalBaselines.radial_fill(gt, uv_mask))
            methods["Radial"]["times"].append(time.time() - t0)

            t0 = time.time()
            methods["Smooth"]["frames"].append(
                TemporalBaselines.smooth_interpolate(gt, uv_mask, iterations=20)
            )
            methods["Smooth"]["times"].append(time.time() - t0)

            out = reconstruct_qrft_frame(
                observed=observed,
                uv_mask=uv_mask,
                metadata=metadata,
                prev_recon=prev_qrft,
                use_refiner=False,
            )

            frame = out["recon"]
            methods["QRFT"]["frames"].append(frame)
            methods["QRFT"]["times"].append(out["time_s"])
            prev_qrft = frame

            sync = Q4PSSynchronizer(R_c=0.5, alpha=8.0, K=1.0)
            gate = CoherenceGate(sync)
            coh = sync.analyze(frame)
            gate_result = gate.evaluate(frame, mask_evaluate=(~uv_mask))
            methods["QRFT"]["coh_valid"].append(coh.stats["valid_fraction"])
            methods["QRFT"]["gate_quality"].append(gate_result.stats["mean_quality"])

            out_ref = reconstruct_qrft_frame(
                observed=observed,
                uv_mask=uv_mask,
                metadata=metadata,
                prev_recon=prev_qrft_ref,
                use_refiner=True,
            )

            frame_ref = out_ref["recon"]
            methods["QRFT+Refiner"]["frames"].append(frame_ref)
            methods["QRFT+Refiner"]["times"].append(out_ref["time_s"])
            prev_qrft_ref = frame_ref

            coh_ref = sync.analyze(frame_ref)
            gate_result_ref = gate.evaluate(frame_ref, mask_evaluate=(~uv_mask))
            methods["QRFT+Refiner"]["coh_valid"].append(coh_ref.stats["valid_fraction"])
            methods["QRFT+Refiner"]["gate_quality"].append(gate_result_ref.stats["mean_quality"])

        print(f"\n  {'Method':<15} {'Mean NxCorr':>11} {'CentErr':>9} {'PeakErr':>9} {'PeakIΔ':>9} {'TempJit':>9} {'EVPAΔ':>8} {'FPS':>8}")
        print(f"  {'-' * 88}")

        for method_name, bundle in methods.items():
            frames = bundle["frames"]

            nxc_vals = []
            cent_vals = []
            peak_vals = []
            peaki_vals = []
            jitter_vals = []
            evpa_vals = []

            for i, gt in enumerate(sequence):
                rc = frames[i]

                nxc_vals.append(nxcorr(gt[..., 0], rc[..., 0]))
                cent_vals.append(centroid_error(gt, rc))
                peak_vals.append(peak_error(gt, rc))
                peaki_vals.append(peak_intensity_error(gt, rc))

                if i > 0:
                    gt_prev = sequence[i - 1]
                    rc_prev = frames[i - 1]

                    gt_c = center_of_mass_I(gt)
                    gt_p = center_of_mass_I(gt_prev)
                    rc_c = center_of_mass_I(rc)
                    rc_p = center_of_mass_I(rc_prev)

                    gt_step = np.sqrt((gt_c[0] - gt_p[0]) ** 2 + (gt_c[1] - gt_p[1]) ** 2)
                    rc_step = np.sqrt((rc_c[0] - rc_p[0]) ** 2 + (rc_c[1] - rc_p[1]) ** 2)
                    jitter_vals.append(abs(rc_step - gt_step))

                    evpa_vals.append(abs(
                        evpa_temporal_rmse(gt_prev, gt) -
                        evpa_temporal_rmse(rc_prev, rc)
                    ))

            mean_time = float(np.mean(bundle["times"]))
            fps = 1.0 / max(mean_time, 1e-8)

            print(
                f"  {method_name:<15}"
                f" {np.mean(nxc_vals):>11.4f}"
                f" {np.mean(cent_vals):>9.3f}"
                f" {np.mean(peak_vals):>9.3f}"
                f" {np.mean(peaki_vals):>9.4f}"
                f" {np.mean(jitter_vals):>9.3f}"
                f" {np.mean(evpa_vals):>8.3f}"
                f" {fps:>8.2f}"
            )

            if "QRFT" in method_name:
                print(f"    Coherence valid: {np.mean(bundle['coh_valid']) * 100:.1f}%")
                print(f"    Gate quality:    {np.mean(bundle['gate_quality']):.4f}")

        merger_idx = len(sequence) - 2
        print(f"\n  Merger-focused frame check (Frame {merger_idx + 1}):")
        for method_name, bundle in methods.items():
            gt = sequence[merger_idx]
            rc = bundle["frames"][merger_idx]
            print(
                f"    {method_name:<15}"
                f" NxCorr={nxcorr(gt[..., 0], rc[..., 0]):.4f}"
                f"  CentErr={centroid_error(gt, rc):.3f}"
                f"  PeakErr={peak_error(gt, rc):.3f}"
                f"  PeakIΔ={peak_intensity_error(gt, rc):.4f}"
            )

    print(f"\n{'=' * 72}")
    print("  PHASE 6.1 COMPLETE")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    run_phase6()
