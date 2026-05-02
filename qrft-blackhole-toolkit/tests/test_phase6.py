"""
QRFT Black Hole Toolkit — Phase 6 Temporal Coherence Benchmark
==============================================================
Benchmarks QRFT against baseline reconstruction methods on
time-evolving binary inspiral / merger sequences.

Focus:
    - Frame-to-frame structural consistency
    - Merger centering accuracy
    - Temporal jitter
    - Coherence stability
    - Polarization continuity
    - Speed

Why Phase 6 exists:
    Phase 5 measures static image fidelity.
    Phase 6 measures dynamic field fidelity.

Run with:
    python -m tests.test_phase6_temporal

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
# Utility metrics
# ============================================================

def nxcorr(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.flatten(), b.flatten()
    if af.std() == 0 or bf.std() == 0:
        return 0.0
    return float(np.corrcoef(af, bf)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def temporal_nxcorr(prev_frame: np.ndarray, next_frame: np.ndarray) -> float:
    return nxcorr(prev_frame[..., 0], next_frame[..., 0])


def center_of_mass_I(stokes: np.ndarray) -> Tuple[float, float]:
    I = np.maximum(stokes[..., 0], 0.0)
    total = I.sum()
    if total <= 1e-12:
        h, w = I.shape
        return h / 2.0, w / 2.0
    y, x = np.indices(I.shape)
    cy = float((y * I).sum() / total)
    cx = float((x * I).sum() / total)
    return cy, cx


def peak_location_I(stokes: np.ndarray) -> Tuple[int, int]:
    I = stokes[..., 0]
    idx = np.unravel_index(np.argmax(I), I.shape)
    return int(idx[0]), int(idx[1])


def centroid_error(gt: np.ndarray, rc: np.ndarray) -> float:
    gy, gx = center_of_mass_I(gt)
    ry, rx = center_of_mass_I(rc)
    return float(np.sqrt((gy - ry) ** 2 + (gx - rx) ** 2))


def peak_error(gt: np.ndarray, rc: np.ndarray) -> float:
    gy, gx = peak_location_I(gt)
    ry, rx = peak_location_I(rc)
    return float(np.sqrt((gy - ry) ** 2 + (gx - rx) ** 2))


def evpa_map(stokes: np.ndarray) -> np.ndarray:
    Q = stokes[..., 1]
    U = stokes[..., 2]
    return 0.5 * np.arctan2(U, Q)


def evpa_temporal_rmse(prev_frame: np.ndarray, next_frame: np.ndarray) -> float:
    e1 = evpa_map(prev_frame)
    e2 = evpa_map(next_frame)
    diff = e2 - e1
    diff = np.arctan2(np.sin(2 * diff), np.cos(2 * diff)) / 2.0
    return float(np.degrees(np.sqrt(np.mean(diff ** 2))))


def frac_pol_map(stokes: np.ndarray) -> np.ndarray:
    I = stokes[..., 0]
    Q = stokes[..., 1]
    U = stokes[..., 2]
    lp = np.sqrt(Q ** 2 + U ** 2)
    return np.clip(lp / np.where(I > 1e-3, I, 1e-10), 0, 1)


def frame_difference_energy(prev_frame: np.ndarray, next_frame: np.ndarray) -> float:
    d = next_frame[..., 0] - prev_frame[..., 0]
    return float(np.mean(d ** 2))


# ============================================================
# Synthetic binary inspiral generator
# ============================================================

def generate_binary_inspiral_sequence(
    loader: EHTLoader,
    num_frames: int = 20,
    npix: int = 128,
    noise_level: float = 0.03,
    max_offset: int = 20,
    seed: int = 425,
) -> List[np.ndarray]:
    """
    Builds a simple binary inspiral by superposing two synthetic black holes
    whose lateral offset shrinks over time.
    """
    rng = np.random.default_rng(seed)
    sequence = []

    base_a = loader.synthetic_blackhole(
        npix=npix,
        fov_uas=150.0,
        mass_msun=6.5e9,
        spin=0.9,
        inclination_deg=17.0,
        noise_level=0.0,
        seed=seed
    )["stokes"]

    base_b = loader.synthetic_blackhole(
        npix=npix,
        fov_uas=150.0,
        mass_msun=6.5e9,
        spin=0.9,
        inclination_deg=17.0,
        noise_level=0.0,
        seed=seed + 1
    )["stokes"]

    for frame in range(num_frames):
        frac = frame / max(num_frames - 1, 1)
        offset = int(round(max_offset * (1.0 - frac)))

        stokes_a = np.roll(base_a, -offset, axis=1)
        stokes_b = np.roll(base_b, +offset, axis=1)

        # Mild brightness modulation to prevent perfect symmetry
        mod_a = 1.0 + 0.08 * np.cos(2 * np.pi * frac)
        mod_b = 1.0 + 0.08 * np.sin(2 * np.pi * frac)

        combined = mod_a * stokes_a + mod_b * stokes_b

        noise = rng.normal(0.0, noise_level, combined.shape)
        combined = combined + noise
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
    def mean_fill(stokes: np.ndarray, uv_mask: np.ndarray) -> np.ndarray:
        result = stokes.copy()
        for ch in range(stokes.shape[-1]):
            ch_mean = stokes[uv_mask, ch].mean()
            result[~uv_mask, ch] = ch_mean
        return result

    @staticmethod
    def radial_fill(stokes: np.ndarray, uv_mask: np.ndarray, n_bins: int = 32) -> np.ndarray:
        H, W = stokes.shape[:2]
        cy, cx = H // 2, W // 2
        Y, X = np.mgrid[0:H, 0:W]
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        theta = np.arctan2(Y - cy, X - cx)

        max_r = R.max()
        bin_edges = np.linspace(0, max_r, n_bins + 1)
        result = stokes.copy()

        for ch in range(4):
            profile = np.zeros(n_bins)
            counts = np.zeros(n_bins)

            for b in range(n_bins):
                in_bin = (R >= bin_edges[b]) & (R < bin_edges[b + 1]) & uv_mask
                if in_bin.any():
                    profile[b] = stokes[in_bin, ch].mean()
                    counts[b] = in_bin.sum()

            valid = counts > 0
            if valid.sum() >= 2:
                centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                profile = np.interp(centers, centers[valid], profile[valid])

            for b in range(n_bins):
                in_gap = (R >= bin_edges[b]) & (R < bin_edges[b + 1]) & (~uv_mask)
                if in_gap.any():
                    valid_in_bin = (R >= bin_edges[b]) & (R < bin_edges[b + 1]) & uv_mask
                    if valid_in_bin.sum() > 3:
                        angles = theta[valid_in_bin]
                        vals = stokes[valid_in_bin, ch]
                        A = vals.mean()
                        B = np.mean(vals * np.cos(angles)) / max(np.mean(np.cos(angles) ** 2), 1e-10)
                        C = np.mean(vals * np.sin(angles)) / max(np.mean(np.sin(angles) ** 2), 1e-10)
                        gap_angles = theta[in_gap]
                        result[in_gap, ch] = A + 0.3 * B * np.cos(gap_angles) + 0.3 * C * np.sin(gap_angles)
                    else:
                        result[in_gap, ch] = profile[b]

        result[..., 0] = np.clip(result[..., 0], 0.0, None)
        return result

    @staticmethod
    def smooth_interpolate(stokes: np.ndarray, uv_mask: np.ndarray, iterations: int = 20) -> np.ndarray:
        result = TemporalBaselines.radial_fill(stokes, uv_mask)
        gap = ~uv_mask
        H, W = stokes.shape[:2]

        for _ in range(iterations):
            padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="reflect")
            neighbor_sum = np.zeros_like(result)
            count = np.zeros((H, W, 1), dtype=np.float32)

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    neighbor_sum += padded[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                    count += 1.0

            avg = neighbor_sum / np.maximum(count, 1.0)
            for ch in range(4):
                result[..., ch] = np.where(gap, avg[..., ch], result[..., ch])

            result[..., 0] = np.clip(result[..., 0], 0.0, None)

        return result


# ============================================================
# Reconstruction wrappers
# ============================================================

def reconstruct_qrft_frame(
    observed: np.ndarray,
    uv_mask: np.ndarray,
    metadata: Dict,
    prev_recon: Optional[np.ndarray] = None,
    temporal_damping: float = 0.35,
    temporal_blend: float = 0.30,
    use_refiner: bool = False,
):
    encoder = Q4StokesEncoder(n_layers=1)
    sync = Q4PSSynchronizer(R_c=0.5, alpha=8.0, K=1.0)
    interpolator = FBAIInterpolator(
        encoder=encoder,
        synchronizer=sync,
        max_iterations=6 if prev_recon is None else 3,
        verbose=False
    )

    seeded = observed.copy()

    if prev_recon is not None:
        gap = ~uv_mask
        seeded[gap] = temporal_damping * prev_recon[gap] + (1.0 - temporal_damping) * seeded[gap]

    t0 = time.time()
    result = interpolator.reconstruct(seeded, uv_mask, metadata)
    recon = result.reconstructed.copy()

    if prev_recon is not None:
        low_conf = result.confidence_map < 0.4
        for ch in range(recon.shape[-1]):
            recon[..., ch] = np.where(
                low_conf,
                temporal_blend * prev_recon[..., ch] + (1.0 - temporal_blend) * recon[..., ch],
                recon[..., ch]
            )

    refine_stats = None
    if use_refiner:
        refiner = ParallelRefiner(sync, use_gpu=True)
        recon, refine_stats = refiner.refine(
            recon,
            uv_mask,
            max_iterations=20,
            global_kernel=True,
            verbose=False
        )

    dt = time.time() - t0

    return {
        "recon": recon,
        "time_s": dt,
        "result": result,
        "refine_stats": refine_stats,
    }


# ============================================================
# Main benchmark
# ============================================================

def run_phase6():
    print("\n" + "=" * 72)
    print("  QRFT BLACK HOLE TOOLKIT — PHASE 6 TEMPORAL COHERENCE BENCHMARK")
    print("  Binary Inspiral / Merger Sequence")
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
            loader,
            num_frames=20,
            npix=128,
            noise_level=config["noise"],
            seed=config["seed"]
        )

        # use one synthetic blackhole call only to get representative UV mask / metadata
        ref = loader.synthetic_blackhole(
            npix=128,
            fov_uas=150.0,
            mass_msun=6.5e9,
            spin=0.9,
            inclination_deg=17.0,
            noise_level=0.0,
            seed=config["seed"]
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

        for i, gt in enumerate(sequence):
            observed = gt.copy()
            observed[~uv_mask] = 0.0

            t0 = time.time()
            methods["Zeros"]["frames"].append(TemporalBaselines.zeros_fill(gt, uv_mask))
            methods["Zeros"]["times"].append(time.time() - t0)

            t0 = time.time()
            methods["Radial"]["frames"].append(TemporalBaselines.radial_fill(gt, uv_mask))
            methods["Radial"]["times"].append(time.time() - t0)

            t0 = time.time()
            methods["Smooth"]["frames"].append(TemporalBaselines.smooth_interpolate(gt, uv_mask, iterations=20))
            methods["Smooth"]["times"].append(time.time() - t0)

            qrft_out = reconstruct_qrft_frame(
                observed,
                uv_mask,
                metadata,
                prev_recon=prev_qrft,
                temporal_damping=0.35,
                temporal_blend=0.30,
                use_refiner=False
            )
            qrft_frame = qrft_out["recon"]
            methods["QRFT"]["frames"].append(qrft_frame)
            methods["QRFT"]["times"].append(qrft_out["time_s"])
            prev_qrft = qrft_frame

            sync = Q4PSSynchronizer(R_c=0.5, alpha=8.0, K=1.0)
            gate = CoherenceGate(sync)
            coh = sync.analyze(qrft_frame)
            gate_result = gate.evaluate(qrft_frame, mask_evaluate=(~uv_mask))
            methods["QRFT"]["coh_valid"].append(coh.stats["valid_fraction"])
            methods["QRFT"]["gate_quality"].append(gate_result.stats["mean_quality"])

            qrft_ref_out = reconstruct_qrft_frame(
                observed,
                uv_mask,
                metadata,
                prev_recon=prev_qrft_ref,
                temporal_damping=0.35,
                temporal_blend=0.30,
                use_refiner=True
            )
            qrft_ref_frame = qrft_ref_out["recon"]
            methods["QRFT+Refiner"]["frames"].append(qrft_ref_frame)
            methods["QRFT+Refiner"]["times"].append(qrft_ref_out["time_s"])
            prev_qrft_ref = qrft_ref_frame

            coh_ref = sync.analyze(qrft_ref_frame)
            gate_result_ref = gate.evaluate(qrft_ref_frame, mask_evaluate=(~uv_mask))
            methods["QRFT+Refiner"]["coh_valid"].append(coh_ref.stats["valid_fraction"])
            methods["QRFT+Refiner"]["gate_quality"].append(gate_result_ref.stats["mean_quality"])

        # -------- aggregate metrics --------
        print(f"\n  {'Method':<15} {'Mean NxCorr':>11} {'CentErr':>9} {'PeakErr':>9} {'TempJit':>9} {'EVPAΔ':>8} {'FPS':>8}")
        print(f"  {'-' * 76}")

        for method_name, bundle in methods.items():
            frames = bundle["frames"]

            per_frame_nxc = []
            per_frame_cent = []
            per_frame_peak = []
            per_frame_time = []

            gt_temporal = []
            rc_temporal = []
            temporal_jitter = []
            evpa_deltas = []

            for i in range(len(sequence)):
                gt = sequence[i]
                rc = frames[i]

                per_frame_nxc.append(nxcorr(gt[..., 0], rc[..., 0]))
                per_frame_cent.append(centroid_error(gt, rc))
                per_frame_peak.append(peak_error(gt, rc))
                per_frame_time.append(bundle["times"][i])

                if i > 0:
                    gt_temporal.append(temporal_nxcorr(sequence[i - 1], sequence[i]))
                    rc_temporal.append(temporal_nxcorr(frames[i - 1], frames[i]))

                    gt_center = center_of_mass_I(sequence[i])
                    rc_center = center_of_mass_I(frames[i])
                    gt_prev_center = center_of_mass_I(sequence[i - 1])
                    rc_prev_center = center_of_mass_I(frames[i - 1])

                    gt_step = np.sqrt((gt_center[0] - gt_prev_center[0]) ** 2 + (gt_center[1] - gt_prev_center[1]) ** 2)
                    rc_step = np.sqrt((rc_center[0] - rc_prev_center[0]) ** 2 + (rc_center[1] - rc_prev_center[1]) ** 2)
                    temporal_jitter.append(abs(rc_step - gt_step))

                    evpa_deltas.append(abs(
                        evpa_temporal_rmse(sequence[i - 1], sequence[i]) -
                        evpa_temporal_rmse(frames[i - 1], frames[i])
                    ))

            mean_nxc = float(np.mean(per_frame_nxc))
            mean_cent = float(np.mean(per_frame_cent))
            mean_peak = float(np.mean(per_frame_peak))
            mean_jit = float(np.mean(temporal_jitter)) if temporal_jitter else 0.0
            mean_evpa_delta = float(np.mean(evpa_deltas)) if evpa_deltas else 0.0
            mean_fps = 1.0 / max(float(np.mean(per_frame_time)), 1e-6)

            print(f"  {method_name:<15} {mean_nxc:>11.4f} {mean_cent:>9.3f} {mean_peak:>9.3f} {mean_jit:>9.3f} {mean_evpa_delta:>8.3f} {mean_fps:>8.2f}")

            if "QRFT" in method_name:
                print(f"    Coherence valid: {np.mean(bundle['coh_valid']) * 100:.1f}%")
                print(f"    Gate quality:    {np.mean(bundle['gate_quality']):.4f}")

        # -------- merger frame emphasis --------
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
            )

    print(f"\n{'=' * 72}")
    print("  PHASE 6 COMPLETE")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    run_phase6()
