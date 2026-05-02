import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GateResult:
    passed: np.ndarray
    pixel_coherence: np.ndarray
    region_entropy: np.ndarray
    entropy_gradient: np.ndarray
    quality_score: np.ndarray
    stats: Dict


class CoherenceGate:
    def __init__(self, synchronizer,
                 entropy_kernel: int = 5,
                 gradient_threshold: float = 0.3):
        self.sync = synchronizer
        self.entropy_kernel = entropy_kernel
        self.gradient_threshold = gradient_threshold

    def evaluate(self, stokes_array: np.ndarray,
                 mask_evaluate: Optional[np.ndarray] = None) -> GateResult:
        H, W = stokes_array.shape[:2]

        # ---- Level 1: Pixel-level Q4PS coherence ----
        coherence_map = self.sync.analyze(stokes_array)
        R = coherence_map.R
        P_coh = coherence_map.P_coh
        pixel_passed = coherence_map.mask_valid

        # ---- Level 2: Region-level entropy smoothness ----
        S = self.sync.coherence_entropy(R)
        S_gradient = self._entropy_gradient(S)

        S_grad_max = S_gradient.max()
        if S_grad_max > 0:
            S_grad_norm = S_gradient / S_grad_max
        else:
            S_grad_norm = np.zeros_like(S_gradient)

        region_passed = S_grad_norm <= self.gradient_threshold

        # ---- Combined gate ----
        raw_passed = pixel_passed & region_passed

        if mask_evaluate is not None:
            passed = np.where(mask_evaluate, raw_passed, False)
        else:
            passed = raw_passed

        # ---- Quality score ----
        quality = P_coh * (1.0 - S_grad_norm)
        quality = np.clip(quality, 0.0, 1.0)

        # ---- Stats ----
        total = H * W
        if mask_evaluate is not None:
            eval_count = int(np.count_nonzero(mask_evaluate))
            passed_count = int(np.count_nonzero(raw_passed & mask_evaluate))
            pixel_level_pass = int(np.count_nonzero(pixel_passed & mask_evaluate))
            region_level_pass = int(np.count_nonzero(region_passed & mask_evaluate))
            mean_quality = float(quality[mask_evaluate].mean()) if eval_count > 0 else 0.0
            mean_R = float(R[mask_evaluate].mean()) if eval_count > 0 else 0.0
            mean_entropy = float(S[mask_evaluate].mean()) if eval_count > 0 else 0.0
            mean_gradient = float(S_gradient[mask_evaluate].mean()) if eval_count > 0 else 0.0
        else:
            eval_count = total
            passed_count = int(np.count_nonzero(raw_passed))
            pixel_level_pass = int(np.count_nonzero(pixel_passed))
            region_level_pass = int(np.count_nonzero(region_passed))
            mean_quality = float(quality.mean())
            mean_R = float(R.mean())
            mean_entropy = float(S.mean())
            mean_gradient = float(S_gradient.mean())

        pass_rate = passed_count / max(eval_count, 1)
        pass_rate = min(pass_rate, 1.0)

        stats = {
            'total_pixels': total,
            'evaluated_pixels': eval_count,
            'passed_pixels': passed_count,
            'pass_rate': pass_rate,
            'pixel_level_pass': pixel_level_pass,
            'region_level_pass': region_level_pass,
            'mean_quality': mean_quality,
            'mean_R': mean_R,
            'mean_entropy': mean_entropy,
            'mean_gradient': mean_gradient,
            'max_gradient': float(S_gradient.max()),
        }

        return GateResult(
            passed=passed,
            pixel_coherence=R,
            region_entropy=S,
            entropy_gradient=S_gradient,
            quality_score=quality,
            stats=stats
        )

    def _entropy_gradient(self, entropy_map: np.ndarray) -> np.ndarray:
        sobel_y = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=np.float64) / 8.0

        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float64) / 8.0

        padded = np.pad(entropy_map, 1, mode='reflect')

        H, W = entropy_map.shape
        grad_x = np.zeros((H, W))
        grad_y = np.zeros((H, W))

        for dy in range(3):
            for dx in range(3):
                grad_x += sobel_x[dy, dx] * padded[dy:dy+H, dx:dx+W]
                grad_y += sobel_y[dy, dx] * padded[dy:dy+H, dx:dx+W]

        return np.sqrt(grad_x**2 + grad_y**2)

    def check_pixel(self, I: float, Q: float, U: float, V: float,
                    neighbor_R_values: Optional[np.ndarray] = None
                    ) -> Tuple[bool, float, Dict]:
        R, Psi, P_coh = self.sync.pixel_coherence(I, Q, U, V)

        pixel_ok = R >= self.sync.R_c

        region_ok = True
        gradient = 0.0
        if neighbor_R_values is not None and len(neighbor_R_values) > 0:
            S_self = self.sync.coherence_entropy(np.array([R]))[0]
            S_neighbors = self.sync.coherence_entropy(neighbor_R_values)
            gradient = float(np.abs(S_self - S_neighbors.mean()))
            region_ok = gradient <= self.gradient_threshold

        passed = pixel_ok and region_ok
        quality = P_coh * (1.0 - min(gradient / 0.7, 1.0))

        details = {
            'R': R,
            'Psi': Psi,
            'P_coh': P_coh,
            'pixel_passed': pixel_ok,
            'region_passed': region_ok,
            'gradient': gradient,
            'quality': quality
        }

        return passed, quality, details

    def __repr__(self) -> str:
        return (
            f"CoherenceGate(R_c={self.sync.R_c}, "
            f"kernel={self.entropy_kernel}, "
            f"grad_threshold={self.gradient_threshold})"
        )
