"""
GPU Batch Encoder
==================
CuPy-accelerated Q4 Base-15 Quadruple encoding and decoding.

Vectorizes the entire encode/decode pipeline so a full image
gets quantized, packed, and unpacked in single GPU kernel launches.

For video: encodes 30+ frames per second at 512x512 resolution.

Falls back to numpy if CuPy is not available.

Created by: Rodney Lee Arnold Jr. ∞ 0425
"""

import numpy as np
from typing import Dict, Optional, Tuple

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False


class GPUBatchEncoder:
    """
    GPU-accelerated Q4 Stokes encoding and decoding.

    Performs normalization → quantization → bit-packing and the
    reverse in fully vectorized GPU operations.

    Usage:
        gpu_enc = GPUBatchEncoder(use_gpu=True)

        # Encode full image
        encoded = gpu_enc.encode(stokes_array)

        # Decode back
        decoded = gpu_enc.decode(encoded)

        # Encode a batch of video frames
        frames_encoded = gpu_enc.encode_batch(frame_stack)
    """

    BASE = 15
    MAX_DIGIT = 14

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

    def encode(self, stokes: np.ndarray) -> Dict:
        """
        Encode a (H, W, 4) Stokes array into Base-15 Quadruples on GPU.

        Single GPU pipeline:
            1. Compute per-channel min/max (reduction kernel)
            2. Normalize to [0, 1] (element-wise)
            3. Quantize to [0, 14] (element-wise + round)
            4. Pack to uint16 (bitshift kernel)

        Args:
            stokes: (H, W, 4) float array

        Returns:
            Dict with 'digits', 'packed', 'norm_params'
        """
        xp = self.xp

        # Transfer to device
        s = xp.asarray(stokes, dtype=xp.float64)
        H, W = s.shape[:2]

        # Per-channel stats (single reduction pass)
        flat = s.reshape(-1, 4)
        ch_min = flat.min(axis=0)
        ch_max = flat.max(axis=0)
        ch_range = ch_max - ch_min
        ch_range = xp.where(ch_range == 0, xp.float64(1.0), ch_range)

        # Normalize [0, 1]
        normalized = (s - ch_min) / ch_range

        # Quantize [0, 14]
        digits = xp.round(normalized * self.MAX_DIGIT).astype(xp.int32)
        digits = xp.clip(digits, 0, self.MAX_DIGIT)

        # Pack to uint16: [d0<<12 | d1<<8 | d2<<4 | d3]
        d = digits.astype(xp.uint16)
        packed = (d[..., 0] << 12) | (d[..., 1] << 8) | \
                 (d[..., 2] << 4) | d[..., 3]

        # Transfer norm params back to CPU for storage
        return {
            'digits': self._to_cpu(digits),
            'packed': self._to_cpu(packed),
            'norm_params': {
                'min': self._to_cpu(ch_min),
                'max': self._to_cpu(ch_max),
                'range': self._to_cpu(ch_range)
            },
            'shape': (H, W)
        }

    def decode(self, encoded: Dict) -> np.ndarray:
        """
        Decode Base-15 Quadruples back to Stokes floats on GPU.

        Args:
            encoded: Dict from encode()

        Returns:
            (H, W, 4) float array
        """
        xp = self.xp

        digits = xp.asarray(encoded['digits'], dtype=xp.float64)
        p = encoded['norm_params']
        ch_min = xp.asarray(p['min'])
        ch_range = xp.asarray(p['range'])

        # Dequantize
        normalized = digits / self.MAX_DIGIT

        # Denormalize
        result = normalized * ch_range + ch_min

        return self._to_cpu(result)

    def unpack(self, packed: np.ndarray) -> np.ndarray:
        """
        Unpack uint16 array to (H, W, 4) digits on GPU.

        Args:
            packed: (H, W) uint16 array

        Returns:
            (H, W, 4) int32 digits array
        """
        xp = self.xp
        p = xp.asarray(packed, dtype=xp.uint16)

        d0 = (p >> 12) & 0xF
        d1 = (p >> 8) & 0xF
        d2 = (p >> 4) & 0xF
        d3 = p & 0xF

        return self._to_cpu(xp.stack([d0, d1, d2, d3], axis=-1).astype(xp.int32))

    # ------------------------------------------------------------------
    # Batch Operations (for video frames)
    # ------------------------------------------------------------------

    def encode_batch(self, frames: np.ndarray) -> Dict:
        """
        Encode a batch of video frames.

        Args:
            frames: (N, H, W, 4) array of N Stokes frames

        Returns:
            Dict with:
                'digits':  (N, H, W, 4) encoded digits
                'packed':  (N, H, W) packed uint16
                'norm_params': per-frame normalization params
        """
        xp = self.xp
        N = frames.shape[0]

        all_digits = []
        all_packed = []
        all_params = []

        # Each frame gets its own normalization
        # (intensity can change between frames)
        for i in range(N):
            encoded = self.encode(frames[i])
            all_digits.append(encoded['digits'])
            all_packed.append(encoded['packed'])
            all_params.append(encoded['norm_params'])

        return {
            'digits': np.stack(all_digits),
            'packed': np.stack(all_packed),
            'norm_params': all_params,
            'n_frames': N,
            'shape': frames.shape[1:3]
        }

    def decode_batch(self, encoded_batch: Dict) -> np.ndarray:
        """
        Decode a batch of encoded video frames.

        Args:
            encoded_batch: Dict from encode_batch()

        Returns:
            (N, H, W, 4) float array
        """
        N = encoded_batch['n_frames']
        frames = []

        for i in range(N):
            single = {
                'digits': encoded_batch['digits'][i],
                'norm_params': encoded_batch['norm_params'][i],
                'shape': encoded_batch['shape']
            }
            frames.append(self.decode(single))

        return np.stack(frames)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _to_cpu(self, arr):
        if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def benchmark(self, npix: int = 128, n_runs: int = 10) -> Dict:
        """
        Benchmark encode/decode speed.

        Args:
            npix: Image size
            n_runs: Number of runs to average

        Returns:
            Dict with timing info
        """
        import time

        test_data = np.random.rand(npix, npix, 4).astype(np.float64)
        test_data[..., 0] = np.abs(test_data[..., 0])  # I >= 0

        # Warmup
        enc = self.encode(test_data)
        _ = self.decode(enc)

        # Benchmark encode
        t0 = time.time()
        for _ in range(n_runs):
            enc = self.encode(test_data)
        encode_time = (time.time() - t0) / n_runs

        # Benchmark decode
        t0 = time.time()
        for _ in range(n_runs):
            _ = self.decode(enc)
        decode_time = (time.time() - t0) / n_runs

        pixels = npix * npix
        return {
            'npix': npix,
            'encode_ms': encode_time * 1000,
            'decode_ms': decode_time * 1000,
            'encode_mpx_per_sec': pixels / encode_time / 1e6,
            'decode_mpx_per_sec': pixels / decode_time / 1e6,
            'device': 'GPU' if self.use_gpu else 'CPU',
            'fps_128': 1.0 / (encode_time + decode_time) if npix == 128 else None
        }

    def __repr__(self) -> str:
        return f"GPUBatchEncoder(gpu={self.use_gpu}, base={self.BASE})"
