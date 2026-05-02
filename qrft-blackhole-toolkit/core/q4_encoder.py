"""
Q4 Stokes Encoder
==================
Maps the four Stokes parameters (I, Q, U, V) into Base-15 Quadruple
encoded representations for use in the QRFT black hole imaging pipeline.

Theory:
    Each pixel in a polarimetric image has 4 Stokes channels:
        I = Total intensity
        Q = Horizontal/Vertical linear polarization
        U = Diagonal (±45°) linear polarization
        V = Circular polarization

    The Base-15 Quadruple encodes each channel as a digit (0-14),
    producing a compact 16-bit binary-aligned representation per pixel.
    This is a factored decomposition of the QRSP base-60 harmonic space
    since 60 = 15 × 4.

    15^4 = 50,625 unique pixel states.

Created by: Rodney Lee Arnold Jr. ∞ 0425
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass


@dataclass
class Q4PixelState:
    """
    Represents the encoded state of a single pixel.
    
    Attributes:
        digits: 4-element list of base-15 digits [I, Q, U, V]
        binary: 16-bit binary string representation
        raw_stokes: Original float Stokes values before quantization
        quantization_error: Per-channel error introduced by quantization
    """
    digits: List[int]
    binary: str
    raw_stokes: np.ndarray
    quantization_error: np.ndarray


class Q4StokesEncoder:
    """
    Encodes and decodes Stokes (I, Q, U, V) parameters using
    the Base-15 Quadruple system.

    The encoder handles:
        - Normalization of raw Stokes data to [0, 14] range
        - Quantization to 15 discrete levels per channel
        - Binary packing for GPU-friendly 16-bit representation
        - Lossless decode back to normalized float space
        - Multi-layer encoding for high dynamic range regions

    Usage:
        encoder = Q4StokesEncoder()
        
        # Single pixel
        pixel = encoder.encode_pixel(I=1.0, Q=0.3, U=-0.2, V=0.01)
        
        # Full image (H x W x 4 Stokes array)
        encoded = encoder.encode_image(stokes_array)
        decoded = encoder.decode_image(encoded)
    """

    BASE = 15
    DIGITS = 4
    MAX_DIGIT = 14
    MAX_VALUE = 15 ** 4  # 50,625 unique states
    CHANNEL_NAMES = ['I', 'Q', 'U', 'V']

    def __init__(self, n_layers: int = 1, dynamic_range_db: float = 60.0):
        """
        Initialize the Q4 Stokes Encoder.

        Args:
            n_layers: Number of encoding layers. Use >1 for high dynamic
                      range regions (photon ring). Each layer encodes the
                      residual from the previous layer.
            dynamic_range_db: Expected dynamic range of the input data in dB.
                              Used to set quantization scaling.
        """
        self.n_layers = n_layers
        self.dynamic_range_db = dynamic_range_db

        # Normalization parameters (set during encode, needed for decode)
        self._norm_params: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _compute_norm_params(self, stokes: np.ndarray) -> Dict:
        """
        Compute per-channel min/max for normalization.

        Stokes I is always >= 0 (total intensity).
        Stokes Q, U, V can be negative.

        Args:
            stokes: Array of shape (..., 4) with channels [I, Q, U, V]

        Returns:
            Dict with 'min', 'max', 'range' arrays of shape (4,)
        """
        # Reshape to (N, 4) for easy per-channel stats
        flat = stokes.reshape(-1, 4).astype(np.float64)

        ch_min = flat.min(axis=0)
        ch_max = flat.max(axis=0)
        ch_range = ch_max - ch_min

        # Avoid division by zero for constant channels
        ch_range = np.where(ch_range == 0, 1.0, ch_range)

        return {
            'min': ch_min,
            'max': ch_max,
            'range': ch_range
        }

    def _normalize(self, stokes: np.ndarray) -> np.ndarray:
        """
        Normalize Stokes values to [0, 1] range per channel.

        Args:
            stokes: Array of shape (..., 4)

        Returns:
            Normalized array of same shape, values in [0, 1]
        """
        p = self._norm_params
        return (stokes - p['min']) / p['range']

    def _denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """
        Reverse normalization back to original Stokes scale.

        Args:
            normalized: Array of shape (..., 4), values in [0, 1]

        Returns:
            Array in original Stokes scale
        """
        p = self._norm_params
        return normalized * p['range'] + p['min']

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def _quantize(self, normalized: np.ndarray) -> np.ndarray:
        """
        Quantize [0, 1] values to integer digits in [0, 14].

        Args:
            normalized: Float array with values in [0, 1]

        Returns:
            Integer array with values in [0, 14]
        """
        # Scale to [0, 14] and round
        quantized = np.round(normalized * self.MAX_DIGIT).astype(np.int32)
        # Clamp to valid range
        return np.clip(quantized, 0, self.MAX_DIGIT)

    def _dequantize(self, digits: np.ndarray) -> np.ndarray:
        """
        Convert base-15 digits back to [0, 1] float range.

        Args:
            digits: Integer array with values in [0, 14]

        Returns:
            Float array with values in [0, 1]
        """
        return digits.astype(np.float64) / self.MAX_DIGIT

    # ------------------------------------------------------------------
    # Binary Packing (GPU-friendly 16-bit representation)
    # ------------------------------------------------------------------

    def _pack_to_uint16(self, digits: np.ndarray) -> np.ndarray:
        """
        Pack 4 base-15 digits into a single 16-bit integer.

        Each digit occupies 4 bits:
            bits [15:12] = digit 0 (Stokes I)
            bits [11:8]  = digit 1 (Stokes Q)
            bits [7:4]   = digit 2 (Stokes U)
            bits [3:0]   = digit 3 (Stokes V)

        Args:
            digits: Array of shape (..., 4) with int values [0, 14]

        Returns:
            Array of shape (...) with uint16 packed values
        """
        d = digits.astype(np.uint16)
        packed = (d[..., 0] << 12) | (d[..., 1] << 8) | \
                 (d[..., 2] << 4) | d[..., 3]
        return packed

    def _unpack_from_uint16(self, packed: np.ndarray) -> np.ndarray:
        """
        Unpack a 16-bit integer back to 4 base-15 digits.

        Args:
            packed: Array of uint16 values

        Returns:
            Array of shape (..., 4) with int values [0, 14]
        """
        d0 = (packed >> 12) & 0xF
        d1 = (packed >> 8) & 0xF
        d2 = (packed >> 4) & 0xF
        d3 = packed & 0xF
        return np.stack([d0, d1, d2, d3], axis=-1).astype(np.int32)

    # ------------------------------------------------------------------
    # Binary String Representation
    # ------------------------------------------------------------------

    @staticmethod
    def digits_to_binary_str(digits: np.ndarray) -> str:
        """
        Convert 4 digits to a 16-bit binary string.

        Args:
            digits: 1D array of 4 integers in [0, 14]

        Returns:
            16-character binary string
        """
        return ''.join(f'{d:04b}' for d in digits)

    @staticmethod
    def binary_str_to_digits(binary: str) -> np.ndarray:
        """
        Convert 16-bit binary string to 4 base-15 digits.

        Args:
            binary: 16-character binary string

        Returns:
            1D array of 4 integers
        """
        if len(binary) != 16:
            raise ValueError(f"Expected 16-bit string, got {len(binary)}")
        return np.array([int(binary[i:i+4], 2) for i in range(0, 16, 4)])

    # ------------------------------------------------------------------
    # Single Pixel Encoding/Decoding
    # ------------------------------------------------------------------

    def encode_pixel(self, I: float, Q: float, U: float, V: float,
                     norm_params: Optional[Dict] = None) -> Q4PixelState:
        """
        Encode a single pixel's Stokes parameters.

        Args:
            I, Q, U, V: Raw Stokes parameter values
            norm_params: Pre-computed normalization params (if encoding
                         as part of a larger image, pass these in)

        Returns:
            Q4PixelState with encoded digits, binary, and error info
        """
        raw = np.array([I, Q, U, V], dtype=np.float64)

        # Use provided norm params or compute from this pixel alone
        if norm_params is not None:
            self._norm_params = norm_params
        elif self._norm_params is None:
            # Single pixel: just map to [0, 14] directly
            self._norm_params = self._compute_norm_params(raw.reshape(1, 4))

        normalized = self._normalize(raw)
        digits = self._quantize(normalized)

        # Compute quantization error
        reconstructed = self._denormalize(self._dequantize(digits))
        error = np.abs(raw - reconstructed)

        binary = self.digits_to_binary_str(digits)

        return Q4PixelState(
            digits=digits.tolist(),
            binary=binary,
            raw_stokes=raw,
            quantization_error=error
        )

    # ------------------------------------------------------------------
    # Full Image Encoding/Decoding
    # ------------------------------------------------------------------

    def encode_image(self, stokes_array: np.ndarray) -> Dict:
        """
        Encode a full Stokes image into Base-15 Quadruple representation.

        Args:
            stokes_array: Array of shape (H, W, 4) where the 4 channels
                          are [I, Q, U, V] in that order.

        Returns:
            Dict containing:
                'digits':     (H, W, 4) int array of base-15 digits
                'packed':     (H, W) uint16 array of packed pixels
                'norm_params': normalization parameters for decoding
                'shape':      original spatial shape (H, W)
                'n_layers':   number of encoding layers used
                'layers':     list of per-layer digit arrays (for multi-layer)
                'residuals':  quantization residual per layer
        """
        stokes = stokes_array.astype(np.float64)
        H, W = stokes.shape[:2]

        # Compute normalization from full image
        self._norm_params = self._compute_norm_params(stokes)

        layers = []
        residuals = []
        current = stokes.copy()

        for layer_idx in range(self.n_layers):
            # Normalize current data
            if layer_idx > 0:
                # Re-compute norm params for residual layer
                layer_norm = self._compute_norm_params(current)
                self._norm_params = layer_norm

            normalized = self._normalize(current)
            digits = self._quantize(normalized)

            # Compute residual for next layer
            reconstructed_norm = self._dequantize(digits)
            reconstructed = self._denormalize(reconstructed_norm)
            residual = current - reconstructed

            layers.append({
                'digits': digits,
                'packed': self._pack_to_uint16(digits),
                'norm_params': {k: v.copy() for k, v in self._norm_params.items()}
            })
            residuals.append(residual)

            # Next layer encodes the residual
            current = residual

        # Primary layer is layer 0
        self._norm_params = layers[0]['norm_params']

        return {
            'digits': layers[0]['digits'],
            'packed': layers[0]['packed'],
            'norm_params': layers[0]['norm_params'],
            'shape': (H, W),
            'n_layers': self.n_layers,
            'layers': layers,
            'residuals': residuals
        }

    def decode_image(self, encoded: Dict) -> np.ndarray:
        """
        Decode a Base-15 Quadruple encoded image back to Stokes floats.

        Reconstructs across all layers if multi-layer encoding was used.

        Args:
            encoded: Dict from encode_image()

        Returns:
            Array of shape (H, W, 4) with reconstructed Stokes values
        """
        H, W = encoded['shape']
        result = np.zeros((H, W, 4), dtype=np.float64)

        for layer_data in encoded['layers']:
            self._norm_params = layer_data['norm_params']
            digits = layer_data['digits']
            normalized = self._dequantize(digits)
            reconstructed = self._denormalize(normalized)
            result += reconstructed

        return result

    # ------------------------------------------------------------------
    # Convenience: Extract individual Stokes channels from encoded data
    # ------------------------------------------------------------------

    def get_stokes_digits(self, encoded: Dict, channel: str) -> np.ndarray:
        """
        Extract a single Stokes channel's digit array.

        Args:
            encoded: Dict from encode_image()
            channel: One of 'I', 'Q', 'U', 'V'

        Returns:
            (H, W) array of base-15 digits for that channel
        """
        idx = self.CHANNEL_NAMES.index(channel.upper())
        return encoded['digits'][..., idx]

    # ------------------------------------------------------------------
    # Polarization Calculations (operate on encoded data)
    # ------------------------------------------------------------------

    def fractional_polarization(self, encoded: Dict) -> np.ndarray:
        """
        Calculate fractional polarization from encoded data.

        m = sqrt(Q^2 + U^2) / I

        Operates in decoded float space for accuracy.

        Args:
            encoded: Dict from encode_image()

        Returns:
            (H, W) array of fractional polarization values [0, 1]
        """
        decoded = self.decode_image(encoded)
        I = decoded[..., 0]
        Q = decoded[..., 1]
        U = decoded[..., 2]

        # Avoid division by zero
        I_safe = np.where(I > 0, I, 1e-10)
        m = np.sqrt(Q**2 + U**2) / I_safe

        return np.clip(m, 0.0, 1.0)

    def evpa(self, encoded: Dict) -> np.ndarray:
        """
        Calculate Electric Vector Position Angle from encoded data.

        chi = 0.5 * arctan2(U, Q)

        This is the "swirl" direction at each pixel.

        Args:
            encoded: Dict from encode_image()

        Returns:
            (H, W) array of EVPA values in radians [-pi/2, pi/2]
        """
        decoded = self.decode_image(encoded)
        Q = decoded[..., 1]
        U = decoded[..., 2]

        return 0.5 * np.arctan2(U, Q)

    def linear_polarization_intensity(self, encoded: Dict) -> np.ndarray:
        """
        Calculate linear polarization intensity.

        LP = sqrt(Q^2 + U^2)

        This is the "strength" of the swirl at each pixel.

        Args:
            encoded: Dict from encode_image()

        Returns:
            (H, W) array of linear polarization intensity
        """
        decoded = self.decode_image(encoded)
        Q = decoded[..., 1]
        U = decoded[..., 2]

        return np.sqrt(Q**2 + U**2)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def quantization_report(self, stokes_array: np.ndarray) -> Dict:
        """
        Run encoding and report quantization quality metrics.

        Args:
            stokes_array: (H, W, 4) Stokes array

        Returns:
            Dict with per-channel RMSE, max error, SNR, and compression info
        """
        encoded = self.encode_image(stokes_array)
        decoded = self.decode_image(encoded)

        error = stokes_array - decoded
        report = {}

        for i, name in enumerate(self.CHANNEL_NAMES):
            ch_err = error[..., i]
            ch_orig = stokes_array[..., i]
            rmse = np.sqrt(np.mean(ch_err**2))
            max_err = np.abs(ch_err).max()

            # Signal-to-quantization-noise ratio
            signal_power = np.mean(ch_orig**2)
            noise_power = np.mean(ch_err**2)
            sqnr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf

            report[name] = {
                'rmse': rmse,
                'max_error': max_err,
                'sqnr_db': sqnr
            }

        report['total_pixels'] = stokes_array.shape[0] * stokes_array.shape[1]
        report['bits_per_pixel'] = 16 * self.n_layers
        report['unique_states'] = self.MAX_VALUE
        report['compression_ratio'] = (stokes_array.nbytes) / \
            (encoded['packed'].nbytes * self.n_layers)

        return report

    def __repr__(self) -> str:
        return (
            f"Q4StokesEncoder(n_layers={self.n_layers}, "
            f"dynamic_range={self.dynamic_range_db}dB, "
            f"states={self.MAX_VALUE:,})"
        )
