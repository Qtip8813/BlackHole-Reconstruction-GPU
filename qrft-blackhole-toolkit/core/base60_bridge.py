"""
Base-60 Bridge
===============
Converts between Base-15 Quadruple encoded pixel states and the
QRSP Base-60 harmonic representation.

Key relationship: 60 = 15 × 4
    A Base-15 Quadruple is a factored decomposition of Base-60 space.
    This bridge allows the Q4 encoder output to feed directly into
    the QRSP resonance analysis and FBAI interpolation engine.

Created by: Rodney Lee Arnold Jr. ∞ 0425
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import math


class Base60Bridge:
    """
    Bidirectional conversion between Base-15 Quadruples and Base-60
    harmonic representations.

    The bridge enables:
        - Q4 encoded pixels → Base-60 harmonic coefficients
        - Base-60 harmonic coefficients → Q4 encoded pixels
        - Quantum modulation of encoded data for QRSP analysis
        - Layered base-60 decomposition for multi-scale analysis

    Usage:
        bridge = Base60Bridge()
        
        # Quadruple to base-60
        b60 = bridge.quad_to_base60([7, 3, 11, 2])
        
        # Base-60 to quadruple
        quad = bridge.base60_to_quad(b60_digits)
        
        # Quantum modulation for QRSP
        modulated = bridge.quantum_modulate(quad, resonance_freq=1.3245)
    """

    BASE_15 = 15
    BASE_60 = 60
    QUAD_DIGITS = 4

    # QRSP Solfeggio frequency mapping (Hz)
    SOLFEGGIO = {
        0: 174.0,   # Foundation
        1: 285.0,   # Quantum cognition
        2: 396.0,   # Liberation
        3: 417.0,   # Transformation
        4: 432.0,   # Cosmic tuning (A=432)
        5: 528.0,   # DNA repair / Love frequency
        6: 639.0,   # Connection
        7: 741.0,   # Expression
        8: 852.0,   # Intuition
        9: 963.0,   # Crown / Higher order
    }

    # QRFT parameters
    DELTA = 0.425      # δ parameter
    HARMONIC_LOCK = 2.0712  # 0425 harmonic lock

    def __init__(self):
        """Initialize the Base-60 Bridge."""
        pass

    # ------------------------------------------------------------------
    # Core Conversion: Base-15 Quad ↔ Base-60
    # ------------------------------------------------------------------

    def quad_to_base60(self, quad_digits: List[int]) -> List[int]:
        """
        Convert a Base-15 Quadruple (4 digits) to Base-60 representation.

        The conversion preserves the integer value:
            value = d0*15^3 + d1*15^2 + d2*15 + d3
        Then re-encodes in base-60:
            value = b0*60^n + ... + b_last

        Args:
            quad_digits: List of 4 integers in [0, 14]

        Returns:
            List of base-60 digits (variable length)
        """
        if len(quad_digits) != self.QUAD_DIGITS:
            raise ValueError(f"Expected 4 digits, got {len(quad_digits)}")
        if not all(0 <= d <= 14 for d in quad_digits):
            raise ValueError(f"All digits must be in [0, 14]")

        # Decode to integer
        value = 0
        for d in quad_digits:
            value = value * self.BASE_15 + d

        # Encode to base-60
        if value == 0:
            return [0]

        b60_digits = []
        temp = value
        while temp > 0:
            b60_digits.append(temp % self.BASE_60)
            temp //= self.BASE_60

        return b60_digits[::-1]

    def base60_to_quad(self, b60_digits: List[int]) -> List[int]:
        """
        Convert Base-60 digits back to a Base-15 Quadruple.

        Args:
            b60_digits: List of base-60 digits

        Returns:
            List of 4 base-15 digits

        Raises:
            ValueError if value exceeds 15^4 = 50,625
        """
        # Decode to integer
        value = 0
        for d in b60_digits:
            value = value * self.BASE_60 + d

        if value >= self.BASE_15 ** self.QUAD_DIGITS:
            raise ValueError(
                f"Value {value} exceeds max Base-15 Quad "
                f"({self.BASE_15 ** self.QUAD_DIGITS})"
            )

        # Encode to base-15 (4 digits)
        digits = []
        temp = value
        for _ in range(self.QUAD_DIGITS):
            digits.append(temp % self.BASE_15)
            temp //= self.BASE_15

        return digits[::-1]

    # ------------------------------------------------------------------
    # Integer Value Access
    # ------------------------------------------------------------------

    @staticmethod
    def quad_to_int(quad_digits: List[int]) -> int:
        """Convert Base-15 Quadruple to integer value."""
        value = 0
        for d in quad_digits:
            value = value * 15 + d
        return value

    @staticmethod
    def int_to_quad(value: int) -> List[int]:
        """Convert integer to Base-15 Quadruple."""
        if not 0 <= value < 50625:
            raise ValueError(f"Value must be in [0, 50624], got {value}")
        digits = []
        temp = value
        for _ in range(4):
            digits.append(temp % 15)
            temp //= 15
        return digits[::-1]

    # ------------------------------------------------------------------
    # Quantum Modulation (QRSP)
    # ------------------------------------------------------------------

    def quantum_modulate(self, quad_digits: List[int],
                          resonance_freq: float) -> List[float]:
        """
        Apply QRSP quantum modulation to a Base-15 Quadruple.

        Each digit is modulated with amplitude and phase based on
        its position within the base-60 harmonic space.

        This produces a continuous-valued "resonance signature" that
        the FBAI system can operate on.

        Args:
            quad_digits: List of 4 base-15 digits
            resonance_freq: Modulation frequency (Hz or normalized)

        Returns:
            List of 4 modulated float values
        """
        output = []
        for i, d in enumerate(quad_digits):
            # Map digit to [0, 1] range within base-60 context
            # 15 * 4 = 60, so digit/15 is the fractional position
            # within one base-60 "slot"
            amplitude = 0.5 + 0.5 * math.sin(2 * math.pi * d / self.BASE_15)
            phase = (d * 2 * math.pi / self.BASE_15) % (2 * math.pi)
            value = amplitude * math.cos(phase + i * resonance_freq)
            output.append(value)
        return output

    def harmonic_signature(self, quad_digits: List[int]) -> Dict:
        """
        Compute the full harmonic signature of a pixel state.

        This maps the Base-15 Quadruple through the Solfeggio
        frequency table and computes resonance properties.

        Args:
            quad_digits: List of 4 base-15 digits

        Returns:
            Dict with frequency mapping, resonance score, and
            harmonic lock distance
        """
        # Map each digit to nearest Solfeggio frequency
        # Digits 0-14 map to Solfeggio indices 0-9 (with wrapping)
        freq_map = []
        for d in quad_digits:
            solf_idx = d % 10  # Map to Solfeggio index
            freq = self.SOLFEGGIO[solf_idx]
            freq_map.append(freq)

        # Resonance score: how close to harmonic ratios
        ratios = []
        for i in range(len(freq_map)):
            for j in range(i + 1, len(freq_map)):
                ratio = max(freq_map[i], freq_map[j]) / \
                        min(freq_map[i], freq_map[j])
                ratios.append(ratio)

        # Distance from known harmonic ratios (octave, fifth, fourth)
        harmonic_targets = [2.0, 1.5, 4/3, 3/2, 5/4]
        min_distances = []
        for r in ratios:
            distances = [abs(r - t) for t in harmonic_targets]
            min_distances.append(min(distances))

        resonance_score = 1.0 / (1.0 + np.mean(min_distances))

        # Distance from 0425 harmonic lock
        b60_digits = self.quad_to_base60(quad_digits)
        b60_value = sum(d * (60 ** (len(b60_digits) - 1 - i))
                        for i, d in enumerate(b60_digits))
        lock_distance = abs(b60_value - self.HARMONIC_LOCK)

        return {
            'frequencies': freq_map,
            'frequency_ratios': ratios,
            'resonance_score': resonance_score,
            'base60_digits': b60_digits,
            'harmonic_lock_distance': lock_distance,
            'delta': self.DELTA
        }

    # ------------------------------------------------------------------
    # Layered Base-60 Decomposition
    # ------------------------------------------------------------------

    def layered_decomposition(self, quad_digits: List[int],
                               n_layers: int = 4,
                               digits_per_layer: int = 2) -> List[List[int]]:
        """
        Decompose a Base-15 Quadruple into layered Base-60 representation.

        This enables multi-scale analysis where each layer captures
        a different precision level:
            Layer 0: Coarse structure (broad features)
            Layer 1: Mid-scale features
            Layer 2: Fine detail
            Layer 3: Sub-pixel precision

        Args:
            quad_digits: List of 4 base-15 digits
            n_layers: Number of decomposition layers
            digits_per_layer: Base-60 digits per layer

        Returns:
            List of layers, each a list of base-60 digits
        """
        # Convert to float for fractional base-60 expansion
        value = self.quad_to_int(quad_digits)
        normalized = value / (self.BASE_15 ** self.QUAD_DIGITS)

        total_digits = n_layers * digits_per_layer
        b60_expanded = []

        frac = normalized
        for _ in range(total_digits):
            frac *= self.BASE_60
            digit = int(frac)
            b60_expanded.append(min(digit, 59))
            frac -= digit

        # Split into layers
        layers = []
        for i in range(n_layers):
            start = i * digits_per_layer
            end = start + digits_per_layer
            layers.append(b60_expanded[start:end])

        return layers

    # ------------------------------------------------------------------
    # Batch Operations (for full images)
    # ------------------------------------------------------------------

    def batch_quad_to_base60(self, digits_array: np.ndarray) -> np.ndarray:
        """
        Convert an array of Base-15 Quadruples to integer values.

        For image processing, it's more efficient to work with
        integer values than variable-length base-60 digit lists.

        Args:
            digits_array: (..., 4) array of base-15 digits

        Returns:
            (...) array of integer values [0, 50624]
        """
        d = digits_array.astype(np.int64)
        return (d[..., 0] * 3375 + d[..., 1] * 225 +
                d[..., 2] * 15 + d[..., 3])

    def batch_modulate(self, digits_array: np.ndarray,
                        resonance_freq: float) -> np.ndarray:
        """
        Vectorized quantum modulation across a full image.

        Args:
            digits_array: (H, W, 4) array of base-15 digits
            resonance_freq: Modulation frequency

        Returns:
            (H, W, 4) array of modulated float values
        """
        d = digits_array.astype(np.float64)

        amplitude = 0.5 + 0.5 * np.sin(2 * np.pi * d / self.BASE_15)
        phase = (d * 2 * np.pi / self.BASE_15) % (2 * np.pi)

        # Channel index broadcast
        idx = np.arange(4).reshape(1, 1, 4)
        modulated = amplitude * np.cos(phase + idx * resonance_freq)

        return modulated

    def __repr__(self) -> str:
        return f"Base60Bridge(delta={self.DELTA}, lock={self.HARMONIC_LOCK})"
