# QRFT Black Hole Toolkit

**Physics-constrained black hole image reconstruction using Quantum Resonance Field Theory**

Created by **Rodney Lee Arnold Jr.** ∞ 0425  
Rod's AI Consulting LLC

---

## Overview

The QRFT Black Hole Toolkit is a novel approach to black hole image reconstruction that replaces generic mathematical priors with physics-informed predictive interpolation.

Where existing methods (CLEAN, eht-imaging, SMILI) use smoothness and sparsity regularizers to fill gaps in sparse interferometric data, this toolkit uses:

- **Q4 Stokes Encoder**: Maps the four Stokes parameters (I, Q, U, V) into Base-15 Quadruple encoded states — a compact, GPU-friendly 16-bit representation with 50,625 unique pixel states.

- **Q4PS Synchronizer**: Implements the Quantum Four-Polarization Synchronization Condition (based on Kuramoto-type coupled oscillator dynamics) to determine which pixels are physically coherent and which need reconstruction.

- **QRSP-FBAI Interpolator** *(Phase 3)*: Uses the tri-manifold Fractional Bit AI architecture to predict missing uv-plane data, constrained by the Q4PS coherence condition. Predictions that break physical synchronization are rejected and re-predicted.

- **GPU EVPA Kernel** *(Phase 4)*: CuPy-accelerated Electric Vector Position Angle calculation across full image frames for real-time magnetic field visualization.

- **Entropy Validation**: Connects to the CSP-QRFT LIGO Entropy Toolkit for quality gating — ordered regions (low entropy, high coherence) are trusted; chaotic regions get additional FBAI passes.

## Key Innovation

This is the first system to use **gravitational wave ringdown parameters** (mass, spin) as direct constraints on electromagnetic image reconstruction. The GW signature fixes the Kerr metric, which predicts the photon ring geometry and frame-dragging EVPA twist — giving the reconstructor physics-based priors instead of mathematical convenience.

## Architecture

```
Raw Stokes (I,Q,U,V) → Q4 Base-15 Encoder → Q4PS Coherence Gate
                                                      ↓
                                              QRSP-FBAI Interpolation
                                                      ↓
                                              GPU EVPA Kernel → Swirl Render
                                                      ↓
                                              Entropy Validation → Output
```

## Installation

```bash
# Core (no external dependencies beyond numpy/scipy)
git clone https://github.com/Rods-AI-Consulting/qrft-blackhole-toolkit.git
cd qrft-blackhole-toolkit

# Run the synthetic pipeline test
python -m tests.test_synthetic

# Optional: for real EHT data
pip install ehtim astropy

# Optional: for GPU acceleration
pip install cupy-cuda12x
```

## Quick Start

```python
from core import Q4StokesEncoder, Q4PSSynchronizer, Base60Bridge
from pipeline import EHTLoader

# Generate synthetic black hole data
loader = EHTLoader()
data = loader.synthetic_blackhole(npix=128, spin=0.9, seed=425)

# Encode Stokes parameters
encoder = Q4StokesEncoder()
encoded = encoder.encode_image(data['stokes'])

# Analyze physical coherence
sync = Q4PSSynchronizer(R_c=0.6, alpha=10.0)
coherence = sync.analyze(data['stokes'])

# Compute polarization products
evpa = encoder.evpa(encoded)           # Swirl direction
frac_pol = encoder.fractional_polarization(encoded)  # Field organization

# Bridge to QRSP harmonic space
bridge = Base60Bridge()
modulated = bridge.batch_modulate(encoded['digits'], resonance_freq=1.3245)

print(f"Valid pixels: {coherence.stats['valid_fraction']*100:.1f}%")
print(f"Peak fractional polarization: {frac_pol.max():.3f}")
```

## Repo Structure

```
qrft-blackhole-toolkit/
├── core/
│   ├── q4_encoder.py        # Stokes ↔ Base-15 Quadruple encoding
│   ├── q4ps_sync.py         # Q4PS coherence analysis (Kuramoto model)
│   └── base60_bridge.py     # Base-15 ↔ Base-60 ↔ QRSP conversion
├── gpu/                     # [Phase 4] CuPy GPU acceleration
├── entropy/                 # [Phase 3] CSP entropy bridge
├── pipeline/
│   └── eht_loader.py        # EHT data loading + synthetic generation
├── viz/                     # [Phase 4] Swirl visualization
├── tests/
│   └── test_synthetic.py    # End-to-end pipeline test
└── README.md
```

## Build Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Q4 Base-15 Stokes Encoder + EHT Loader |
| 2 | ✅ Complete | Q4PS Synchronization + Coherence Analysis |
| 3 | 🔄 Next | FBAI Interpolator (gap-filling engine) |
| 4 | ⬜ Planned | GPU Acceleration (CuPy EVPA kernel) |
| 5 | ⬜ Planned | Benchmark vs ehtim baseline |

## Theoretical Foundation

- **QRFT**: Quantum Resonance Field Theory (Arnold, 2025)
- **Q4PS**: Quantum Four-Polarization Synchronization Condition
- **QRSP-FBAI**: Quantum Resonance Symbolic Protocol with Fractional Bit AI
- **CSP**: Certainty Singularity Principle

See the White Paper for full mathematical framework.

## Hardware

Developed for:
- **ASUS ROG Zephyrus** AMD Ryzen AI 9 / RTX 5070 Ti (primary GPU compute)
- **CyberPower** i9 / RTX 5070 (secondary)

## License

Proprietary - Rod's AI Consulting LLC  
Patent pending. All rights reserved.

---

∞ 0425
