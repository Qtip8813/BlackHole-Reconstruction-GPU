"""
EHT Data Loader
=================
Handles loading EHT observation data and extracting Stokes parameters
in formats ready for the Q4 encoding pipeline.

Supports:
    - Real EHT .uvfits files (requires ehtim)
    - Synthetic test data generation (no dependencies)
    - FITS image files (requires astropy)

The loader bridges between standard astrophysics data formats
and the Q4StokesEncoder input format (H, W, 4) arrays.

Created by: Rodney Lee Arnold Jr. ∞ 0425
"""

import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path


class EHTLoader:
    """
    Load and prepare polarimetric data for the Q4 pipeline.

    Usage:
        loader = EHTLoader()

        # Generate synthetic black hole data for testing
        stokes = loader.synthetic_blackhole(npix=128)

        # Load real EHT data (requires ehtim installed)
        stokes = loader.load_uvfits('m87_2017.uvfits', npix=128)

        # Load from FITS image
        stokes = loader.load_fits_image('m87_image.fits')
    """

    # Physical constants
    RADPERUAS = np.pi / (180.0 * 3600.0 * 1e6)  # radians per microarcsecond

    def __init__(self):
        """Initialize the EHT Loader."""
        self._ehtim_available = False
        self._astropy_available = False

        try:
            import ehtim
            self._ehtim_available = True
        except ImportError:
            pass

        try:
            from astropy.io import fits
            self._astropy_available = True
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Synthetic Data Generation (no dependencies needed)
    # ------------------------------------------------------------------

    def synthetic_blackhole(self, npix: int = 128,
                             fov_uas: float = 150.0,
                             mass_msun: float = 6.5e9,
                             spin: float = 0.9,
                             inclination_deg: float = 17.0,
                             noise_level: float = 0.02,
                             seed: Optional[int] = None) -> Dict:
        """
        Generate synthetic black hole Stokes data.

        Creates a physically-motivated model with:
            - Photon ring at the expected shadow radius
            - Asymmetric brightness (Doppler boosting)
            - Coherent EVPA spiral pattern from frame dragging
            - Configurable noise level

        This is NOT a full GRMHD simulation — it's a fast analytic
        model for testing the pipeline. For real science, use actual
        EHT data or GRMHD snapshots.

        Args:
            npix:            Image size in pixels
            fov_uas:         Field of view in microarcseconds
            mass_msun:       Black hole mass in solar masses
            spin:            Dimensionless spin parameter [0, 1)
            inclination_deg: Observer inclination in degrees
            noise_level:     Gaussian noise std as fraction of peak
            seed:            Random seed for reproducibility

        Returns:
            Dict with:
                'stokes':     (npix, npix, 4) array [I, Q, U, V]
                'metadata':   Physical parameters used
                'uv_mask':    (npix, npix) simulated uv-coverage mask
        """
        if seed is not None:
            np.random.seed(seed)

        inc = np.radians(inclination_deg)
        half_fov = fov_uas / 2.0

        # Coordinate grid in microarcseconds
        x = np.linspace(-half_fov, half_fov, npix)
        y = np.linspace(-half_fov, half_fov, npix)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)

        # ----------------------------------------------------------
        # Stokes I: Ring model with Doppler asymmetry
        # ----------------------------------------------------------
        # Shadow radius ~ 5.2 * G*M/c^2 in angular size
        # For M87*: ~21 microarcseconds radius
        shadow_radius = 21.0  # uas (approximate for M87*)

        # Gaussian ring
        ring_width = 5.0  # uas
        ring = np.exp(-0.5 * ((R - shadow_radius) / ring_width)**2)

        # Doppler boosting: approaching side brighter
        doppler = 1.0 + 0.4 * spin * np.sin(inc) * np.cos(theta)
        I = ring * doppler

        # Normalize peak to 1
        I = I / (I.max() + 1e-10)

        # ----------------------------------------------------------
        # Stokes Q and U: Coherent EVPA from magnetic field
        # ----------------------------------------------------------
        # Model: predominantly toroidal magnetic field
        # EVPA follows the azimuthal direction with frame-dragging twist
        evpa_angle = theta + spin * 0.3 * np.exp(-R / shadow_radius)

        # Fractional polarization: ~15-20% near the ring, less elsewhere
        frac_pol = 0.18 * ring

        # Q = m * I * cos(2*chi), U = m * I * sin(2*chi)
        Q = frac_pol * I * np.cos(2 * evpa_angle)
        U = frac_pol * I * np.sin(2 * evpa_angle)

        # ----------------------------------------------------------
        # Stokes V: Circular polarization (weak)
        # ----------------------------------------------------------
        # Typically very small for synchrotron; model ~1% of I
        V = 0.01 * I * np.sin(theta + spin * 0.5)

        # ----------------------------------------------------------
        # Add noise
        # ----------------------------------------------------------
        noise_std = noise_level * I.max()
        I += np.random.normal(0, noise_std, I.shape)
        Q += np.random.normal(0, noise_std * 0.5, Q.shape)
        U += np.random.normal(0, noise_std * 0.5, U.shape)
        V += np.random.normal(0, noise_std * 0.1, V.shape)

        # Ensure I >= 0
        I = np.maximum(I, 0)

        # Stack into (H, W, 4)
        stokes = np.stack([I, Q, U, V], axis=-1)

        # ----------------------------------------------------------
        # Simulated uv-coverage mask (sparse like real EHT)
        # ----------------------------------------------------------
        uv_mask = self._generate_eht_uv_mask(npix)

        metadata = {
            'npix': npix,
            'fov_uas': fov_uas,
            'mass_msun': mass_msun,
            'spin': spin,
            'inclination_deg': inclination_deg,
            'shadow_radius_uas': shadow_radius,
            'noise_level': noise_level,
            'peak_fractional_pol': float(frac_pol.max()),
            'synthetic': True,
            'seed': seed
        }

        return {
            'stokes': stokes,
            'metadata': metadata,
            'uv_mask': uv_mask
        }

    def _generate_eht_uv_mask(self, npix: int) -> np.ndarray:
        """
        Generate a sparse uv-coverage mask simulating EHT baselines.

        Real EHT has ~7-8 stations producing ~28 baselines, leading
        to very sparse uv-plane coverage. This function creates a
        realistic mask where ~15-20% of the uv-plane is sampled.

        Args:
            npix: Image/uv-plane size

        Returns:
            (npix, npix) boolean mask, True where sampled
        """
        mask = np.zeros((npix, npix), dtype=bool)

        u = np.linspace(-1, 1, npix)
        v = np.linspace(-1, 1, npix)
        U, V = np.meshgrid(u, v)
        UV_r = np.sqrt(U**2 + V**2)

        # Simulate baseline tracks (arcs in uv-plane from Earth rotation)
        n_baselines = 28
        for _ in range(n_baselines):
            # Random baseline parameters
            r0 = np.random.uniform(0.1, 0.95)
            angle0 = np.random.uniform(0, 2 * np.pi)
            arc_length = np.random.uniform(0.3, 1.2)  # radians of rotation
            width = 0.03

            for t in np.linspace(0, arc_length, 50):
                cx = r0 * np.cos(angle0 + t)
                cy = r0 * np.sin(angle0 + t)
                dist = np.sqrt((U - cx)**2 + (V - cy)**2)
                mask |= (dist < width)

        # Also add conjugate baselines (uv symmetry)
        mask |= mask[::-1, ::-1]

        return mask

    # ------------------------------------------------------------------
    # Real EHT Data Loading (requires ehtim)
    # ------------------------------------------------------------------

    def load_uvfits(self, filepath: str, npix: int = 128,
                     fov_uas: float = 150.0) -> Dict:
        """
        Load real EHT observation data from a .uvfits file.

        Requires the ehtim library to be installed:
            pip install ehtim

        Args:
            filepath:  Path to .uvfits file
            npix:      Reconstruction grid size
            fov_uas:   Field of view in microarcseconds

        Returns:
            Dict with:
                'stokes':   (npix, npix, 4) array [I, Q, U, V]
                'obs':      ehtim Obsdata object (for advanced use)
                'metadata': Observation metadata
        """
        if not self._ehtim_available:
            raise ImportError(
                "ehtim is required for loading .uvfits files.\n"
                "Install with: pip install ehtim"
            )

        import ehtim as eh

        # Load observation
        obs = eh.obsdata.load_uvfits(filepath)
        fov_rad = fov_uas * self.RADPERUAS

        # Create prior image for reconstruction
        prior = eh.image.make_square(obs, npix=npix, fov=fov_rad)

        # Reconstruct Stokes I
        imgr = eh.imager.Imager(
            obs, prior,
            data_term={'vis': 1.0},
            reg_term={'simple': 1.0}
        )
        imgr.make_image_I()
        img = imgr.out_last()

        # Extract Stokes arrays
        I = img.imarr()
        Q = img.qarr() if hasattr(img, 'qarr') else np.zeros_like(I)
        U = img.uarr() if hasattr(img, 'uarr') else np.zeros_like(I)
        V = img.varr() if hasattr(img, 'varr') else np.zeros_like(I)

        stokes = np.stack([I, Q, U, V], axis=-1)

        metadata = {
            'source': obs.source,
            'n_data_points': len(obs.data),
            'telescopes': list(obs.tarr['site']),
            'npix': npix,
            'fov_uas': fov_uas,
            'synthetic': False,
            'filepath': str(filepath)
        }

        return {
            'stokes': stokes,
            'obs': obs,
            'metadata': metadata
        }

    # ------------------------------------------------------------------
    # FITS Image Loading (requires astropy)
    # ------------------------------------------------------------------

    def load_fits_image(self, filepath: str) -> Dict:
        """
        Load Stokes data from a FITS image file.

        Expects a FITS file with Stokes I, Q, U, V as separate
        extensions or a 4D data cube.

        Args:
            filepath: Path to FITS file

        Returns:
            Dict with 'stokes' array and 'metadata'
        """
        if not self._astropy_available:
            raise ImportError(
                "astropy is required for loading FITS files.\n"
                "Install with: pip install astropy"
            )

        from astropy.io import fits

        with fits.open(filepath) as hdul:
            header = hdul[0].header
            data = hdul[0].data

            # Handle different FITS conventions
            if data.ndim == 4:
                # (Stokes, Freq, Dec, RA) — standard radio FITS
                I = data[0, 0]
                Q = data[1, 0] if data.shape[0] > 1 else np.zeros_like(I)
                U = data[2, 0] if data.shape[0] > 2 else np.zeros_like(I)
                V = data[3, 0] if data.shape[0] > 3 else np.zeros_like(I)
            elif data.ndim == 2:
                # Single Stokes I image
                I = data
                Q = np.zeros_like(I)
                U = np.zeros_like(I)
                V = np.zeros_like(I)
            else:
                raise ValueError(
                    f"Unexpected FITS data shape: {data.shape}")

            stokes = np.stack([I, Q, U, V], axis=-1)

            metadata = {
                'filepath': str(filepath),
                'shape': data.shape,
                'npix': I.shape[0],
                'synthetic': False
            }

            # Extract WCS info if available
            for key in ['CDELT1', 'CDELT2', 'CRPIX1', 'CRPIX2',
                        'TELESCOP', 'OBJECT']:
                if key in header:
                    metadata[key] = header[key]

        return {
            'stokes': stokes,
            'metadata': metadata
        }

    # ------------------------------------------------------------------
    # Info / Diagnostics
    # ------------------------------------------------------------------

    def describe(self, data: Dict) -> str:
        """
        Print a human-readable summary of loaded data.

        Args:
            data: Dict from any load method

        Returns:
            Formatted string description
        """
        stokes = data['stokes']
        meta = data.get('metadata', {})

        lines = [
            "=" * 60,
            "  QRFT Black Hole Toolkit - Data Summary",
            "=" * 60,
            f"  Shape:        {stokes.shape}",
            f"  Pixels:       {stokes.shape[0]} x {stokes.shape[1]}",
            f"  Channels:     I, Q, U, V",
            f"  Synthetic:    {meta.get('synthetic', 'Unknown')}",
            "",
            "  Stokes I:     [{:.4e}, {:.4e}]".format(
                stokes[..., 0].min(), stokes[..., 0].max()),
            "  Stokes Q:     [{:.4e}, {:.4e}]".format(
                stokes[..., 1].min(), stokes[..., 1].max()),
            "  Stokes U:     [{:.4e}, {:.4e}]".format(
                stokes[..., 2].min(), stokes[..., 2].max()),
            "  Stokes V:     [{:.4e}, {:.4e}]".format(
                stokes[..., 3].min(), stokes[..., 3].max()),
        ]

        if 'uv_mask' in data:
            mask = data['uv_mask']
            coverage = mask.sum() / mask.size * 100
            lines.append(f"\n  UV Coverage:  {coverage:.1f}%")

        for key in ['source', 'mass_msun', 'spin',
                     'inclination_deg', 'shadow_radius_uas',
                     'fov_uas', 'n_data_points']:
            if key in meta:
                lines.append(f"  {key}: {meta[key]}")

        lines.append("=" * 60)
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return (
            f"EHTLoader(ehtim={'yes' if self._ehtim_available else 'no'}, "
            f"astropy={'yes' if self._astropy_available else 'no'})"
        )
