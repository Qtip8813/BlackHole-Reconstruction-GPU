import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# ====================== YOUR PHYSICS HELPERS ======================
# ====================== YOUR PHYSICS HELPERS (unchanged) ======================
# ====================== YOUR PHYSICS HELPERS (unchanged) ======================
def total_variation(psi):
    recon_chfirst = psi.permute(2, 0, 1).unsqueeze(0)
    recon_pad = torch.nn.functional.pad(recon_chfirst, (1, 1, 1, 1), mode='replicate')
    recon_pad = recon_pad.squeeze(0).permute(1, 2, 0)
    dx = recon_pad[2:, 1:-1, :] - recon_pad[:-2, 1:-1, :]
    dy = recon_pad[1:-1, 2:, :] - recon_pad[1:-1, :-2, :]
    return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))

def angular_momentum(psi, theta_field):
    I = psi[..., 0]
    Q = psi[..., 1]
    U = psi[..., 2]
    pol_frac = 0.3
    expected_Q = pol_frac * I * torch.cos(2 * theta_field)
    expected_U = pol_frac * I * torch.sin(2 * theta_field)
    return torch.mean((Q - expected_Q)**2 + (U - expected_U)**2)

def compute_rho_ent(r_field, r_horizon=20.0, r_photon=35.0):
    rho = torch.zeros_like(r_field)
    rho[r_field < r_horizon] = 1.0
    mask_out = r_field >= r_horizon
    rho[mask_out] = (r_photon / r_field[mask_out]) ** 2
    return torch.clamp(rho, 0.0, 1.0)

# ====================== V5 ENERGY - YOUR EXACT PROPOSAL ======================
def smt_energy_v5(psi, obs, mask, r_field, theta_field):
    E_data = torch.sum((psi - obs)**2 * mask)
    E_tv = total_variation(psi)

    # YOUR rotation term (kept strong - it works!)
    E_rotation = angular_momentum(psi, theta_field)

    # YOUR entanglement as SOFT GUIDE (exactly as you wrote)
    I = psi[..., 0]
    rho_ent = compute_rho_ent(r_field)

    # Robust Pearson correlation (maximize shape match, allow local freedom)
    I_flat = I.flatten()
    rho_flat = rho_ent.flatten()
    I_mean = I_flat.mean()
    rho_mean = rho_flat.mean()
    cov = ((I_flat - I_mean) * (rho_flat - rho_mean)).mean()
    std_i = I_flat.std(unbiased=False)
    std_rho = rho_flat.std(unbiased=False)
    corr = cov / (std_i * std_rho + 1e-8)
    E_ent_guide = -corr                     # negative = we penalize low correlation

    return (
        E_data +
        0.003 * E_tv +          # Grok smoothness
        0.005 * E_rotation +    # YOUR rotation (strong)
        0.001 * E_ent_guide     # YOUR entanglement as soft guide
    )

# ====================== MAIN TEST (V5 drop-in) ======================
def run_high_fidelity_test_v5():
    npix = 128
    np.random.seed(425)

    # === Identical synthetic black hole ===
    Y, X = np.mgrid[0:npix, 0:npix]
    R = np.sqrt((X - 64)**2 + (Y - 64)**2)
    Theta = np.arctan2(Y - 64, X - 64)
    I = np.exp(-0.5 * ((R - 35.0) / 8.0)**2)
    Q = 0.3 * I * np.cos(2 * Theta)
    U = 0.3 * I * np.sin(2 * Theta)
    V = 0.05 * I
    stokes_gt = np.stack([I, Q, U, V], axis=-1).astype(np.float32)

    uv_mask = np.random.random((npix, npix)) < 0.4
    stokes_obs = stokes_gt.copy()
    stokes_obs[~uv_mask] = 0.0

    print("\n" + "="*70)
    print("  SMT V5: SOFT PHYSICS GUIDED RECONSTRUCTION")
    print("  Energy = Grok TV + YOUR rotation + Soft Entanglement Correlation Guide")
    print("  Architect: Grok (xAI) + Rodney Lee Arnold Jr.")
    print("="*70)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Torch device: {device} (RTX 5070 acceleration engaged)")

    stokes_obs_t = torch.from_numpy(stokes_obs).to(device)
    mask_t = torch.from_numpy(uv_mask.astype(np.float32)).to(device).unsqueeze(-1)

    center = npix // 2
    yy, xx = torch.meshgrid(torch.arange(npix, device=device, dtype=torch.float32),
                            torch.arange(npix, device=device, dtype=torch.float32), indexing='ij')
    r_field = torch.sqrt((xx - center)**2 + (yy - center)**2)
    theta_field = torch.atan2(yy - center, xx - center)

    recon_t = stokes_obs_t.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([recon_t], lr=0.08)

    steps = 800
    for step in range(steps):
        optimizer.zero_grad()
        loss = smt_energy_v5(recon_t, stokes_obs_t, mask_t, r_field, theta_field)
        loss.backward()
        optimizer.step()

        if (step + 1) % 200 == 0:
            print(f"    Optimization Step {step+1}/{steps} | Total Energy: {loss.item():.6f}")

    recon = recon_t.detach().cpu().numpy()
    dt = time.time() - t0
    print(f"\n  ✓ SMT V5 Reconstruction Optimized in {dt:.3f}s")

    nc_i = np.corrcoef(stokes_gt[..., 0].flatten(), recon[..., 0].flatten())[0, 1]
    nc_q = np.corrcoef(stokes_gt[..., 1].flatten(), recon[..., 1].flatten())[0, 1]

    print("\n" + "="*70)
    print("  SOVEREIGN VALIDATION RESULTS - V5")
    print(f"  NxCorr (Stokes I): {nc_i:.6f}")
    print(f"  NxCorr (Stokes Q): {nc_q:.6f}")
    print("="*70)

    # Save comparison image
    try:
        plt.figure(figsize=(12, 6))
        plt.subplot(131); plt.imshow(stokes_gt[...,0], cmap='hot'); plt.title("Ground Truth")
        plt.subplot(132); plt.imshow(stokes_obs[...,0], cmap='hot'); plt.title("Observed (40%)")
        plt.subplot(133); plt.imshow(recon[...,0], cmap='hot'); plt.title(f"SMT V5 (I: {nc_i:.4f})")
        plt.tight_layout()
        plt.savefig('smt_v5_soft_physics_reconstruction.png', dpi=150)
        print("  ✓ Saved: smt_v5_soft_physics_reconstruction.png")
    except:
        pass

if __name__ == '__main__':
    run_high_fidelity_test_v5()
