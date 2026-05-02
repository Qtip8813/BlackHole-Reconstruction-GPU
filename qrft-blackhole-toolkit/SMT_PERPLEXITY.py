import torch
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt

# [Your exact physics helpers: total_variation, entanglement_density, angular_momentum, layer_coupling, smt_energy]
# ... (copy from V4.3 verbatim)
# ====================== YOUR PHYSICS HELPERS ======================
def total_variation(psi):
    recon_chfirst = psi.permute(2, 0, 1).unsqueeze(0)
    recon_pad = torch.nn.functional.pad(recon_chfirst, (1, 1, 1, 1), mode='replicate')
    recon_pad = recon_pad.squeeze(0).permute(1, 2, 0)
    dx = recon_pad[2:, 1:-1, :] - recon_pad[:-2, 1:-1, :]
    dy = recon_pad[1:-1, 2:, :] - recon_pad[1:-1, :-2, :]
    return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))

def entanglement_density(psi, r_field, r_horizon=20.0, r_photon=35.0):
    rho = torch.zeros_like(r_field)
    rho[r_field < r_horizon] = 1.0
    mask_out = r_field >= r_horizon
    rho[mask_out] = (r_photon / r_field[mask_out]) ** 2
    rho = torch.clamp(rho, 0.0, 1.0)
    I = psi[..., 0]
    I_norm = I / (I.max() + 1e-8)
    return torch.mean((I_norm - rho) ** 2)

def angular_momentum(psi, theta_field):
    I = psi[..., 0]
    Q = psi[..., 1]
    U = psi[..., 2]
    pol_frac = 0.3
    expected_Q = pol_frac * I * torch.cos(2 * theta_field)
    expected_U = pol_frac * I * torch.sin(2 * theta_field)
    return torch.mean((Q - expected_Q)**2 + (U - expected_U)**2)

def layer_coupling(psi, r_field):
    r1, r2, r3 = 25.0, 35.0, 50.0
    I = psi[..., 0]
    mask1 = r_field < r1
    mask2 = (r_field >= r1) & (r_field < r2)
    mask3 = (r_field >= r2) & (r_field < r3)
    var1 = torch.var(I[mask1]) if mask1.sum() > 0 else torch.tensor(0.0, device=I.device)
    var2 = torch.var(I[mask2]) if mask2.sum() > 0 else torch.tensor(0.0, device=I.device)
    var3 = torch.var(I[mask3]) if mask3.sum() > 0 else torch.tensor(0.0, device=I.device)
    return var1 + var2 + var3

def smt_energy(psi, obs, mask, r_field, theta_field,
               lambda1=0.003, lambda2=0.001, lambda3=0.01, lambda4=0.01):
    # Grok's terms
    E_data = torch.sum((psi - obs)**2 * mask)
    E_tv = total_variation(psi)

    # YOUR terms (3-layer structure) — exactly as you wrote
    E_layer = layer_coupling(psi, r_field)          # radial flow / layer smoothness
    E_rotation = angular_momentum(psi, theta_field) # rotation
    E_entanglement = entanglement_density(psi, r_field)  # ρ_ent

    return E_data + lambda1*E_tv + lambda2*E_layer + lambda3*E_rotation + lambda4*E_entanglement

def run_smt_v5(npix=512, noise_std=0.05, seed=425, verbose=True):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Synthetic GT (scales perfectly)
    Y, X = np.mgrid[0:npix, 0:npix]
    R = np.sqrt((X - npix//2)**2 + (Y - npix//2)**2)
    Theta = np.arctan2(Y - npix//2, X - npix//2)
    I = np.exp(-0.5 * ((R - 70.0) / 16.0)**2)  # Scaled ring
    Q = 0.3 * I * np.cos(2 * Theta)
    U = 0.3 * I * np.sin(2 * Theta)
    V = 0.05 * I * np.random.randn(*I.shape)  # V noise intrinsic
    stokes_gt = np.stack([I, Q, U, V], axis=-1).astype(np.float32)

    # Sparse + noise
    uv_mask = np.random.random((npix, npix)) < 0.4
    stokes_obs = stokes_gt.copy()
    stokes_obs[~uv_mask] *= 0  # Hole
    stokes_obs += noise_std * np.random.randn(*stokes_obs.shape)  # VLBI noise

    device = torch.device("cuda")
    stokes_obs_t = torch.from_numpy(stokes_obs).to(device)
    mask_t = torch.from_numpy(uv_mask.astype(np.float32)).to(device).unsqueeze(-1)

    # Fields (scaled)
    center = npix // 2
    yy, xx = torch.meshgrid(torch.arange(npix, device=device, dtype=torch.float32),
                            torch.arange(npix, device=device, dtype=torch.float32), indexing='ij')
    r_field = torch.sqrt((xx - center)**2 + (yy - center)**2)
    theta_field = torch.atan2(yy - center, xx - center)

    recon_t = stokes_obs_t.clone().requires_grad_(True)
    opt = torch.optim.Adam([recon_t], lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1200)

    print("="*80)
    print(f" SMT V5: 512px NOISE-ROBUST | σ={noise_std} | {np.mean(uv_mask):.1%} cov")
    print("="*80)

    best_loss, best_recon = float('inf'), None
    steps = 1200
    for step in range(steps):
        opt.zero_grad()
        loss = smt_energy(recon_t, stokes_obs_t, mask_t, r_field, theta_field,
                          lambda1=0.002, lambda2=0.0008, lambda3=0.008, lambda4=0.012)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([recon_t], 1.0)  # Stable grads
        opt.step()
        scheduler.step()

        recon_t.data.clamp_(0.0, 3.0)  # Positivity + bound

        if loss.item() < best_loss:
            best_loss, best_recon = loss.item(), recon_t.clone()

        if verbose and (step + 1) % 300 == 0:
            print(f"Step {step+1}/{steps} | Loss: {loss.item():.2e} | LR: {scheduler.get_last_lr()[0]:.2e}")

    recon = best_recon.detach().cpu().numpy()
    nc_i = np.corrcoef(stokes_gt[..., 0].flatten(), recon[..., 0].flatten())[0, 1]
    nc_q = np.corrcoef(stokes_gt[..., 1].flatten(), recon[..., 1].flatten())[0, 1]

    # Plot/save
    plt.figure(figsize=(15,5))
    plt.subplot(131); plt.imshow(stokes_gt[...,0], cmap='hot'); plt.title('GT')
    plt.subplot(132); plt.imshow(stokes_obs[...,0], cmap='hot'); plt.title(f'Obs+Noise (40%)')
    plt.subplot(133); plt.imshow(recon[...,0], cmap='hot'); plt.title(f'V5 (I:{nc_i:.4f})')
    plt.savefig('smt_v5_512_noise.png', dpi=150, bbox_inches='tight')

    print(f"✓ V5 512px: I={nc_i:.4f} Q={nc_q:.4f} | BestLoss={best_loss:.2e}")
    return recon, nc_i, nc_q

if __name__ == '__main__':
    run_smt_v5()
