#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner for eht_dip_enhanced.py that:
- Loads data using npz files
- Calls reconstruct_eht_dip(...) from eht_dip_enhanced.py
- Displays and saves:
  * reconstructed image (PNG + NPY)
  * visibility amplitude fit plot
  * closure phase fit plot (if closure inputs available)
  * model outputs (NPZ)

Expected keys in data_dict returned by data_reader:
  'u', 'v', 'vis', 'vsigma', 'cphase', 'sigmacp', 'u1', 'v1', 'u2', 'v2'
(Closure arrays are optional; plots will be skipped if absent.)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from eht_dip_enhanced3_bootstrapping2 import reconstruct_eht_dip, reconstruct_eht_dip_bootstrap
import time

start_time = time.perf_counter()

# ----------------------------
# Helpers
# ----------------------------
def ensure_float_arrays(data: dict) -> dict:
    """Make sure expected arrays are float numpy arrays."""
    keys_float = ["u","v","vis","vsigma","u1","v1","u2","v2","cphase","sigmacp"]
    out = dict(data)
    for k in keys_float:
        if k in out and out[k] is not None:
            out[k] = np.asarray(out[k], dtype=float)
    return out


def cell_size_rad_from_uas(pixel_size_uas: float) -> float:
    # 1 arcsec = pi/648000 rad; 1 μas = 1e-6 arcsec
    return (pixel_size_uas * 1e-6) * (np.pi / 648000.0)


def nufft_image_to_vis(image, u, v, cell_size_rad):
    """
    Separable NUDFT to compute model visibilities at (u,v):
      Real = sum_h [Cvm * A^T] - sum_h [Svm * B^T]
      Imag = - (sum_h [Cvm * B^T] + sum_h [Svm * A^T])
    """
    H, W = image.shape
    l = (np.arange(W) - (W // 2)) * cell_size_rad
    m = (np.arange(H) - (H // 2)) * cell_size_rad
    ul = 2.0 * np.pi * np.outer(u, l)   # [N,W]
    vm = 2.0 * np.pi * np.outer(v, m)   # [N,H]
    Cul = np.cos(ul);  Sul = np.sin(ul)
    Cvm = np.cos(vm);  Svm = np.sin(vm)
    A = image @ Cul.T                   # [H,N]
    B = image @ Sul.T
    real = (Cvm * A.T).sum(axis=1) - (Svm * B.T).sum(axis=1)
    imag = -((Cvm * B.T).sum(axis=1) + (Svm * A.T).sum(axis=1))
    return real + 1j * imag


def closure_phases_from_image(image, u1, v1, u2, v2, cell_size_rad):
    """
    Compute closure phases for triangles:
      baseline3: (u3, v3) = (-(u1+u2), -(v1+v2))
      phi = arg( V1 * V2 * V3 )
    Returns phi in radians.
    """
    if u1 is None or v1 is None or u2 is None or v2 is None:
        return None
    u1 = np.asarray(u1, dtype=float); v1 = np.asarray(v1, dtype=float)
    u2 = np.asarray(u2, dtype=float); v2 = np.asarray(v2, dtype=float)
    u3 = -(u1 + u2); v3 = -(v1 + v2)
    V1 = nufft_image_to_vis(image, u1, v1, cell_size_rad)
    V2 = nufft_image_to_vis(image, u2, v2, cell_size_rad)
    V3 = nufft_image_to_vis(image, u3, v3, cell_size_rad)
    return np.angle(V1 * V2 * V3)


# ----------------------------
# Main
# ----------------------------
def main():
 
    outdir='results_lo095_tested'
    outdir_boots = 'results_bootstrapping_l095_tested'
    pixel_size_uas = 2.0
    os.makedirs(outdir, exist_ok=True)

    # ----------------------------
    # Load data via npz files
    # ----------------------------
    data_dict = dict(np.load('SR1_M87_2017_101_lo_hops_netcal_StokesI.npz'))


    if not isinstance(data_dict, dict):
        raise TypeError("data_reader(...) must return a dictionary.")

    data = ensure_float_arrays(data_dict)

    # Basic validation for amp plot
    for key in ["u", "v", "vis"]:
        if key not in data or data[key] is None:
            raise ValueError(f"data_dict must contain key '{key}'.")

    # If no vsigma is provided, fall back to a small constant (prevents division by zero)
    if "vsigma" not in data or data["vsigma"] is None:
        data["vsigma"] = np.full_like(data["vis"], 1e-3, dtype=float)

    # ----------------------------
    # Run reconstruction
    # ----------------------------
    image = reconstruct_eht_dip(
        data_dict=data,
        npix=64,
        pixel_size=pixel_size_uas,  # μas / pixel
        device='mps',
        save_dir=outdir,
        save_every=400,
        # weights / losses
        amp_weight=1.5,
        cp_weight=1.0,
        lambda_tv=0,
        lambda_tsv=10,
        # Student's-t
        use_t_amp=True, t_nu_amp=3,
        use_t_cp=True,   t_nu_cp=3,
        # Flux / Compactness / Void
        lambda_flux=0.0, target_flux=0.6,
        lambda_compact=0.5, r_compact_uas=40,
        lambda_void=1e-2,       r_void_uas=7,
        lambda_l1=0,                
        lambda_centroid=0,           
        num_iter=10000,
        init_gaussian=True,
        gauss_fwhm_uas=20,
        out_every=2000,
        blur_sigma0_pix=2.0, blur_sigma1_pix=0.0, blur_decay_iters=3000,
        learn_global_scale=True, global_scale_init=0.6
    )

    # ----------------------------
    # Save reconstructed image
    # ----------------------------
    
    import astropy.io.fits as pyfits
    pyfits.writeto('test_im_ref10_1000.fits', np.flipud(image), overwrite=True)
    np.save(os.path.join(outdir, "image_final_ref10_1000.npy"), image)
    end_time = time.perf_counter()
    time_minutes = (end_time - start_time)/60
    print(f"Elapsed time: {time_minutes:.4f} minutes")

    os.makedirs(outdir_boots, exist_ok=True)
    
    res = reconstruct_eht_dip_bootstrap(
        data_dict=data,                     # your dict with u,v,vis,vsigma,(cphase,sigmacp,u1,v1,u2,v2)
        B=10,                         # number of bootstrap replicates
        do_baseline_map=False,          # run a MAP once for warm-start + z reuse
        warm_start=True,
        short_num_iter=6000,
        reuse_z=True,
        save_stack=True,

        # pass through your usual recon settings:
        npix=64,
        pixel_size=pixel_size_uas,  # μas / pixel
        device='mps',
        save_dir=outdir_boots,
        save_every=400,
        # weights / losses
        amp_weight=1.5,
        cp_weight=1.0,
        lambda_tv=0,
        lambda_tsv=10,
        # Student's-t
        use_t_amp=True, t_nu_amp=3,
        use_t_cp=True,   t_nu_cp=3,
        # Flux / Compactness / Void
        lambda_flux=0.0, target_flux=0.6,
        lambda_compact=0.5, r_compact_uas=40,
        lambda_void=1e-2,       r_void_uas=7,
        lambda_l1=0,                 # NEW
        lambda_centroid=0,           # NEW
        num_iter=4000,
        init_gaussian=True,
        gauss_fwhm_uas=20,
        out_every=2000,
        blur_sigma0_pix=2.0, blur_sigma1_pix=0.0, blur_decay_iters=3000,
        learn_global_scale=True, global_scale_init=0.6
    )

    # Per-pixel uncertainty map (1σ) from the bootstrap:
    std_map = res['std']     # shape (H, W)
    mean_img = res['mean']   # shape (H, W)
    
    pyfits.writeto('test_mean_im_ref10.fits', np.flipud(mean_img), overwrite=True)
    pyfits.writeto('test_std_im_ref10.fits', np.flipud(std_map), overwrite=True)

    image = mean_img * 1.0
    
    # ----------------------------
    # Diagnostic plots and model outputs
    # ----------------------------
    u, v = data["u"], data["v"]
    vis = data["vis"]; vsigma = data["vsigma"]
    cell_size_rad = cell_size_rad_from_uas(pixel_size_uas)
    Vmodel = nufft_image_to_vis(image, u, v, cell_size_rad)
    amp_model = np.abs(Vmodel)
    bl = np.sqrt(u*u + v*v)

    # Save arrays
    np.savez(os.path.join(outdir, "model_outputs.npz"),
             u=u, v=v, bl=bl,
             vis_obs=vis, vis_sigma=vsigma, vis_model=amp_model,
             cphase_obs=data.get("cphase", None), cphase_sigma=data.get("sigmacp", None),
             u1=data.get("u1", None), v1=data.get("v1", None),
             u2=data.get("u2", None), v2=data.get("v2", None))

    # Reconstructed image plot
    H, W = image.shape
    extent_uas = np.array([-W/2, W/2, -H/2, H/2]) * pixel_size_uas
    fig0, ax0 = plt.subplots(1, 1, figsize=(5.4, 4.6), constrained_layout=True)
    im = ax0.imshow(image, origin="upper", cmap="inferno",
                    extent=[extent_uas[0], extent_uas[1], extent_uas[2], extent_uas[3]])
    ax0.set_xlabel(r"RA offset [$\mu$as]")
    ax0.set_ylabel(r"Dec offset [$\mu$as]")
    ax0.set_title("Reconstructed Image")
    cb = plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cb.set_label("Brightness (arb.)")
    fig0.savefig(os.path.join(outdir, "image_final.png"), dpi=220)

    # Visibility amplitude fit
    order = np.argsort(bl)
    fig1, ax1 = plt.subplots(1, 1, figsize=(6.2, 4.4), constrained_layout=True)
    if vsigma is not None:
        ax1.errorbar(bl[order], vis[order], yerr=vsigma[order], fmt="o", ms=3, lw=0.7, alpha=0.85, label="Observed")
    else:
        ax1.plot(bl[order], vis[order], "o", ms=3, alpha=0.85, label="Observed")
    ax1.plot(bl[order], amp_model[order], "-", marker="x", ms=3, lw=1.0, label="Model")
    ax1.set_xlabel("Baseline length (wavelengths)")
    ax1.set_ylabel("Visibility amplitude (Jy)")
    ax1.set_title("Visibility Amplitude vs Baseline")
    ax1.legend(loc="best", frameon=False)
    fig1.savefig(os.path.join(outdir, "vis_amp_fit.png"), dpi=220)

    # Closure phases (if available)
    u1 = data.get("u1", None); v1 = data.get("v1", None)
    u2 = data.get("u2", None); v2 = data.get("v2", None)
    cphase = data.get("cphase", None); sigmacp = data.get("sigmacp", None)

    if cphase is not None and u1 is not None and v1 is not None and u2 is not None and v2 is not None:
        cp_model = closure_phases_from_image(image, u1, v1, u2, v2, cell_size_rad)
        l1 = np.sqrt(np.asarray(u1)**2 + np.asarray(v1)**2)
        l2 = np.sqrt(np.asarray(u2)**2 + np.asarray(v2)**2)
        l3 = np.sqrt((-(np.asarray(u1)+np.asarray(u2)))**2 + (-(np.asarray(v1)+np.asarray(v2)))**2)
        cp_bl = np.maximum(l1, np.maximum(l2, l3))

        fig2, ax2 = plt.subplots(1, 1, figsize=(6.2, 4.4), constrained_layout=True)
        if sigmacp is None:
            ax2.plot(cp_bl, cphase, "o", ms=3, alpha=0.85, label="Observed")
        else:
            ax2.errorbar(cp_bl, cphase, yerr=sigmacp,
                         fmt="o", ms=3, lw=0.7, alpha=0.85, label="Observed")
        ax2.plot(cp_bl, np.rad2deg(cp_model), "r^", ms=4, label="Model")
        ax2.set_xlabel("Max baseline in triangle (wavelengths)")
        ax2.set_ylabel("Closure phase (deg)")
        ax2.set_title("Closure Phase vs Baseline")
        ax2.legend(loc="best", frameon=False)
        fig2.savefig(os.path.join(outdir, "closure_phase_fit.png"), dpi=220)

    plt.show()
    plt.close("all")

    print(f"✓ Done. Saved results to: {outdir}")


if __name__ == "__main__":
    main()
