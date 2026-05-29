---
layout: page
title: API Reference
permalink: /api/
---

# API Reference

All public functions and classes are defined in `eht_dip_enhanced3_bootstrapping2.py`.

---

## `reconstruct_eht_dip`

Main reconstruction function. Returns the best-fit image (lowest combined data-term loss).

```python
reconstruct_eht_dip(
    data_dict,
    npix=128, pixel_size=5.0, device=None,
    save_dir=None, save_every=None,
    amp_weight=None, cp_weight=None,
    lambda_tv=None, lambda_tsv=None,
    use_t_amp=None, t_nu_amp=None,
    use_t_cp=None,  t_nu_cp=None,
    lambda_flux=None, target_flux=None,
    lambda_compact=None, r_compact_uas=None,
    lambda_void=None,   r_void_uas=None,
    lambda_l1=None, lambda_centroid=None,
    init_gaussian=None, gauss_fwhm_uas=None,
    blur_sigma0_pix=None, blur_sigma1_pix=None, blur_decay_iters=None,
    learn_global_scale=None, global_scale_init=None,
    lr=None, num_iter=None, out_every=None,
    # bootstrap hooks (internal):
    amp_extra_weights=None, cp_extra_weights=None,
    fixed_z=None, init_model_state=None, return_state=False,
)
```

**Returns:** `np.ndarray` of shape `(H, W)` — reconstructed image. If `return_state=True`, returns `dict` with keys `"image"` and `"state"` (for warm-starting bootstrap).

### Key parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dict` | `dict` | — | Interferometric data (see [Usage](../usage/)) |
| `npix` | `int` | 128 | Image side length in pixels |
| `pixel_size` | `float` | 5.0 | Pixel scale (μas) |
| `device` | `str` | auto | `'cuda'`, `'mps'`, or `'cpu'` |
| `amp_weight` | `float` | 1.0 | Visibility amplitude loss weight |
| `cp_weight` | `float` | 1.0 | Closure phase loss weight |
| `lambda_tsv` | `float` | 0.0 | Total squared variation weight |
| `lambda_tv` | `float` | 0.0 | Total variation weight |
| `use_t_amp` | `bool` | True | Use Student's-*t* for amplitude loss |
| `t_nu_amp` | `float` | 3.0 | Student's-*t* degrees of freedom (amplitude) |
| `use_t_cp` | `bool` | True | Use Student's-*t* for closure phase loss |
| `t_nu_cp` | `float` | 3.0 | Student's-*t* degrees of freedom (closure phase) |
| `lambda_compact` | `float` | 0.0 | Compactness penalty weight |
| `r_compact_uas` | `float` | None | Compactness radius (μas) |
| `lambda_void` | `float` | 0.0 | Inner-void penalty weight |
| `r_void_uas` | `float` | None | Shadow exclusion radius (μas) |
| `lambda_flux` | `float` | 0.0 | Flux conservation weight |
| `target_flux` | `float` | None | Target integrated flux density (Jy) |
| `init_gaussian` | `bool` | False | Initialize network input with a Gaussian |
| `gauss_fwhm_uas` | `float` | None | FWHM of initialization Gaussian (μas) |
| `blur_sigma0_pix` | `float` | None | Initial blur σ (pixels); `None` disables |
| `blur_decay_iters` | `int` | 20000 | Iterations over which blur decays to `blur_sigma1_pix` |
| `learn_global_scale` | `bool` | False | Learn a global image scale factor α |
| `global_scale_init` | `float` | 1.0 | Initial value of α |
| `num_iter` | `int` | 2000 | Number of optimization iterations |
| `lr` | `float` | 1e-3 | Adam learning rate |

---

## `reconstruct_eht_dip_bootstrap`

Runs *B* bootstrap replicates and returns ensemble statistics.

```python
reconstruct_eht_dip_bootstrap(
    data_dict,
    B=200,
    bootstrap_kind='poisson',   # {'poisson', 'dirichlet'}
    seed=12345,
    do_baseline_map=True,
    warm_start=True,
    short_num_iter=600,
    reuse_z=True,
    save_dir=None,
    save_stack=False,
    **recon_kwargs               # forwarded to reconstruct_eht_dip
)
```

**Returns:** `dict` with keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `mean` | `(H, W)` | Per-pixel mean image |
| `std` | `(H, W)` | Per-pixel 1σ standard deviation |
| `q16` | `(H, W)` | 16th percentile |
| `q84` | `(H, W)` | 84th percentile |
| `baseline_image` | `(H, W)` | MAP image from baseline run |
| `images` | `(B, H, W)` | Full ensemble (if `save_stack=False`) |
| `stack_path` | `str` | Path to saved `.npy` stack (if `save_stack=True`) |

### Key parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `B` | `int` | 200 | Number of bootstrap replicates |
| `bootstrap_kind` | `str` | `'poisson'` | Weight distribution: `'poisson'` or `'dirichlet'` |
| `do_baseline_map` | `bool` | True | Run a full baseline MAP reconstruction before bootstrap |
| `warm_start` | `bool` | True | Initialize all replicates from baseline network weights |
| `reuse_z` | `bool` | True | Fix the network input **z** to the baseline value across replicates |
| `short_num_iter` | `int` | 600 | Optimization iterations per replicate |

---

## `DIPUNet`

U-Net used as the image generator.

```python
DIPUNet(input_depth=32, base=64, depth=5, dropout=0.0)
```

**Forward:** `z: [B, input_depth, H, W]` → `[B, 1, H, W]` (unnormalized; pass through `softplus` externally for positivity).

---

## `StudentTLoss1D`

Differentiable Student's-*t* negative log-likelihood for 1D residuals.

```python
loss = StudentTLoss1D(nu=3.0, learn_sigma=False, init_sigma=1.0)
nll  = loss(r, sigma=sigma_tensor, weight=weight_tensor)
```

- `r`: residuals `[N]`
- `sigma`: per-point uncertainty `[N]` or `None`
- `weight`: per-point sample weights `[N]` or `None`

---

## Utility Functions

| Function | Description |
|----------|-------------|
| `predict_vis(img, Cul, Sul, Cvm, Svm)` | NUDFT forward pass; returns `[N, 2]` (Re, Im) |
| `precompute_uv(u, v, H, W, cell_size_rad, ...)` | Precompute cos/sin DFT tables |
| `tv_loss(img)` | Isotropic total variation |
| `tsv_loss(img)` | Total squared variation |
| `gaussian_blur(img, sigma_pix, ...)` | Separable Gaussian blur with reflection padding |
| `wrap_phase(x)` | Wrap angle tensor to [−π, π] |
| `wrap_phase_diff(pred, obs)` | Wrapped difference (pred − obs) in [−π, π] |
| `radial_masks(H, W, cell_size_uas, r_compact, r_void, ...)` | Binary compactness and void masks |
| `pick_device(explicit)` | Auto-select MPS → CUDA → CPU |
