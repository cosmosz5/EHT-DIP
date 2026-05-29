---
layout: page
title: Usage
permalink: /usage/
---

# Usage

## Installation

```bash
git clone https://github.com/cosmosz5/EHT-DIP.git
cd EHT-DIP
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.1.

GPU acceleration is supported via:
- **CUDA** (NVIDIA): detected automatically if `torch.cuda.is_available()`
- **MPS** (Apple Silicon): detected automatically if `torch.backends.mps.is_available()`
- **CPU** fallback otherwise

All computations run in float32, which is required for MPS compatibility.

---

## Data Format

Input data must be a Python dictionary (or `.npz` file) with the following keys:

| Key | Shape | Unit | Required |
|-----|-------|------|----------|
| `u` | `[N]` | wavelengths | Yes |
| `v` | `[N]` | wavelengths | Yes |
| `vis` | `[N]` | Jy | Yes |
| `vsigma` | `[N]` | Jy | Yes |
| `cphase` | `[M]` | degrees | No |
| `sigmacp` | `[M]` | degrees | No |
| `u1`, `v1` | `[M]` | wavelengths | No (required if `cphase` present) |
| `u2`, `v2` | `[M]` | wavelengths | No (required if `cphase` present) |

Closure triangle baseline 3 is derived internally as `(u3, v3) = (-(u1+u2), -(v1+v2))`.

Loading an `.npz` file:

```python
import numpy as np
data = dict(np.load('SR1_M87_2017_095_lo_hops_netcal_StokesI.npz'))
```

---

## Running a Reconstruction

```bash
python run_eht_dip_tested.py
```

The driver script loads day-101 data by default. Edit the `data_dict = ...` line to switch observation epochs.

### Programmatic interface

```python
from eht_dip_enhanced3_bootstrapping2 import reconstruct_eht_dip
import numpy as np

data = dict(np.load('SR1_M87_2017_095_lo_hops_netcal_StokesI.npz'))

image = reconstruct_eht_dip(
    data_dict=data,
    npix=64,
    pixel_size=2.0,       # μas / pixel
    device='mps',         # or 'cuda', 'cpu'
    save_dir='results/',
    save_every=400,
    amp_weight=1.5,
    cp_weight=1.0,
    lambda_tsv=10,
    use_t_amp=True, t_nu_amp=3,
    use_t_cp=True,  t_nu_cp=3,
    lambda_compact=0.5, r_compact_uas=40,
    lambda_void=1e-2,   r_void_uas=7,
    num_iter=10000,
    init_gaussian=True, gauss_fwhm_uas=20,
    blur_sigma0_pix=2.0, blur_sigma1_pix=0.0, blur_decay_iters=3000,
    learn_global_scale=True, global_scale_init=0.6,
)
# image: numpy array, shape (64, 64)
```

---

## Running Bootstrap Uncertainty Estimation

```python
from eht_dip_enhanced3_bootstrapping2 import reconstruct_eht_dip_bootstrap
import numpy as np

data = dict(np.load('SR1_M87_2017_095_lo_hops_netcal_StokesI.npz'))

result = reconstruct_eht_dip_bootstrap(
    data_dict=data,
    B=50,                       # number of replicates
    bootstrap_kind='poisson',   # or 'dirichlet'
    warm_start=True,
    reuse_z=True,
    short_num_iter=6000,
    save_stack=True,
    save_dir='results_bootstrap/',
    # pass the same reconstruction kwargs as above:
    npix=64,
    pixel_size=2.0,
    device='mps',
    amp_weight=1.5, cp_weight=1.0,
    lambda_tsv=10,
    use_t_amp=True, t_nu_amp=3,
    use_t_cp=True,  t_nu_cp=3,
    lambda_compact=0.5, r_compact_uas=40,
    lambda_void=1e-2,   r_void_uas=7,
    num_iter=4000,
    init_gaussian=True, gauss_fwhm_uas=20,
    blur_sigma0_pix=2.0, blur_sigma1_pix=0.0, blur_decay_iters=3000,
    learn_global_scale=True, global_scale_init=0.6,
)

mean_image = result['mean']    # (H, W) — mean across replicates
std_map    = result['std']     # (H, W) — per-pixel 1σ uncertainty
q16        = result['q16']     # (H, W) — 16th percentile
q84        = result['q84']     # (H, W) — 84th percentile
```

---

## Outputs

| File | Format | Description |
|------|--------|-------------|
| `results/img_NNNNN.npy` | NumPy | Image snapshot at iteration N |
| `results/image_final_*.npy` | NumPy | Final best image (lowest data loss) |
| `*.fits` | FITS | Same image in FITS format (flipped N→S) |
| `results/model_outputs.npz` | NumPy | Observed + model visibilities, baselines |
| `results/image_final.png` | PNG | Reconstructed image (inferno colormap) |
| `results/vis_amp_fit.png` | PNG | Visibility amplitude: observed vs. model |
| `results/closure_phase_fit.png` | PNG | Closure phase: observed vs. model |
| `bootstrap_stack.npy` | NumPy | Full (B, H, W) bootstrap ensemble (if `save_stack=True`) |

---

## Switching Observation Epochs

| Day | File |
|-----|------|
| 095 | `SR1_M87_2017_095_lo_hops_netcal_StokesI.npz` |
| 096 | `SR1_M87_2017_096_lo_hops_netcal_StokesI.npz` |
| 100 | `SR1_M87_2017_100_lo_hops_netcal_StokesI.npz` |
| 101 | `SR1_M87_2017_101_lo_hops_netcal_StokesI.npz` |

Change the filename in the `data_dict = dict(np.load(...))` line of `run_eht_dip_tested.py`.
