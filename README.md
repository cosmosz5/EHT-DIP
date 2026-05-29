# EHT-DIP: Deep Image Prior for EHT Imaging

A PyTorch implementation of **Deep Image Prior (DIP)** for radio interferometric image reconstruction, designed for Event Horizon Telescope (EHT) data. The pipeline fits visibility amplitudes and closure phases using a convolutional U-Net as an implicit structural prior, with nonparametric bootstrap resampling for per-pixel uncertainty quantification.

**Documentation:** [https://cosmosz5.github.io/EHT-DIP](https://cosmosz5.github.io/EHT-DIP)

---

## Overview

Standard CLEAN-based imaging requires strong assumptions about source morphology. EHT-DIP instead uses the architecture of an untrained U-Net as a regularizer (Ulyanov et al. 2018), fitting the observed interferometric data by optimizing only the network weights while keeping the random input fixed. This approach naturally suppresses high-frequency noise and does not require a training set.

Key features:
- **Robust data fidelity**: Student's-*t* likelihood (ν = 3) for visibility amplitudes and closure phases, providing resistance to outliers from poorly calibrated baselines or RFI
- **Physics-consistent forward model**: separable Non-Uniform DFT (NUDFT) maps pixel images directly to arbitrary (*u*, *v*) points without gridding
- **Geometric priors**: compactness, inner-void (shadow), flux conservation, and centroid regularizers encoded as differentiable penalty terms
- **Coarse-to-fine initialization**: linearly decaying Gaussian blur schedule prevents early convergence to high-frequency artifacts
- **Bootstrap uncertainty maps**: Poisson or Dirichlet multiplier resampling over B replicates yields per-pixel 1σ uncertainty images

---

## Repository Structure

```
EHT-DIP/
├── eht_dip_enhanced3_bootstrapping2.py   # Core library: network, NUDFT, losses, bootstrap
├── run_eht_dip_tested.py                 # Driver: load data → reconstruct → plots → save
├── SR1_M87_2017_095_lo_hops_netcal_StokesI.npz   # EHT 2017, day 095, lo-band
├── SR1_M87_2017_096_lo_hops_netcal_StokesI.npz   # EHT 2017, day 096, lo-band
├── SR1_M87_2017_100_lo_hops_netcal_StokesI.npz   # EHT 2017, day 100, lo-band
├── SR1_M87_2017_101_lo_hops_netcal_StokesI.npz   # EHT 2017, day 101, lo-band
├── requirements.txt
└── docs/                                 # GitHub Pages documentation
```

---

## Installation

```bash
git clone https://github.com/cosmosz5/EHT-DIP.git
cd EHT-DIP
pip install -r requirements.txt
```

Requires Python ≥ 3.10. GPU acceleration is supported via CUDA or Apple Silicon MPS (float32).

---

## Data Format

Each `.npz` file contains the following arrays (all real-valued, float64):

| Key | Shape | Description |
|-----|-------|-------------|
| `u`, `v` | `[N]` | Baseline coordinates (wavelengths) |
| `vis` | `[N]` | Visibility amplitudes (Jy) |
| `vsigma` | `[N]` | Amplitude uncertainties (Jy) |
| `cphase` | `[M]` | Closure phases (degrees) |
| `sigmacp` | `[M]` | Closure phase uncertainties (degrees) |
| `u1`, `v1`, `u2`, `v2` | `[M]` | Baseline (*u*, *v*) pairs for closure triangles (wavelengths) |

The data originate from the EHT 2017 M87 campaign (Paper IV, The EHT Collaboration et al. 2019), low-band (lo), HOPS pipeline, network-calibrated, Stokes I.

---

## Quick Start

Edit the data path and output directory in `run_eht_dip_tested.py`, then:

```bash
python run_eht_dip_tested.py
```

Outputs written to `results_*/`:
- `img_NNNNN.npy` — image snapshots every `save_every` iterations
- `image_final_*.npy` / `*.fits` — final reconstructed image
- `model_outputs.npz` — model visibilities, observed data, baselines
- `image_final.png`, `vis_amp_fit.png`, `closure_phase_fit.png` — diagnostic plots

Bootstrap outputs (mean/std images) are written as `.fits` files in the working directory.

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `npix` | 64 | Image side length (pixels) |
| `pixel_size` | 2.0 μas | Pixel scale |
| `num_iter` | 10 000 | Optimization iterations |
| `amp_weight` | 1.5 | Visibility amplitude loss weight |
| `cp_weight` | 1.0 | Closure phase loss weight |
| `lambda_tsv` | 10 | Total squared variation weight |
| `t_nu_amp` / `t_nu_cp` | 3 | Student's-*t* degrees of freedom |
| `lambda_compact` | 0.5 | Compactness penalty weight |
| `r_compact_uas` | 40 μas | Compactness radius |
| `lambda_void` | 0.01 | Inner-void penalty weight |
| `r_void_uas` | 7 μas | Shadow exclusion radius |
| `blur_sigma0_pix` | 2.0 | Initial Gaussian blur σ (pixels) |
| `blur_decay_iters` | 3 000 | Iterations to decay blur to zero |
| `B` (bootstrap) | 10 | Number of bootstrap replicates |
| `short_num_iter` | 6 000 | Iterations per bootstrap replicate |

---

## Citation

If you use this code, please cite:

```bibtex
@article{EHT_M87_2019_IV,
  author  = {{The Event Horizon Telescope Collaboration}},
  title   = {First M87 Event Horizon Telescope Results. {IV}. Imaging the Central Supermassive Black Hole},
  journal = {The Astrophysical Journal Letters},
  volume  = {875},
  pages   = {L4},
  year    = {2019},
  doi     = {10.3847/2041-8213/ab0e85}
}

@inproceedings{Ulyanov2018,
  author    = {Ulyanov, Dmitry and Vedaldi, Andrea and Lempitsky, Victor},
  title     = {Deep Image Prior},
  booktitle = {CVPR},
  year      = {2018}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
