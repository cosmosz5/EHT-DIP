---
layout: home
title: EHT-DIP
---

# EHT-DIP: Deep Image Prior for EHT Imaging

EHT-DIP reconstructs radio interferometric images from Event Horizon Telescope (EHT) data using a **Deep Image Prior** (DIP) approach. An untrained convolutional U-Net serves as an implicit structural regularizer; its weights are optimized to fit observed visibility amplitudes and closure phases without requiring a training dataset or explicit image priors beyond network architecture.

Bootstrap resampling over the data provides **per-pixel uncertainty maps** — a direct, model-agnostic estimate of imaging fidelity.

---

## Features

- **Robust Student's-*t* likelihood** for visibility amplitudes and closure phases (ν = 3), resistant to outliers from antenna calibration errors or RFI
- **Separable NUDFT forward model** — exact, gridding-free mapping from image pixels to arbitrary (*u*, *v*) points
- **Geometric regularization** — compactness, inner-void (shadow), flux, and centroid constraints encoded as differentiable penalties
- **Coarse-to-fine optimization** via a decaying Gaussian blur schedule on the image output
- **Nonparametric bootstrap** — Poisson or Dirichlet multiplier resampling over *B* replicates yields per-pixel 1σ uncertainty images

---

## Navigation

- [Method](method/) — forward model, network architecture, loss terms, bootstrap
- [Usage](usage/) — installation, data format, running a reconstruction
- [API Reference](api/) — function and class signatures

---

## Data

The included `.npz` files contain calibrated Stokes I visibility data from the EHT 2017 M87 campaign (HOPS pipeline, lo-band, network-calibrated), spanning four observation epochs: days 095, 096, 100, and 101.
