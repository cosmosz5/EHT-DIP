---
layout: page
title: Method
permalink: /method/
---

# Method

## 1. Forward Model

The interferometric measurement equation relates the source brightness distribution *I*(*l*, *m*) to the complex visibility *V*(*u*, *v*) via the van Cittert–Zernike theorem:

$$V(u, v) = \iint I(l, m)\, e^{-2\pi i (ul + vm)}\, dl\, dm$$

For a discrete *H* × *W* pixel image, this reduces to a **Non-Uniform Discrete Fourier Transform (NUDFT)** evaluated at the observed (*u*, *v*) baselines. The NUDFT is implemented as a separable matrix product, precomputing cosine and sine tables for all baselines and pixel coordinates. This avoids gridding artifacts at the cost of O(*N* · *H* · *W*) complexity per forward pass.

**Closure phases** — triple products of visibility phases around a baseline triangle:

$$\psi_{ijk} = \arg(V_{ij} \cdot V_{jk} \cdot V_{ki})$$

are used in addition to visibility amplitudes because they are immune to antenna-based gain errors, making them the most calibration-robust EHT observable.

---

## 2. Network Architecture

The image is parameterized as:

$$\hat{I} = f_\theta(\mathbf{z})$$

where **z** is a fixed random input tensor of shape `[1, 32, H, W]` and *f*_θ is a **U-Net** with 5 encoder/decoder levels. The network weights θ are the only optimized parameters.

| Component | Specification |
|-----------|--------------|
| Input depth | 32 channels |
| Base channels | 64 |
| Encoder depth | 5 levels (channels: 64, 128, 256, 512, 512) |
| Downsampling | Average pooling (2×) |
| Upsampling | Bilinear (2×) + skip concatenation |
| Normalization | GroupNorm (groups = min(8, *C*)) |
| Activation | LeakyReLU (slope 0.2) |
| Output | 1-channel, passed through `softplus` for positivity |

The output image is optionally multiplied by a learnable global scale factor α = exp(log α), allowing the network to modulate total flux independently of network weight magnitudes.

---

## 3. Loss Function

The total objective is:

$$\mathcal{L} = w_\text{amp}\,\mathcal{L}_\text{amp} + w_\text{cp}\,\mathcal{L}_\text{cp} + \lambda_\text{TSV}\,\mathcal{R}_\text{TSV} + \lambda_\text{TV}\,\mathcal{R}_\text{TV} + \lambda_\text{flux}\,\mathcal{P}_\text{flux} + \lambda_\text{comp}\,\mathcal{P}_\text{comp} + \lambda_\text{void}\,\mathcal{P}_\text{void} + \lambda_{L1}\,\mathcal{R}_{L1} + \lambda_\text{cent}\,\mathcal{P}_\text{cent}$$

### Data fidelity

Both the amplitude and closure phase terms use a **Student's-*t* negative log-likelihood**:

$$\mathcal{L}_\text{t}(r, \sigma) = \frac{\nu + 1}{2} \log\!\left(1 + \frac{r^2}{\nu\,\sigma^2 s^2}\right)$$

with degrees of freedom ν and an optional learnable scale *s*. For ν → ∞ this reduces to Gaussian (least-squares) fitting; small ν (e.g., ν = 3) yields heavy tails that down-weight outliers.

- **Amplitude residual**: *r* = |*V*_model| − |*V*_obs|
- **Closure phase residual**: *r* = wrap(*ψ*_model − *ψ*_obs), where wrap maps to [−π, π] via `atan2(sin(·), cos(·))`

### Regularization

| Term | Formula | Effect |
|------|---------|--------|
| TSV | Σ (∇*I*)² | Penalizes squared pixel differences; promotes smooth images |
| TV | Σ \|∇*I*\| | Isotropic total variation; edge-preserving smoothness |
| L1 | mean(\|*I*\|) | Sparsity; suppresses diffuse flux |

### Geometric penalties

| Term | Formula | Effect |
|------|---------|--------|
| Flux | (*S*_total − *S*_target)² / *S*_target² | Constrains integrated flux density |
| Compactness | mean(*I* · 1[*R* > *R*_c]) | Suppresses emission outside radius *R*_c |
| Void | mean(*I* · 1[*R* < *R*_0]) | Suppresses emission inside shadow radius *R*_0 |
| Centroid | (*c*_x² + *c*_y²) / *R*_max² | Penalizes flux-weighted centroid offset from image center |

---

## 4. Coarse-to-Fine Optimization

The reconstructed image is blurred with a Gaussian kernel of σ_pix before computing losses. The blur width decays linearly from σ₀ to σ₁ over the first `blur_decay_iters` iterations:

$$\sigma(t) = (1 - t)\,\sigma_0 + t\,\sigma_1, \quad t = \min\!\left(1,\, \frac{\text{iter}}{\text{blur\_decay\_iters}}\right)$$

This coarse-to-fine schedule prevents premature commitment to high-frequency structure before the low-frequency image morphology is established.

---

## 5. Bootstrap Uncertainty Quantification

Nonparametric bootstrap resampling provides per-pixel uncertainty estimates without parametric noise assumptions. For each of *B* replicates:

1. Draw a weight vector **w** ∈ ℝ^N for the amplitude data points and (if available) **w**_cp ∈ ℝ^M for the closure phases, using either:
   - **Poisson(1) bootstrap**: *w_i* ~ Poisson(1) — equivalent to the standard pairs bootstrap in the large-*N* limit
   - **Dirichlet bootstrap**: **w** ~ Dirichlet(**1**) · *N* — a Bayesian bootstrap variant
2. Run a short reconstruction (`short_num_iter` iterations) with a warm start from the baseline MAP solution, applying the bootstrap weights as per-sample multipliers on the loss terms
3. Collect the *B* reconstructed images {*Î*^(b)}

From the ensemble, compute per-pixel statistics:
- Mean image: $\bar{I} = \frac{1}{B}\sum_b \hat{I}^{(b)}$
- 1σ uncertainty: $\sigma_I = \mathrm{std}_b\{\hat{I}^{(b)}\}$
- 16th and 84th percentiles for non-Gaussian posteriors

The warm start (reusing the fixed input **z** and initialized network weights from the baseline run) substantially reduces the per-replicate compute cost while still producing diverse ensemble members through distinct bootstrap weight draws.
