# Plan 002: Phase Disambiguation Network (PDN) Architecture

## Context & Motivation
Single-shot on-axis interferometry is mathematically ill-posed because $I(x,y) = A + B\cos(\Delta\phi)$. The inverse $\arccos(I)$ only yields the wrapped magnitude $|\Delta\phi_{wrap}|$, leaving a sign ambiguity ($\pm$) and a $2\pi k$ wrap ambiguity.
Direct continuous regression via UNet causes the model to memorize dataset priors because predicting the continuous phase is unidentifiable.

**The Solution:** The network will NOT predict continuous phase. Instead, it will act as a discrete classification engine over the ambiguity space.

## Architecture Steps

1. **Analytical Feature Extraction**
   - Given the normalized input intensity $I$, analytically compute the wrapped magnitude:
     $$F_{wrap} = \arccos(I)$$
   - Pass a 2-channel tensor $(I, F_{wrap})$ into the network.

2. **Network Topology (Discrete Prediction)**
   - Use a generic semantic segmentation architecture (e.g., U-Net or SegFormer).
   - The network has **two independent classification heads**:
     1. **Sign Head**: A 1x1 convolution outputting a 2-channel logit map for binary classification ($\hat{S}(x,y) \in \{-1, +1\}$).
     2. **Wrap Head**: A 1x1 convolution outputting an $N$-channel logit map for multi-class classification ($\hat{K}(x,y) \in \{0, 1, 2, ..., N-1\}$).

3. **Loss Formulation**
   - **Sign Loss**: Pixel-wise Cross-Entropy between predicted sign probabilities and ground truth sign $S_{gt}$.
   - **Wrap Loss**: Pixel-wise Cross-Entropy (or Ordinal Regression loss) between predicted wrap probabilities and $K_{gt}$.
   - *Crucial difference*: There is no direct MSE loss on the continuous phase! The physics are strictly preserved because the final phase is assembled deterministically.

4. **Deterministic Phase Assembly (Inference)**
   - At inference time, the final phase is constructed mathematically:
     $$\hat{\phi}_{pred} = \text{argmax}(\hat{S}) \cdot \arccos(I) + 2\pi \cdot \text{argmax}(\hat{K})$$

## Verification
- Run a dummy forward pass: `out_sign, out_wrap = model(I)` and assert `out_sign.shape == (B, 2, H, W)` and `out_wrap.shape == (B, N, H, W)`.
- Assert that training loss decreases strictly using Cross Entropy.
- Assert that the reconstructed intensity $\cos(\hat{\phi}_{pred})$ exactly matches the input intensity $I$ (guaranteed by the architecture).

## Maintenance Notes
- Choose $N$ based on the maximum expected phase depth in the dataset.
- Because this relies on exact normalization of $I \in [-1, 1]$, robust intensity normalization preprocessing is required before taking $\arccos$.
