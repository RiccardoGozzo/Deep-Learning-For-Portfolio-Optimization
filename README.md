# Deep Learning for Portfolio Optimization: End-to-End Allocation via TSMixer, ASAM, and Linear Utility Loss

This repository implements an advanced end-to-end deep learning framework for tactical portfolio optimization. Unlike traditional two-stage approaches that suffer from noise propagation during the prediction-optimization gap, this model directly maps market features to optimal portfolio weights by internalizing financial metrics into the gradient ascent process.

## Key Innovations

### 1. Architectural Evolution: From LSTM to TSMixer
**LSTM Baseline**: Captures sequential dependencies via a recurrent core (64 units) with Layer Normalization and Dropout to stabilize learning across market regimes.
**TSMixer Transition**: To mitigate the spectral bias inherent in RNNs—which preferentially propagate low-frequency trends while filtering out noisy covariance structures—an All-MLP TSMixer architecture is implemented. This model utilizes time-mixing and feature-mixing layers to capture both temporal patterns and cross-sectional correlations.

### 2. Robust Optimization via ASAM
* Financial loss landscapes are notoriously non-convex, often driving optimizers toward sharp local minima with poor generalization.
* Adaptive Sharpness-Aware Minimization (ASAM) is integrated to target flat minima, significantly reducing metric fluctuations and stabilizing the out-of-sample Sharpe Ratio across turbulent and calm regimes.

### 3. Mitigating Greedy Bias with Linear Utility Loss
* Standard Sharpe Ratio maximization often exhibits a Momentum Trap, prioritizing expected returns (numerator) while ignoring risk (denominator) due to vanishing variance gradients during cold starts.
* A Linear Utility Loss (L = -mu + lambda * sigma^2) is introduced to mimic Mean-Variance utility. This formulation ensures an additive risk penalty that remains active throughout training, promoting a structurally stable and risk-aware allocation.

## Empirical Results
**Performance**: The framework outperforms traditional benchmarks (Min-Var, Maximum Diversification) in both high and low volatility regimes.
**Stability**: The combination of ASAM and Linear Utility Loss reduced daily weight turnover from 14.76% to 8.74% compared to standard Sharpe optimization, without sacrificing alpha capture.

## Tech Stack
**Core**: Python, PyTorch / TensorFlow.
**Data**: Daily frequency (VTI, AGG, DBC, VIX) spanning 20 years (2006–2026).
**Analysis**: Permutation Feature Importance, Circular Block Bootstrap for statistical significance.

## Documentation
* [View Presentation Slides](./Riccardo_Gozzo_Quant_Finance.pdf)
