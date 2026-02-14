# Owl Tutorial - Statistical Functions (Chapter Notes)

**Date:** 2026-02-14  
**Chapter:** Statistical Functions  
**Source:** https://ocaml.xyz/tutorial/chapters/stats.html

## Overview

Statistics in Owl covers three groups: descriptive statistics, distributions, and hypothesis tests.

## Random Variables

**Definition**: Function mapping sample outcomes to numbers of interest. Reduces sample space complexity.

### Discrete Random Variables

**Binomial Distribution**: n independent trials, each with success probability p

```
P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
```

Owl functions:
- `Stats.binomial_rvs p n` - random variate sampling
- `Stats.binomial_pdf ~p ~n k` - probability density function
- `Stats.binomial_cdf ~p ~n k` - cumulative distribution function

**Key concepts**:
- PDF: p_X(k) = P(X = k)
- CDF: P(X ≤ k) = sum of PDF up to k

### Continuous Random Variables

**Gaussian (Normal) Distribution**:
```
p(x) = (1/(σ√(2π))) * e^(-1/2 * ((x-μ)/σ)²)
```

Owl functions:
- `Stats.gaussian_rvs ~mu ~sigma` - random sampling
- `Stats.gaussian_pdf ~mu ~sigma x` - PDF
- `Stats.gaussian_cdf ~mu ~sigma x` - CDF

## Descriptive Statistics

### Central Tendency & Spread
- `Stats.mean` - E(X) = (1/n)Σxᵢ
- `Stats.var` - Var(X) = (1/n)Σ(xᵢ - E(X))²
- `Stats.std` - sqrt(variance)

### Moments
- **1st moment**: Mean
- **2nd moment**: Variance  
- **3rd moment**: Skewness (asymmetry)
- **4th moment**: Kurtosis (tail length)

`Stats.central_moment n data` - compute nth central moment

### Order Statistics
- `Stats.min`, `Stats.max`
- `Stats.median` - 2nd quartile
- `Stats.first_quartile` - 25th percentile
- `Stats.third_quartile` - 75th percentile
- `Stats.quantile`, `Stats.percentile`

## Special Distributions

For each distribution, Owl provides a function family:
- `_rvs` - random variate sampler
- `_pdf` - probability density function
- `_cdf` - cumulative distribution function
- `_ppf` - percent point function (inverse CDF)
- `_sf` - survival function (1-CDF)
- `_isf` - inverse survival function
- `_logpdf`, `_logcdf`, `_logsf` - log variants

**Supported distributions**:
- Gaussian (normal)
- Gamma
- Beta
- Cauchy
- Student's t-distribution
- Binomial (discrete)
- Uniform
- Many others...

Example usage:
```ocaml
Stats.gamma_rvs ~shape:1. ~scale:2.      (* sample *)
Stats.gamma_pdf x ~shape:2. ~scale:2.    (* PDF *)
Stats.gamma_cdf x ~shape:2. ~scale:2.    (* CDF *)
```

## Multiple Variables

### Joint Probability
- `p(X,Y)` - probability both events occur
- For independent: `p(X,Y) = p(X) * p(Y)`

### Conditional Probability
```
P(X|Y) = P(X ∩ Y) / P(Y)
```

### Bayes' Theorem
```
P(X|Y) = P(Y|X) * P(X) / P(Y)
```

Key application: Update prior beliefs P(X) with observed evidence to get posterior P(X|Y)

## Sampling & Inference

### Sampling Methods
- Random sampling
- Stratified random sampling (by subgroups)

### Unbiased Estimators
- Sample statistics can estimate population parameters
- Some estimators are unbiased (expected value = population parameter)

## Key Insights for Fowl

1. **Function Naming Convention**: `{distribution}_{function}` pattern (e.g., `gaussian_pdf`, `gamma_cdf`)
2. **Comprehensive API**: Each distribution has ~9 related functions covering PDF, CDF, sampling, etc.
3. **Parameter Naming**: OCaml uses `~param` for named arguments - F# can use optional parameters or records
4. **Moment Calculations**: Built-in support for arbitrary central moments
5. **Bayesian Foundation**: Bayes' theorem implementation enables probabilistic reasoning

## F# Mapping Considerations

```fsharp
// OCaml: Stats.gaussian_pdf ~mu:1. ~sigma:1. x
// F# options:
let gaussianPdf (mu: float) (sigma: float) (x: float) = ...
gaussianPdf 1.0 1.0 x

// Or with optional parameters:
let gaussianPdf ?mu ?sigma x = ...

// Or with record type:
type GaussianParams = { Mu: float; Sigma: float }
let gaussianPdf (p: GaussianParams) x = ...
```

**Distribution modules pattern**:
```fsharp
module Gaussian =
    let sample mu sigma = ...
    let pdf mu sigma x = ...
    let cdf mu sigma x = ...
    // ... etc
```

---

_Next: Dataframe for Tabular Data chapter_
