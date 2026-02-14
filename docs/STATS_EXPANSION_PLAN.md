# Stats Module Expansion Plan

## Research Sources
- Owl Tutorial: Statistical Functions chapter (studied 2026-02-14)
- Architecture Book: Chapter 6 (Statistical Computing)
- Essential F#: Domain modeling patterns
- OCaml Scientific Computing: Distribution implementations

## Design Principles
1. **Type Safety**: Phantom types for distribution parameters
2. **Consistency**: `{dist}_{func}` naming as in Owl
3. **Completeness**: pdf, cdf, ppf, rvs, mean, var, std for all
4. **F# Idioms**: Result types, computation expressions
5. **Performance**: Vectorized operations where possible

## Distribution Categories

### Continuous Distributions
1. **Gaussian** (Normal) ✅ Already implemented
   - gaussian_pdf, gaussian_cdf, gaussian_ppf, gaussian_rvs
   - mean, var, std

2. **Uniform** ✅ Already implemented

3. **Gamma** ✅ Already implemented

4. **Beta** (NEW)
   - beta_pdf, beta_cdf, beta_ppf, beta_rvs
   - Uses: Bayesian stats, order statistics
   - Special functions: Beta function, incomplete beta

5. **Student's t** (NEW)
   - t_pdf, t_cdf, t_ppf, t_rvs
   - Uses: Hypothesis testing, confidence intervals
   - Special functions: Gamma ratios

6. **Chi-Square** (NEW)
   - chi2_pdf, chi2_cdf, chi2_ppf, chi2_rvs
   - Uses: Goodness of fit, variance testing
   - Relationship to Gamma

7. **F-Distribution** (NEW)
   - f_pdf, f_cdf, f_ppf, f_rvs
   - Uses: ANOVA, variance ratio tests

8. **Exponential** ✅ Already implemented

9. **Log-Normal** (NEW)
   - lognormal_pdf, lognormal_cdf, lognormal_ppf, lognormal_rvs
   - Uses: Financial modeling, particle sizes

10. **Cauchy** (NEW)
    - cauchy_pdf, cauchy_cdf, cauchy_ppf, cauchy_rvs
    - Uses: Robust statistics, physics

### Discrete Distributions
1. **Binomial** (NEW)
   - binomial_pmf, binomial_cdf, binomial_ppf
   - Uses: Success/failure experiments

2. **Poisson** (NEW)
   - poisson_pmf, poisson_cdf, poisson_ppf
   - Uses: Count data, rare events

3. **Geometric** (NEW)
   - geometric_pmf, geometric_cdf, geometric_ppf
   - Uses: Waiting times

4. **Hypergeometric** (NEW)
   - hypergeometric_pmf, hypergeometric_cdf
   - Uses: Sampling without replacement

## Statistical Tests (NEW Section)

### Hypothesis Testing
1. **One-sample t-test**
   - ttest_1samp: Compare sample mean to known value
   
2. **Two-sample t-test**
   - ttest_ind: Compare means of two independent samples
   - ttest_rel: Paired samples

3. **Chi-square test**
   - chisquare: Goodness of fit
   - chisquare_contingency: Independence test

4. **F-test**
   - ftest: Compare variances

5. **Normality tests**
   - shapiro_wilk: Shapiro-Wilk test
   - anderson_darling: Anderson-Darling test
   - kolmogorov_smirnov: KS test

### Descriptive Statistics Extensions
1. **Correlation**
   - pearson_correlation ✅ Already have
   - spearman_correlation: Rank-based
   - kendall_tau: Concordance-based

2. **Moments**
   - skewness ✅ Already have
   - kurtosis ✅ Already have
   - moment: General n-th moment ✅ Already have

3. **Order Statistics**
   - percentile ✅ Already have
   - quartiles: Q1, Q2, Q3
   - iqr: Interquartile range
   - rankdata: Ranking with ties

## Implementation Strategy

### Phase 1: Continuous Distributions (Priority)
- Beta distribution
- Student's t
- Chi-square
- F-distribution
- Log-normal
- Cauchy

### Phase 2: Discrete Distributions
- Binomial
- Poisson
- Geometric

### Phase 3: Statistical Tests
- t-tests
- Chi-square tests
- Normality tests

### Phase 4: Advanced Features
- Maximum likelihood estimation
- Distribution fitting
- Random variable algebra

## Technical Details

### Special Functions Required
1. **Beta function**: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
2. **Incomplete beta**: I_x(a,b) for Beta CDF
3. **Gamma ratios**: For t-distribution
4. **Error function**: For normal CDF (already have)

### Implementation Approach
1. Use existing SpecialFunctions module
2. Add new special functions as needed
3. Property-based testing with FsCheck
4. Reference Cephes/Netlib implementations

### API Design Example
```fsharp
module BetaDistribution =
    // Parameters: alpha > 0, beta > 0
    let pdf alpha beta x : FowlResult<float> = ...
    let cdf alpha beta x : FowlResult<float> = ...
    let ppf alpha beta p : FowlResult<float> = ...
    let rvs alpha beta shape : FowlResult<Ndarray> = ...
    let mean alpha beta : FowlResult<float> = ...
    let var alpha beta : FowlResult<float> = ...
    let std alpha beta : FowlResult<float> = ...
```

## References
1. Owl Tutorial - Statistical Functions (studied 2026-02-14)
2. "Statistical Computing in OCaml" - Owl documentation
3. Cephes Mathematical Library - netlib.org/cephes
4. "Numerical Recipes" - Press et al. (for algorithms)
5. NIST Handbook of Statistical Methods

## Success Criteria
- [ ] All distributions have 6 functions (pdf/cdf/ppf/rvs/mean/var)
- [ ] Property-based tests for all distributions
- [ ] Error handling for invalid parameters
- [ ] Documentation with examples
- [ ] Performance comparable to Owl/NumPy

---

_Plan created: 2026-02-14_
_Based on: Owl tutorial, Architecture book, Cephes library_
