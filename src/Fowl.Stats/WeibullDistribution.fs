namespace Fowl.Stats

open System
open Fowl
open Fowl.Core.Types

/// <summary>Weibull distribution.
/// Flexible distribution for reliability analysis and survival modeling.
/// PDF(x) = (k/λ)(x/λ)^(k-1) * exp(-(x/λ)^k)
/// where k = shape, λ = scale
/// </summary>module WeibullDistribution =
    
    /// <summary>Validate Weibull parameters.
/// </summary>let private validate (shape: float) (scale: float) : FowlResult<unit> =
        if shape <= 0.0 then
            Error.invalidArgument "Weibull shape must be positive"
        elif scale <= 0.0 then
            Error.invalidArgument "Weibull scale must be positive"
        else
            Ok ()
    
    /// <summary>Probability density function.
/// </summary>let pdf (shape: float) (scale: float) (x: float) : FowlResult<float> =
        result {
            do! validate shape scale
            
            if x < 0.0 then
                return 0.0
            elif x = 0.0 then
                if shape < 1.0 then
                    return Double.PositiveInfinity
                elif shape = 1.0 then
                    return 1.0 / scale
                else
                    return 0.0
            else
                let xOverScale = x / scale
                let powTerm = xOverScale ** (shape - 1.0)
                let expTerm = exp (-(xOverScale ** shape))
                return (shape / scale) * powTerm * expTerm
        }
    
    /// <summary>Cumulative distribution function.
/// CDF(x) = 1 - exp(-(x/λ)^k)
/// </summary>let cdf (shape: float) (scale: float) (x: float) : FowlResult<float> =
        result {
            do! validate shape scale
            
            if x <= 0.0 then
                return 0.0
            else
                let xOverScale = x / scale
                return 1.0 - exp (-(xOverScale ** shape))
        }
    
    /// <summary>Survival function (complementary CDF).
/// S(x) = exp(-(x/λ)^k)
/// </summary>let survival (shape: float) (scale: float) (x: float) : FowlResult<float> =
        result {
            do! validate shape scale
            
            if x <= 0.0 then
                return 1.0
            else
                let xOverScale = x / scale
                return exp (-(xOverScale ** shape))
        }
    
    /// <summary>Hazard function (failure rate).
/// h(x) = (k/λ)(x/λ)^(k-1)
/// </summary>let hazard (shape: float) (scale: float) (x: float) : FowlResult<float> =
        result {
            do! validate shape scale
            
            if x <= 0.0 then
                if shape < 1.0 then
                    return Double.PositiveInfinity
                elif shape = 1.0 then
                    return 1.0 / scale
                else
                    return 0.0
            else
                let xOverScale = x / scale
                return (shape / scale) * (xOverScale ** (shape - 1.0))
        }
    
    /// <summary>Percent point function (inverse CDF).
/// PPF(p) = λ * (-ln(1-p))^(1/k)
/// </summary>let ppf (shape: float) (scale: float) (p: float) : FowlResult<float> =
        result {
            do! validate shape scale
            
            if p < 0.0 || p > 1.0 then
                return! Error.invalidArgument "Weibull PPF requires 0 ≤ p ≤ 1"
            elif p = 0.0 then
                return 0.0
            elif p = 1.0 then
                return Double.PositiveInfinity
            else
                return scale * ((-log (1.0 - p)) ** (1.0 / shape))
        }
    
    /// <summary>Random variate sampling.
/// Uses inverse transform: x = λ * (-ln(u))^(1/k)
/// </summary>let rvs (shape: float) (scale: float) (seed: int option) : FowlResult<float> =
        result {
            do! validate shape scale
            
            let rng = match seed with Some s -> Random(s) | None -> Random()
            let u = rng.NextDouble()
            // Avoid u = 0 which gives infinity
            let u = max u 1e-15
            return scale * ((-log u) ** (1.0 / shape))
        }
    
    /// <summary>Mean: λ * Γ(1 + 1/k)
/// </summary>let mean (shape: float) (scale: float) : FowlResult<float> =
        result {
            do! validate shape scale
            let gamma1OverK = SpecialFunctions.gamma (1.0 + 1.0 / shape)
            return scale * gamma1OverK
        }
    
    /// <summary>Variance: λ²[Γ(1 + 2/k) - Γ(1 + 1/k)²]
/// </summary>let variance (shape: float) (scale: float) : FowlResult<float> =
        result {
            do! validate shape scale
            let gamma1OverK = SpecialFunctions.gamma (1.0 + 1.0 / shape)
            let gamma2OverK = SpecialFunctions.gamma (1.0 + 2.0 / shape)
            return scale * scale * (gamma2OverK - gamma1OverK * gamma1OverK)
        }
    
    /// <summary>Standard deviation.
/// </summary>let std (shape: float) (scale: float) : FowlResult<float> =
        result {
            let! var = variance shape scale
            return sqrt var
        }
    
    /// <summary>Mode: λ((k-1)/k)^(1/k) for k > 1, 0 for k = 1
/// </summary>let mode (shape: float) (scale: float) : FowlResult<float> =
        result {
            do! validate shape scale
            
            if shape < 1.0 then
                return 0.0
            elif shape = 1.0 then
                return 0.0
            else
                return scale * ((shape - 1.0) / shape) ** (1.0 / shape)
        }
    
    /// <summary>Median: λ(ln 2)^(1/k)
/// </summary>let median (shape: float) (scale: float) : FowlResult<float> =
        result {
            do! validate shape scale
            return scale * (log 2.0) ** (1.0 / shape)
        }
    
    /// <summary>Skewness: [Γ(1+3/k)λ³ - 3μσ² - μ³] / σ³
/// </summary>let skewness (shape: float) (scale: float) : FowlResult<float> =
        result {
            do! validate shape scale
            let g1 = SpecialFunctions.gamma (1.0 + 1.0 / shape)
            let g2 = SpecialFunctions.gamma (1.0 + 2.0 / shape)
            let g3 = SpecialFunctions.gamma (1.0 + 3.0 / shape)
            
            let mu = scale * g1
            let var = scale * scale * (g2 - g1 * g1)
            let sigma = sqrt var
            let mu3 = scale * scale * scale * (g3 - 3.0 * g2 * g1 + 2.0 * g1 * g1 * g1)
            
            return (mu3 - 3.0 * mu * var - mu * mu * mu) / (sigma * sigma * sigma)
        }
    
    /// <summary>Entropy: γ(1-1/k) + ln(λ/k) + 1
/// where γ is Euler-Mascheroni constant.
/// </summary>let entropy (shape: float) (scale: float) : FowlResult<float> =
        result {
            do! validate shape scale
            let eulerGamma = 0.5772156649015328606
            return eulerGamma * (1.0 - 1.0 / shape) + log (scale / shape) + 1.0
        }