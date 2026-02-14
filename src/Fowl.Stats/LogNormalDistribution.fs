namespace Fowl.Stats

open System
open Fowl
open Fowl.Core.Types

/// <summary>Log-Normal distribution.
/// If X ~ LogNormal(μ, σ²), then ln(X) ~ Normal(μ, σ²).
/// Used extensively in financial modeling for asset prices.
/// </summary>module LogNormalDistribution =
    
    /// <summary>Validate Log-Normal parameters.
/// </summary>let private validate (mu: float) (sigma: float) : FowlResult<unit> =
        if sigma <= 0.0 then
            Error.invalidArgument "Log-Normal sigma must be positive"
        else
            Ok ()
    
    /// <summary>Probability density function.
/// PDF(x) = 1/(xσ√(2π)) * exp(-(ln(x) - μ)² / (2σ²))
/// </summary>let pdf (mu: float) (sigma: float) (x: float) : FowlResult<float> =
        result {
            do! validate mu sigma
            
            if x <= 0.0 then
                return 0.0
            else
                let lnX = log x
                let diff = lnX - mu
                let exponent = - (diff * diff) / (2.0 * sigma * sigma)
                let norm = 1.0 / (x * sigma * sqrt (2.0 * System.Math.PI))
                return norm * exp exponent
        }
    
    /// <summary>Cumulative distribution function.
/// CDF(x) = Φ((ln(x) - μ) / σ)
/// where Φ is the standard normal CDF.
/// </summary>let cdf (mu: float) (sigma: float) (x: float) : FowlResult<float> =
        result {
            do! validate mu sigma
            
            if x <= 0.0 then
                return 0.0
            else
                let z = (log x - mu) / sigma
                return! GaussianDistribution.cdf 0.0 1.0 z
        }
    
    /// <summary>Percent point function (inverse CDF).
/// PPF(p) = exp(μ + σ * Φ⁻¹(p))
/// </summary>let ppf (mu: float) (sigma: float) (p: float) : FowlResult<float> =
        result {
            do! validate mu sigma
            
            if p < 0.0 || p > 1.0 then
                return! Error.invalidArgument "Log-Normal PPF requires 0 ≤ p ≤ 1"
            elif p = 0.0 then
                return 0.0
            elif p = 1.0 then
                return Double.PositiveInfinity
            else
                let! z = GaussianDistribution.ppf 0.0 1.0 p
                return exp (mu + sigma * z)
        }
    
    /// <summary>Random variate sampling.
/// Sample from Normal(μ, σ²) and exponentiate.
/// </summary>let rvs (mu: float) (sigma: float) (seed: int option) : FowlResult<float> =
        result {
            do! validate mu sigma
            
            let rng = match seed with Some s -> Random(s) | None -> Random()
            // Box-Muller transform for normal
            let u1 = rng.NextDouble()
            let u2 = rng.NextDouble()
            let z = sqrt (-2.0 * log u1) * cos (2.0 * System.Math.PI * u2)
            return exp (mu + sigma * z)
        }
    
    /// <summary>Mean: exp(μ + σ²/2)
/// </summary>let mean (mu: float) (sigma: float) : FowlResult<float> =
        result {
            do! validate mu sigma
            return exp (mu + sigma * sigma / 2.0)
        }
    
    /// <summary>Variance: [exp(σ²) - 1] * exp(2μ + σ²)
/// </summary>let variance (mu: float) (sigma: float) : FowlResult<float> =
        result {
            do! validate mu sigma
            let sigma2 = sigma * sigma
            let term1 = exp sigma2 - 1.0
            let term2 = exp (2.0 * mu + sigma2)
            return term1 * term2
        }
    
    /// <summary>Standard deviation.
/// </summary>let std (mu: float) (sigma: float) : FowlResult<float> =
        result {
            let! var = variance mu sigma
            return sqrt var
        }
    
    /// <summary>Mode: exp(μ - σ²)
/// </summary>let mode (mu: float) (sigma: float) : FowlResult<float> =
        result {
            do! validate mu sigma
            return exp (mu - sigma * sigma)
        }
    
    /// <summary>Median: exp(μ)
/// </summary>let median (mu: float) (sigma: float) : FowlResult<float> =
        result {
            do! validate mu sigma
            return exp mu
        }
    
    /// <summary>Skewness: (exp(σ²) + 2) * √(exp(σ²) - 1)
/// </summary>let skewness (mu: float) (sigma: float) : FowlResult<float> =
        result {
            do! validate mu sigma
            let sigma2 = sigma * sigma
            let es2 = exp sigma2
            let term1 = es2 + 2.0
            let term2 = sqrt (es2 - 1.0)
            return term1 * term2
        }
    
    /// <summary>Kurtosis excess: exp(4σ²) + 2exp(3σ²) + 3exp(2σ²) - 6
/// </summary>let kurtosisExcess (mu: float) (sigma: float) : FowlResult<float> =
        result {
            do! validate mu sigma
            let sigma2 = sigma * sigma
            let es2 = exp sigma2
            let es4 = exp (4.0 * sigma2)
            let es6 = exp (6.0 * sigma2)
            return es4 + 2.0 * es2 * es2 * es2 + 3.0 * es2 * es2 - 6.0
        }
    
    /// <summary>Entropy: ln(σ * exp(μ + 1/2) * √(2π))
/// </summary>let entropy (mu: float) (sigma: float) : FowlResult<float> =
        result {
            do! validate mu sigma
            return log sigma + mu + 0.5 + 0.5 * log (2.0 * System.Math.PI)
        }