namespace Fowl.Stats

open System
open Fowl
open Fowl.Core.Types

/// <summary>
/// Cauchy distribution (Lorentz distribution).
/// Heavy-tailed distribution used in robust statistics.
/// PDF(x) = 1/(πγ(1 + ((x-x₀)/γ)²))
/// </summary>
module CauchyDistribution =
    
    /// <summary>
    /// Probability density function.
    /// </summary>
    let pdf (x0: float) (gamma: float) (x: float) : FowlResult<float> =
        result {
            if gamma <= 0.0 then
                return! Error.invalidArgument "Cauchy gamma must be positive"
            
            let diff = x - x0
            let denom = System.Math.PI * gamma * (1.0 + (diff/gamma) ** 2.0)
            return 1.0 / denom
        }
    
    /// <summary>
    /// Cumulative distribution function.
    /// CDF(x) = 1/π * arctan((x-x₀)/γ) + 0.5
    /// </summary>
    let cdf (x0: float) (gamma: float) (x: float) : FowlResult<float> =
        result {
            if gamma <= 0.0 then
                return! Error.invalidArgument "Cauchy gamma must be positive"
            
            return 1.0 / System.Math.PI * atan ((x - x0) / gamma) + 0.5
        }
    
    /// <summary>
    /// Percent point function (inverse CDF).
    /// PPF(p) = x₀ + γ * tan(π(p - 0.5))
    /// </summary>
    let ppf (x0: float) (gamma: float) (p: float) : FowlResult<float> =
        result {
            if gamma <= 0.0 then
                return! Error.invalidArgument "Cauchy gamma must be positive"
            elif p < 0.0 || p > 1.0 then
                return! Error.invalidArgument "Cauchy PPF requires 0 ≤ p ≤ 1"
            elif p = 0.0 then
                return Double.NegativeInfinity
            elif p = 1.0 then
                return Double.PositiveInfinity
            else
                return x0 + gamma * tan (System.Math.PI * (p - 0.5))
        }
    
    /// <summary>
    /// Random variate sampling.
    /// Uses inverse transform sampling: x = x₀ + γ * tan(π(u - 0.5))
    /// </summary>
    let rvs (x0: float) (gamma: float) (seed: int option) : FowlResult<float> =
        result {
            if gamma <= 0.0 then
                return! Error.invalidArgument "Cauchy gamma must be positive"
            
            let rng = match seed with Some s -> Random(s) | None -> Random()
            let u = rng.NextDouble()
            return x0 + gamma * tan (System.Math.PI * (u - 0.5))
        }
    
    /// <summary>
    /// Median: x₀
    /// </summary>
    let median (x0: float) (gamma: float) : FowlResult<float> =
        result {
            if gamma <= 0.0 then
                return! Error.invalidArgument "Cauchy gamma must be positive"
            return x0
        }
    
    /// <summary>
    /// Mode: x₀
    /// </summary>
    let mode (x0: float) (gamma: float) : FowlResult<float> =
        result {
            if gamma <= 0.0 then
                return! Error.invalidArgument "Cauchy gamma must be positive"
            return x0
        }
    
    /// <summary>
    /// Mean: undefined (infinite)
    /// Cauchy distribution has no finite mean.
    /// </summary>
    let mean (x0: float) (gamma: float) : FowlResult<float> =
        result {
            if gamma <= 0.0 then
                return! Error.invalidArgument "Cauchy gamma must be positive"
            return! Error.invalidState "Cauchy distribution has no finite mean"
        }
    
    /// <summary>
    /// Variance: undefined (infinite)
    /// Cauchy distribution has no finite variance.
    /// </summary>
    let variance (x0: float) (gamma: float) : FowlResult<float> =
        result {
            if gamma <= 0.0 then
                return! Error.invalidArgument "Cauchy gamma must be positive"
            return! Error.invalidState "Cauchy distribution has no finite variance"
        }
    
    /// <summary>
    /// Standard deviation: undefined
    /// </summary>
    let std (x0: float) (gamma: float) : FowlResult<float> =
        result {
            if gamma <= 0.0 then
                return! Error.invalidArgument "Cauchy gamma must be positive"
            return! Error.invalidState "Cauchy distribution has no finite standard deviation"
        }
    
    /// <summary>
    /// Entropy: ln(4πγ)
    /// </summary>
    let entropy (x0: float) (gamma: float) : FowlResult<float> =
        result {
            if gamma <= 0.0 then
                return! Error.invalidArgument "Cauchy gamma must be positive"
            return log (4.0 * System.Math.PI * gamma)
        }
    
    /// <summary>
    /// Interquartile range: 2γ
    /// Distance between 25th and 75th percentiles.
    /// </summary>
    let iqr (x0: float) (gamma: float) : FowlResult<float> =
        result {
            if gamma <= 0.0 then
                return! Error.invalidArgument "Cauchy gamma must be positive"
            return 2.0 * gamma
        }