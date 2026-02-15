namespace Fowl.Stats

open System
open Fowl
open Fowl.Core.Types

/// <summary>Dirichlet distribution.
/// Multivariate generalization of Beta distribution.
/// Conjugate prior to Multinomial in Bayesian inference.
/// </summary>
module DirichletDistribution =
    
    /// <summary>Validate Dirichlet parameters.
/// All alpha must be positive.
/// </summary>
    let private validate (alpha: float[]) : FowlResult<unit> =
        if alpha.Length = 0 then
            Error.invalidArgument "Dirichlet alpha cannot be empty"
        elif alpha |> Array.exists (fun a -> a <= 0.0) then
            Error.invalidArgument "Dirichlet alpha must all be positive"
        else
            Ok ()
    
    /// <summary>Log of multivariate beta function (normalizing constant).
/// ln(B(α)) = Σ ln(Γ(αᵢ)) - ln(Γ(Σαᵢ))
/// </summary>
    let private logMultivariateBeta (alpha: float[]) : float =
        let sumAlpha = Array.sum alpha
        let sumLogGamma = alpha |> Array.sumBy SpecialFunctions.logGamma
        sumLogGamma - SpecialFunctions.logGamma sumAlpha
    
    /// <summary>Probability density function.
/// PDF(x) = 1/B(α) * Π xᵢ^(αᵢ-1)
/// where xᵢ > 0 and Σ xᵢ = 1.
/// </summary>
    let pdf (alpha: float[]) (x: float[]) : FowlResult<float> =
        result {
            do! validate alpha
            
            if x.Length <> alpha.Length then
                return! Error.invalidArgument "Dirichlet x and alpha must have same length"
            elif x |> Array.exists (fun xi -> xi < 0.0) then
                return! Error.invalidArgument "Dirichlet x must all be non-negative"
            elif abs (Array.sum x - 1.0) > 1e-10 then
                return! Error.invalidArgument "Dirichlet x must sum to 1"
            else
                let logB = logMultivariateBeta alpha
                let logProd = 
                    Array.map2 (fun xi ai -
                        if xi = 0.0 then
                            if ai < 1.0 then 
                                // Limit as x->0 of x^(a-1) = infinity if a < 1
                                Double.PositiveInfinity
                            else 
                                0.0
                        else
                            (ai - 1.0) * log xi) x alpha
                    |> Array.sum
                
                if Double.IsInfinity logProd then
                    return 0.0
                else
                    return exp (logProd - logB)
        }
    
    /// <summary>Random variate sampling.
/// Sample k Gamma(αᵢ, 1) variables and normalize.
/// </summary>
    let rvs (alpha: float[]) (seed: int option) : FowlResult<float[]> =
        result {
            do! validate alpha
            
            let rng = match seed with Some s -> Random(s) | None -> Random()
            let k = alpha.Length
            
            // Sample from Gamma(αᵢ, 1) for each component
            let gammaSamples = 
                alpha
                |> Array.map (fun ai -
                    // Marsaglia-Tsang for Gamma
                    let d = ai - 1.0/3.0
                    let c = 1.0 / sqrt (9.0 * d)
                    let mutable found = false
                    let mutable result = 0.0
                    while not found do
                        let z = rng.NextGaussian()
                        let u = rng.NextDouble()
                        let v = pown (1.0 + c * z) 3
                        if z > -1.0/c && log u < 0.5 * z * z + d - d * v + d * log v then
                            result <- d * v
                            found <- true
                    result)
            
            // Normalize to sum to 1
            let sum = Array.sum gammaSamples
            return gammaSamples |> Array.map (fun g -> g / sum)
        }
    
    /// <summary>Mean for each component: E[Xᵢ] = αᵢ / Σα
/// </summary>
    let mean (alpha: float[]) : FowlResult<float[]> =
        result {
            do! validate alpha
            let sumAlpha = Array.sum alpha
            return alpha |> Array.map (fun ai -> ai / sumAlpha)
        }
    
    /// <summary>Variance for each component: Var(Xᵢ) = αᵢ(Σα - αᵢ) / (Σα)²(Σα + 1)
/// </summary>
    let variance (alpha: float[]) : FowlResult<float[]> =
        result {
            do! validate alpha
            let sumAlpha = Array.sum alpha
            let denom = sumAlpha * sumAlpha * (sumAlpha + 1.0)
            return alpha |> Array.map (fun ai -> ai * (sumAlpha - ai) / denom)
        }
    
    /// <summary>Mode for each component: (αᵢ - 1) / (Σα - k) for αᵢ > 1
/// If any αᵢ ≤ 1, mode is at boundary (0 for that component).
/// </summary>
    let mode (alpha: float[]) : FowlResult<float[]> =
        result {
            do! validate alpha
            let k = float alpha.Length
            let sumAlpha = Array.sum alpha
            
            if sumAlpha <= k then
                // Mode is at boundary
                return! Error.invalidState "Dirichlet mode not defined when sum(alpha) <= k"
            else
                let denom = sumAlpha - k
                return alpha |> Array.map (fun ai -> (ai - 1.0) / denom)
        }
    
    /// <summary>Covariance matrix: Cov(Xᵢ, Xⱼ) = -αᵢαⱼ / (Σα)²(Σα + 1) for i ≠ j
/// Var(Xᵢ) = αᵢ(Σα - αᵢ) / (Σα)²(Σα + 1)
/// </summary>
    let covariance (alpha: float[]) : FowlResult<float[,]> =
        result {
            do! validate alpha
            let k = alpha.Length
            let sumAlpha = Array.sum alpha
            let denom = sumAlpha * sumAlpha * (sumAlpha + 1.0)
            
            let cov = Array2D.zeroCreate k k
            for i = 0 to k - 1 do
                for j = 0 to k - 1 do
                    if i = j then
                        cov.[i, j] <- alpha.[i] * (sumAlpha - alpha.[i]) / denom
                    else
                        cov.[i, j] <- -alpha.[i] * alpha.[j] / denom
            return cov
        }
    
    /// <summary>Marginal distribution for single component: Beta(αᵢ, Σα - αᵢ)
/// </summary>
    let marginal (alpha: float[]) (idx: int) : FowlResult<(float * float)> =
        result {
            do! validate alpha
            
            if idx < 0 || idx >= alpha.Length then
                return! Error.invalidArgument "Dirichlet marginal index out of range"
            else
                let sumAlpha = Array.sum alpha
                return (alpha.[idx], sumAlpha - alpha.[idx])
        }
    
    /// <summary>Entropy: ln(B(α)) + (Σα - k)ψ(Σα) - Σ(αᵢ - 1)ψ(αᵢ)
/// where ψ is the digamma function.
/// Simplified approximation.
/// </summary>
    let entropy (alpha: float[]) : FowlResult<float> =
        result {
            do! validate alpha
            let k = alpha.Length
            let sumAlpha = Array.sum alpha
            let logB = logMultivariateBeta alpha
            
            // Approximate digamma: ψ(x) ≈ ln(x) - 1/(2x)
            let digamma x = log x - 1.0 / (2.0 * x)
            
            let term1 = (sumAlpha - float k) * digamma sumAlpha
            let term2 = alpha |> Array.sumBy (fun ai -> (ai - 1.0) * digamma ai)
            
            return logB + term1 - term2
        }
    
    /// <summary>Concentration parameter: Σα
/// Higher values = more concentrated around mean.
/// </summary>
    let concentration (alpha: float[]) : FowlResult<float> =
        result {
            do! validate alpha
            return Array.sum alpha
        }
    
    /// <summary>Symmetric Dirichlet: all αᵢ equal.
/// Creates uniform prior when αᵢ = 1.
/// </summary>
    let symmetric (alpha: float) (k: int) : FowlResult<float[]> =
        result {
            if alpha <= 0.0 then
                return! Error.invalidArgument "Symmetric Dirichlet alpha must be positive"
            elif k <= 0 then
                return! Error.invalidArgument "Symmetric Dirichlet k must be positive"
            else
                return Array.create k alpha
        }