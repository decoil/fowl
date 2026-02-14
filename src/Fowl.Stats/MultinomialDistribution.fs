namespace Fowl.Stats

open System
open Fowl
open Fowl.Core.Types

/// <summary>Multinomial distribution.
/// Generalization of Binomial to multiple outcomes.
/// Models categorical data with n trials and k categories.
/// </summary>module MultinomialDistribution =
    
    /// <summary>Validate Multinomial parameters.
/// </summary>let private validate (probs: float[]) (n: int) : FowlResult<unit> =
        if n < 0 then
            Error.invalidArgument "Multinomial n must be non-negative"
        elif probs.Length = 0 then
            Error.invalidArgument "Multinomial probabilities cannot be empty"
        elif probs |> Array.exists (fun p -> p < 0.0 || p > 1.0) then
            Error.invalidArgument "Multinomial probabilities must be in [0, 1]"
        elif abs (Array.sum probs - 1.0) > 1e-10 then
            Error.invalidArgument "Multinomial probabilities must sum to 1"
        else
            Ok ()
    
    /// <summary>Logarithm of multinomial coefficient.
/// ln(n! / (x₁! x₂! ... xₖ!)) = ln(n!) - Σ ln(xᵢ!)
/// </summary>let private logMultinomialCoefficient (n: int) (counts: int[]) : float =
        // Use log-gamma: ln(n!) = ln(Γ(n+1))
        SpecialFunctions.logGamma (float n + 1.0)
        - Array.sumBy (fun x -> SpecialFunctions.logGamma (float x + 1.0)) counts
    
    /// <summary>Probability mass function.
/// PMF(x) = n!/(x₁!...xₖ!) * p₁^x₁ * ... * pₖ^xₖ
/// </summary>let pmf (probs: float[]) (n: int) (counts: int[]) : FowlResult<float> =
        result {
            do! validate probs n
            
            if counts.Length <> probs.Length then
                return! Error.invalidArgument "Multinomial counts and probabilities must have same length"
            elif counts |> Array.exists (fun x -> x < 0) then
                return! Error.invalidArgument "Multinomial counts must be non-negative"
            elif Array.sum counts <> n then
                return! Error.invalidArgument "Multinomial counts must sum to n"
            else
                // Use log-space for numerical stability
                let logCoef = logMultinomialCoefficient n counts
                let logProbs = 
                    Array.map2 (fun x p -
                        if x = 0 then 0.0 else float x * log p) counts probs
                    |> Array.sum
                return exp (logCoef + logProbs)
        }
    
    /// <summary>Random variate sampling.
/// Sample from categorical distribution n times and count outcomes.
/// </summary>let rvs (probs: float[]) (n: int) (seed: int option) : FowlResult<int[]> =
        result {
            do! validate probs n
            
            let rng = match seed with Some s -> Random(s) | None -> Random()
            let k = probs.Length
            let counts = Array.zeroCreate k
            
            // Cumulative probabilities for efficient sampling
            let cumProbs = 
                probs
                |> Array.scan (+) 0.0
                |> Array.skip 1
            
            // Sample n times
            for _ = 1 to n do
                let u = rng.NextDouble()
                // Find which bin this falls into
                let idx = 
                    cumProbs
                    |> Array.findIndex (fun cp -> u <= cp)
                counts.[idx] <- counts.[idx] + 1
            
            return counts
        }
    
    /// <summary>Mean for each category: E[Xᵢ] = n * pᵢ
/// </summary>let mean (probs: float[]) (n: int) : FowlResult<float[]> =
        result {
            do! validate probs n
            return probs |> Array.map (fun p -> float n * p)
        }
    
    /// <summary>Variance for each category: Var(Xᵢ) = n * pᵢ * (1 - pᵢ)
/// </summary>let variance (probs: float[]) (n: int) : FowlResult<float[]> =
        result {
            do! validate probs n
            return probs |> Array.map (fun p -> float n * p * (1.0 - p))
        }
    
    /// <summary>Standard deviation for each category.
/// </summary>let std (probs: float[]) (n: int) : FowlResult<float[]> =
        result {
            let! var = variance probs n
            return var |> Array.map sqrt
        }
    
    /// <summary>Covariance between categories: Cov(Xᵢ, Xⱼ) = -n * pᵢ * pⱼ (i ≠ j)
/// For i = j, Cov(Xᵢ, Xᵢ) = Var(Xᵢ)
/// </summary>let covariance (probs: float[]) (n: int) : FowlResult<float[,]> =
        result {
            do! validate probs n
            let k = probs.Length
            let cov = Array2D.zeroCreate k k
            
            for i = 0 to k - 1 do
                for j = 0 to k - 1 do
                    if i = j then
                        // Variance
                        cov.[i, j] <- float n * probs.[i] * (1.0 - probs.[i])
                    else
                        // Covariance
                        cov.[i, j] <- -float n * probs.[i] * probs.[j]
            
            return cov
        }
    
    /// <summary>Marginal distribution for single category: Binomial(n, pᵢ)
/// </summary>let marginal (probs: float[]) (n: int) (idx: int) : FowlResult<float> =
        result {
            do! validate probs n
            
            if idx < 0 || idx >= probs.Length then
                return! Error.invalidArgument "Multinomial marginal index out of range"
            else
                // Returns probability mass for this category
                return probs.[idx]
        }
    
    /// <summary>Entropy (approximate): H = -Σ pᵢ ln(pᵢ) * n
/// This is the entropy per trial times n.
/// </summary>let entropy (probs: float[]) (n: int) : FowlResult<float> =
        result {
            do! validate probs n
            let h = 
                probs
                |> Array.sumBy (fun p -
                    if p = 0.0 then 0.0 else -p * log p)
            return float n * h
        }