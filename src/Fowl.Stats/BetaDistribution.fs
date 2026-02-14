module Fowl.Stats.BetaDistribution

open System
open Fowl
open Fowl.Core.Types
open Fowl.Stats.SpecialFunctions

/// <summary>Beta distribution with shape parameters alpha and beta.</summary>/// <remarks>
/// The Beta distribution is defined on [0, 1] and is commonly used in Bayesian
/// statistics, modeling proportions, and as a conjugate prior for binomial
/// and geometric distributions.
/// 
/// PDF: f(x; α, β) = x^(α-1) * (1-x)^(β-1) / B(α, β)
/// where B(α, β) is the beta function.
/// </remarks>

/// <summary>Validate Beta distribution parameters.</summary>let validateParams (alpha: float) (beta: float) : FowlResult<unit> =
    if alpha <= 0.0 then
        Error.invalidArgument "alpha must be positive"
    elif beta <= 0.0 then
        Error.invalidArgument "beta must be positive"
    else
        Ok ()

/// <summary>Probability density function for Beta distribution.</summary>/// <param name="alpha">Shape parameter α > 0.</param>/// <param name="beta">Shape parameter β > 0.</param>/// <param name="x">Point at which to evaluate PDF (0 <= x <= 1).</param>/// <returns>PDF value at x.</returns>/// <example>
/// <code>
/// let pdf = BetaDistribution.pdf 2.0 3.0 0.5  // Mode at ~0.33
/// </code>
/// </example>
let pdf (alpha: float) (beta: float) (x: float) : FowlResult<float> =
    validateParams alpha beta
    |> Result.bind (fun () ->
        if x < 0.0 || x > 1.0 then
            Ok 0.0  // PDF is 0 outside [0, 1]
        else
            result {
                let! lbeta = logBeta alpha beta
                return exp ((alpha - 1.0) * log x + (beta - 1.0) * log (1.0 - x) - lbeta)
            })

/// <summary>Cumulative distribution function for Beta distribution.</summary>/// <param name="alpha">Shape parameter α > 0.</param>/// <param name="beta">Shape parameter β > 0.</param>/// <param name="x">Point at which to evaluate CDF (0 <= x <= 1).</param>/// <returns>CDF value at x (probability that random variable ≤ x).</returns>let cdf (alpha: float) (beta: float) (x: float) : FowlResult<float> =
    validateParams alpha beta
    |> Result.bind (fun () ->
        if x <= 0.0 then
            Ok 0.0
        elif x >= 1.0 then
            Ok 1.0
        else
            incompleteBeta alpha beta x)

/// <summary>Percent point function (inverse CDF) for Beta distribution.</summary>/// <param name="alpha">Shape parameter α > 0.</param>/// <param name="beta">Shape parameter β > 0.</param>/// <param name="p">Probability (0 <= p <= 1).</param>/// <returns>Value x such that CDF(x) = p.</returns>/// <remarks>
/// Uses Newton-Raphson iteration to find the inverse.
/// </remarks>let ppf (alpha: float) (beta: float) (p: float) : FowlResult<float> =
    validateParams alpha beta
    |> Result.bind (fun () ->
        if p < 0.0 || p > 1.0 then
            Error.invalidArgument "p must be in [0, 1]"
        elif p = 0.0 then
            Ok 0.0
        elif p = 1.0 then
            Ok 1.0
        else
            // Newton-Raphson iteration
            let maxIter = 100
            let tol = 1e-12
            
            // Initial guess using approximation
            let x0 = alpha / (alpha + beta)
            
            let rec newtonRaphson x iter =
                if iter >= maxIter then
                    Ok x  // Return best estimate
                else
                    match cdf alpha beta x with
                    | Ok cdfX ->
                        let error = cdfX - p
                        if abs error < tol then
                            Ok x
                        else
                            // Approximate derivative using PDF
                            match pdf alpha beta x with
                            | Ok pdfX ->
                                if pdfX > 1e-15 then
                                    let xNew = x - error / pdfX
                                    // Ensure we stay in (0, 1)
                                    let xClamped = max 1e-10 (min (1.0 - 1e-10) xNew)
                                    newtonRaphson xClamped (iter + 1)
                                else
                                    Ok x
                            | Error e -> Error e
                    | Error e -> Error e
            
            newtonRaphson x0 0)

/// <summary>Random variate sampling from Beta distribution.</summary>/// <param name="alpha">Shape parameter α > 0.</param>/// <param name="beta">Shape parameter β > 0.</param>/// <param name="shape">Shape of output array.</param>/// <returns>Array of random samples.</returns>/// <remarks>
/// Uses the relationship: if X ~ Gamma(α, 1) and Y ~ Gamma(β, 1),
/// then X/(X+Y) ~ Beta(α, β).
/// </remarks>let rvs (alpha: float) (beta: float) (shape: Shape) : FowlResult<Ndarray<Float64, float>> =
    validateParams alpha beta
    |> Result.bind (fun () ->
        let n = Shape.numel shape
        let rng = Random()
        
        // Generate Gamma samples using shape-scale parameterization
        // For Beta, we use Gamma(alpha, 1) and Gamma(beta, 1)
        let result = Array.zeroCreate n
        
        for i = 0 to n - 1 do
            // Use Gamma distribution rvs via relationship
            // X ~ Gamma(alpha, 1), Y ~ Gamma(beta, 1)
            // X/(X+Y) ~ Beta(alpha, beta)
            let x = GammaDistribution.rvsWithState alpha 1.0 [|1|] (GammaDistribution.init()) |> fst |> Ndarray.toArray |> Array.head
            let y = GammaDistribution.rvsWithState beta 1.0 [|1|] (GammaDistribution.init()) |> fst |> Ndarray.toArray |> Array.head
            result.[i] <- x / (x + y)
        
        Ndarray.ofArray result shape)

/// <summary>Mean of Beta distribution.</summary>/// <param name="alpha">Shape parameter α > 0.</param>/// <param name="beta">Shape parameter β > 0.</param>/// <returns>Mean = α / (α + β).</returns>let mean (alpha: float) (beta: float) : FowlResult<float> =
    validateParams alpha beta
    |> Result.map (fun () -> alpha / (alpha + beta))

/// <summary>Variance of Beta distribution.</summary>/// <param name="alpha">Shape parameter α > 0.</param>/// <param name="beta">Shape parameter β > 0.</param>/// <returns>Variance = αβ / ((α + β)²(α + β + 1)).</returns>let var (alpha: float) (beta: float) : FowlResult<float> =
    validateParams alpha beta
    |> Result.map (fun () ->
        let sum = alpha + beta
        alpha * beta / (sum * sum * (sum + 1.0)))

/// <summary>Standard deviation of Beta distribution.</summary>/// <param name="alpha">Shape parameter α > 0.</param>/// <param name="beta">Shape parameter β > 0.</param>/// <returns>Standard deviation.</returns>let std (alpha: float) (beta: float) : FowlResult<float> =
    var alpha beta |> Result.map sqrt

/// <summary>Mode of Beta distribution.</summary>/// <param name="alpha">Shape parameter α > 0.</param>/// <param name="beta">Shape parameter β > 0.</param>/// <returns>Mode = (α - 1) / (α + β - 2) for α, β > 1.</returns>let mode (alpha: float) (beta: float) : FowlResult<float> =
    validateParams alpha beta
    |> Result.bind (fun () ->
        if alpha > 1.0 && beta > 1.0 then
            Ok ((alpha - 1.0) / (alpha + beta - 2.0))
        elif alpha < 1.0 && beta >= 1.0 then
            Ok 0.0  // Mode at 0
        elif alpha >= 1.0 && beta < 1.0 then
            Ok 1.0  // Mode at 1
        else
            Error.invalidState "Mode undefined when both alpha <= 1 and beta <= 1")