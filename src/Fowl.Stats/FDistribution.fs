module Fowl.Stats.FDistribution

open System
open Fowl
open Fowl.Core.Types
open Fowl.Stats.SpecialFunctions

/// <summary>F-distribution (Fisher-Snedecor) with d1 and d2 degrees of freedom.</summary>/// <remarks>
/// The F-distribution arises frequently as the null distribution of a test statistic,
/// especially in analysis of variance (ANOVA) and in the F-test to compare two
/// variances or nested models.
///
/// PDF: f(x; d1, d2) = √(d1^k1 * d2^k2 * x^(k1-1)) / (B(k1, k2) * (d2 + d1*x)^(k1+k2))
/// where k1 = d1/2, k2 = d2/2, and B is the beta function.
///
/// Relationship: If X ~ Chi-square(d1) and Y ~ Chi-square(d2), then
/// (X/d1) / (Y/d2) ~ F(d1, d2)
/// </remarks>

/// <summary>Validate F-distribution parameters.</summary>let validateParams (d1: float) (d2: float) : FowlResult<unit> =
    if d1 <= 0.0 then
        Error.invalidArgument "d1 (numerator df) must be positive"
    elif d2 <= 0.0 then
        Error.invalidArgument "d2 (denominator df) must be positive"
    else
        Ok ()

/// <summary>Probability density function for F-distribution.</summary>/// <param name="d1">Numerator degrees of freedom > 0.</param>/// <param name="d2">Denominator degrees of freedom > 0.</param>/// <param name="x">Point at which to evaluate PDF (x >= 0).</param>/// <returns>PDF value at x.</returns>/// <example>
/// <code>
/// let pdf = FDistribution.pdf 5.0 10.0 1.0
/// </code>
/// </example>
let pdf (d1: float) (d2: float) (x: float) : FowlResult<float> =
    validateParams d1 d2
    |> Result.bind (fun () ->
        if x < 0.0 then
            Ok 0.0  // PDF is 0 for x < 0
        else
            result {
                let k1 = d1 / 2.0
                let k2 = d2 / 2.0
                
                // PDF formula using log for numerical stability
                let! lbeta = logBeta k1 k2
                let logNum = k1 * log d1 + k2 * log d2 + (k1 - 1.0) * log x
                let logDen = lbeta + (k1 + k2) * log (d2 + d1 * x)
                
                return exp (logNum - logDen)
            })

/// <summary>Cumulative distribution function for F-distribution.</summary>/// <param name="d1">Numerator degrees of freedom > 0.</param>/// <param name="d2">Denominator degrees of freedom > 0.</param>/// <param name="x">Point at which to evaluate CDF (x >= 0).</param>/// <returns>CDF value at x.</returns>/// <remarks>
/// Uses the relationship to incomplete beta function:
/// CDF(x) = I_{d1*x/(d2+d1*x)}(d1/2, d2/2)
/// where I is the regularized incomplete beta function.
/// </remarks>
let cdf (d1: float) (d2: float) (x: float) : FowlResult<float> =
    validateParams d1 d2
    |> Result.bind (fun () ->
        if x <= 0.0 then
            Ok 0.0
        else
            // CDF(x) = I_{d1*x/(d2+d1*x)}(d1/2, d2/2)
            let k1 = d1 / 2.0
            let k2 = d2 / 2.0
            let z = d1 * x / (d2 + d1 * x)
            incompleteBeta k1 k2 z)

/// <summary>Percent point function (inverse CDF) for F-distribution.</summary>/// <param name="d1">Numerator degrees of freedom > 0.</param>/// <param name="d2">Denominator degrees of freedom > 0.</param>/// <param name="p">Probability (0 <= p <= 1).</param>/// <returns>Value x such that CDF(x) = p.</returns>/// <remarks>
/// Uses the relationship: if F ~ F(d1, d2), then
/// F = (d2/d1) * (X/(1-X)) where X ~ Beta(d1/2, d2/2)
/// So F^{-1}(p) = (d2/d1) * (Beta^{-1}(p) / (1 - Beta^{-1}(p)))
/// </remarks>
let ppf (d1: float) (d2: float) (p: float) : FowlResult<float> =
    validateParams d1 d2
    |> Result.bind (fun () ->
        if p < 0.0 || p > 1.0 then
            Error.invalidArgument "p must be in [0, 1]"
        elif p = 0.0 then
            Ok 0.0
        elif p = 1.0 then
            Ok infinity
        else
            // F^{-1}(p) = (d2/d1) * (Beta^{-1}(p) / (1 - Beta^{-1}(p)))
            result {
                let! betaInv = BetaDistribution.ppf (d1 / 2.0) (d2 / 2.0) p
                return (d2 / d1) * (betaInv / (1.0 - betaInv))
            })

/// <summary>Random variate sampling from F-distribution.</summary>/// <param name="d1">Numerator degrees of freedom > 0.</param>/// <param name="d2">Denominator degrees of freedom > 0.</param>/// <param name="shape">Shape of output array.</param>/// <returns>Array of random samples.</returns>/// <remarks>
/// Uses the relationship: F(d1, d2) = (Chi-square(d1)/d1) / (Chi-square(d2)/d2)
/// Or equivalently: F(d1, d2) = (d2/d1) * (X/(1-X)) where X ~ Beta(d1/2, d2/2)
/// </remarks>
let rvs (d1: float) (d2: float) (shape: Shape) : FowlResult<Ndarray<Float64, float>> =
    validateParams d1 d2
    |> Result.bind (fun () ->
        let n = Shape.numel shape
        let rng = Random()
        
        // Inline Gamma sampler for Beta generation
        let sampleGamma (shape: float) (scale: float) : float =
            if shape >= 1.0 then
                let d = shape - 1.0 / 3.0
                let c = 1.0 / sqrt (9.0 * d)
                
                let rec loop () =
                    let z = SpecialFunctions.randn rng
                    let u = rng.NextDouble()
                    let v = (1.0 + c * z) ** 3.0
                    
                    if z > -1.0 / c && log u < 0.5 * z * z + d - d * v + d * log v then
                        d * v * scale
                    else
                        loop ()
                
                loop ()
            else
                let rec loop () =
                    let d = (shape + 1.0) - 1.0 / 3.0
                    let c = 1.0 / sqrt (9.0 * d)
                    let z = SpecialFunctions.randn rng
                    let u = rng.NextDouble()
                    let v = (1.0 + c * z) ** 3.0
                    
                    if z > -1.0 / c && log u < 0.5 * z * z + d - d * v + d * log v then
                        let sample = d * v
                        let u' = rng.NextDouble()
                        sample * (u' ** (1.0 / shape)) * scale
                    else
                        loop ()
                loop ()
        
        let result = Array.zeroCreate n
        
        for i = 0 to n - 1 do
            // Generate Beta(d1/2, d2/2) via Gamma ratio
            let x = sampleGamma (d1 / 2.0) 1.0
            let y = sampleGamma (d2 / 2.0) 1.0
            let betaSample = x / (x + y)
            
            // F = (d2/d1) * (Beta / (1 - Beta))
            result.[i] <- (d2 / d1) * (betaSample / (1.0 - betaSample))
        
        Ndarray.ofArray result shape)

/// <summary>Mean of F-distribution.</summary>/// <param name="d1">Numerator degrees of freedom > 0.</param>/// <param name="d2">Denominator degrees of freedom > 2.</param>/// <returns>Mean = d2 / (d2 - 2) for d2 > 2.</returns>let mean (d1: float) (d2: float) : FowlResult<float> =
    validateParams d1 d2
    |> Result.bind (fun () ->
        if d2 > 2.0 then
            Ok (d2 / (d2 - 2.0))
        else
            Error.invalidState "Mean undefined for d2 <= 2")

/// <summary>Variance of F-distribution.</summary>/// <param name="d1">Numerator degrees of freedom > 0.</param>/// <param name="d2">Denominator degrees of freedom > 4.</param>/// <returns>Variance = 2*d2²(d1+d2-2) / (d1*(d2-2)²*(d2-4)) for d2 > 4.</returns>let var (d1: float) (d2: float) : FowlResult<float> =
    validateParams d1 d2
    |> Result.bind (fun () ->
        if d2 > 4.0 then
            let num = 2.0 * d2 * d2 * (d1 + d2 - 2.0)
            let den = d1 * (d2 - 2.0) * (d2 - 2.0) * (d2 - 4.0)
            Ok (num / den)
        elif d2 > 2.0 then
            Error.invalidState "Variance infinite for 2 < d2 <= 4"
        else
            Error.invalidState "Variance undefined for d2 <= 2")

/// <summary>Standard deviation of F-distribution.</summary>/// <param name="d1">Numerator degrees of freedom > 0.</param>/// <param name="d2">Denominator degrees of freedom > 4.</param>/// <returns>Standard deviation.</returns>let std (d1: float) (d2: float) : FowlResult<float> =
    var d1 d2 |> Result.map sqrt

/// <summary>Mode of F-distribution.</summary>/// <param name="d1">Numerator degrees of freedom > 2.</param>/// <param name="d2">Denominator degrees of freedom > 0.</param>/// <returns>Mode = d2(d1-2) / (d1(d2+2)) for d1 > 2.</returns>let mode (d1: float) (d2: float) : FowlResult<float> =
    validateParams d1 d2
    |> Result.bind (fun () ->
        if d1 > 2.0 then
            Ok (d2 * (d1 - 2.0) / (d1 * (d2 + 2.0)))
        else
            Error.invalidState "Mode undefined for d1 <= 2")