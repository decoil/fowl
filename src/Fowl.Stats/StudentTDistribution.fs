module Fowl.Stats.StudentTDistribution

open System
open Fowl
open Fowl.Core.Types
open Fowl.Stats.SpecialFunctions

/// <summary>Student's t-distribution with ν (nu) degrees of freedom.</summary>/// <remarks>
/// The t-distribution arises when estimating the mean of a normally distributed
/// population with unknown variance, especially in small samples. It's fundamental
/// for hypothesis testing (t-tests) and confidence intervals.
///
/// PDF: f(x; ν) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)
///
/// As ν → ∞, t-distribution approaches standard normal.
/// </remarks>

/// <summary>Inverse standard normal CDF approximation (Abramowitz & Stegun).</summary>let private inverseNormalCdf (p: float) : float =
    if p <= 0.0 then -infinity
    elif p >= 1.0 then infinity
    elif p = 0.5 then 0.0
    else
        let pp = if p < 0.5 then p else 1.0 - p
        let t = sqrt (-2.0 * log pp)
        
        // Polynomial approximation coefficients
        let c0 = 2.515517
        let c1 = 0.802853
        let c2 = 0.010328
        let d1 = 1.432788
        let d2 = 0.189269
        let d3 = 0.001308
        
        let x = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
        if p < 0.5 then -x else x

/// <summary>Validate degrees of freedom parameter.</summary>let validateParams (df: float) : FowlResult<unit> =
    if df <= 0.0 then
        Error.invalidArgument "degrees of freedom must be positive"
    else
        Ok ()

/// <summary>Probability density function for Student's t-distribution.</summary>/// <param name="df">Degrees of freedom ν > 0.</param>/// <param name="x">Point at which to evaluate PDF.</param>/// <returns>PDF value at x.</returns>/// <example>
/// <code>
/// let pdf = StudentTDistribution.pdf 10.0 0.0  // Peak at 0
/// </code>
/// </example>
let pdf (df: float) (x: float) : FowlResult<float> =
    validateParams df
    |> Result.bind (fun () ->
        result {
            // PDF formula: Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)
            let! gamma1 = gamma ((df + 1.0) / 2.0)
            let! gamma2 = gamma (df / 2.0)
            
            let coeff = gamma1 / (sqrt (df * Math.PI) * gamma2)
            let baseTerm = 1.0 + (x * x) / df
            let exponent = -(df + 1.0) / 2.0
            
            return coeff * (baseTerm ** exponent)
        })

/// <summary>Cumulative distribution function for Student's t-distribution.</summary>/// <param name="df">Degrees of freedom ν > 0.</param>/// <param name="x">Point at which to evaluate CDF.</param>/// <returns>CDF value at x.</returns>/// <remarks>
/// Uses the relationship between t-distribution CDF and incomplete beta function:
/// For x >= 0: CDF(x) = 1 - 0.5 * I_{ν/(ν+x²)}(ν/2, 1/2)
/// For x < 0: CDF(x) = 1 - CDF(-x)
/// </remarks>let cdf (df: float) (x: float) : FowlResult<float> =
    validateParams df
    |> Result.bind (fun () ->
        if x = 0.0 then
            Ok 0.5  // Symmetric about 0
        elif x > 0.0 then
            // CDF(x) = 1 - 0.5 * I_{ν/(ν+x²)}(ν/2, 1/2)
            let t = df / (df + x * x)
            result {
                let! betaVal = incompleteBeta (df / 2.0) 0.5 t
                return 1.0 - 0.5 * betaVal
            }
        else
            // CDF(-x) = 1 - CDF(x) for x > 0
            result {
                let! cdfNegX = cdf df (-x)
                return 1.0 - cdfNegX
            })

/// <summary>Percent point function (inverse CDF) for Student's t-distribution.</summary>/// <param name="df">Degrees of freedom ν > 0.</param>/// <param name="p">Probability (0 <= p <= 1).</param>/// <returns>Value x such that CDF(x) = p.</returns>/// <remarks>
/// Uses Newton-Raphson iteration to find the inverse.
/// Initial guess uses Cornish-Fisher expansion approximation.
/// </remarks>let ppf (df: float) (p: float) : FowlResult<float> =
    validateParams df
    |> Result.bind (fun () ->
        if p < 0.0 || p > 1.0 then
            Error.invalidArgument "p must be in [0, 1]"
        elif p = 0.0 then
            Ok -infinity
        elif p = 1.0 then
            Ok infinity
        elif p = 0.5 then
            Ok 0.0
        else
            // Initial guess using Cornish-Fisher expansion
            let zVal = inverseNormalCdf p
            
            let h = 1.0 / df
            let a = ((3.0 * h + 4.0) * h + 6.0) / 12.0
            let x0 = zVal + (zVal * zVal * zVal + zVal) * a / 4.0
            
            // Newton-Raphson iteration
            let maxIter = 100
            let tol = 1e-12
            
            let rec newtonRaphson x iter =
                if iter >= maxIter then
                    Ok x
                else
                    match cdf df x with
                    | Ok cdfX ->
                        let error = cdfX - p
                        if abs error < tol then
                            Ok x
                        else
                            match pdf df x with
                            | Ok pdfX ->
                                if pdfX > 1e-15 then
                                    newtonRaphson (x - error / pdfX) (iter + 1)
                                else
                                    Ok x
                            | Error e -> Error e
                    | Error e -> Error e
            
            newtonRaphson x0 0)

/// <summary>Random variate sampling from Student's t-distribution.</summary>/// <param name="df">Degrees of freedom ν > 0.</param>/// <param name="shape">Shape of output array.</param>/// <returns>Array of random samples.</returns>/// <remarks>
/// Uses the relationship: if Z ~ N(0,1) and V ~ χ²(ν) are independent,
/// then Z / √(V/ν) ~ t(ν).
/// </remarks>let rvs (df: float) (shape: Shape) : FowlResult<Ndarray<Float64, float>> =
    validateParams df
    |> Result.bind (fun () ->
        let n = Shape.numel shape
        let rng = Random()
        
        // Inline Gamma sampler for Chi-square generation
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
            // Generate standard normal
            let z = randn rng
            
            // Generate chi-squared via Gamma(ν/2, 2)
            let chi2 = sampleGamma (df / 2.0) 2.0
            
            // t = Z / sqrt(Chi2 / ν)
            result.[i] <- z / sqrt (chi2 / df)
        
        Ndarray.ofArray result shape)

/// <summary>Mean of Student's t-distribution.</summary>/// <param name="df">Degrees of freedom ν > 0.</param>/// <returns>Mean = 0 for ν > 1, undefined otherwise.</returns>let mean (df: float) : FowlResult<float> =
    validateParams df
    |> Result.bind (fun () ->
        if df > 1.0 then
            Ok 0.0
        else
            Error.invalidState "Mean undefined for degrees of freedom <= 1")

/// <summary>Variance of Student's t-distribution.</summary>/// <param name="df">Degrees of freedom ν > 0.</param>/// <returns>Variance = ν / (ν - 2) for ν > 2, undefined otherwise.</returns>let var (df: float) : FowlResult<float> =
    validateParams df
    |> Result.bind (fun () ->
        if df > 2.0 then
            Ok (df / (df - 2.0))
        elif df > 1.0 then
            Error.invalidState "Variance infinite for 1 < degrees of freedom <= 2"
        else
            Error.invalidState "Variance undefined for degrees of freedom <= 1")

/// <summary>Standard deviation of Student's t-distribution.</summary>/// <param name="df">Degrees of freedom ν > 0.</param>/// <returns>Standard deviation.</returns>let std (df: float) : FowlResult<float> =
    var df |> Result.map sqrt

/// <summary>Entropy of Student's t-distribution.</summary>/// <param name="df">Degrees of freedom ν > 0.</param>/// <returns>Entropy in nats.</returns>/// <remarks>
/// H(X) = (ν+1)/2 * [ψ((ν+1)/2) - ψ(ν/2)] + log(√ν * B(ν/2, 1/2))
/// where ψ is the digamma function.
/// </remarks>let entropy (df: float) : FowlResult<float> =
    validateParams df
    |> Result.bind (fun () ->
        result {
            // For now, approximate using numerical integration
            // Full implementation would use digamma function
            // Approximation: approaches 0.5*log(2πe) for large ν (normal entropy)
            let normalEntropy = 0.5 * log (2.0 * Math.PI * Math.E)
            // Correction for finite ν (approximate)
            let correction = 1.5 / df
            return normalEntropy + correction
        })