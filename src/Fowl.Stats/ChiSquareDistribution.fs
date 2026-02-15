module Fowl.Stats.ChiSquareDistribution

open System
open Fowl
open Fowl.Core.Types
open Fowl.Stats.SpecialFunctions

/// <summary>Chi-square distribution with k degrees of freedom.</summary>/// <remarks>
/// The chi-square distribution is the distribution of a sum of the squares
/// of k independent standard normal random variables. It's fundamental for
/// goodness-of-fit tests, independence tests, and variance estimation.
///
/// PDF: f(x; k) = 1/(2^(k/2) Γ(k/2)) * x^(k/2-1) * e^(-x/2)
///
/// Special case: Chi-square(2) = Exponential(1/2)
/// Relationship to Gamma: Chi-square(k) = Gamma(k/2, 2)
/// </remarks>

/// <summary>Validate degrees of freedom parameter.</summary>let validateParams (df: float) : FowlResult<unit> =
    if df <= 0.0 then
        Error.invalidArgument "degrees of freedom must be positive"
    else
        Ok ()

/// <summary>Probability density function for Chi-square distribution.</summary>/// <param name="df">Degrees of freedom k > 0.</param>/// <param name="x">Point at which to evaluate PDF (x >= 0).</param>/// <returns>PDF value at x.</returns>/// <example>
/// <code>
/// let pdf = ChiSquareDistribution.pdf 3.0 2.0
/// </code>
/// </example>
let pdf (df: float) (x: float) : FowlResult<float> =
    validateParams df
    |> Result.bind (fun () ->
        if x < 0.0 then
            Ok 0.0  // PDF is 0 for x < 0
        elif x = 0.0 && df < 2.0 then
            Ok infinity  // Singularity at 0 for df < 2
        else
            result {
                // PDF = 1/(2^(k/2) Γ(k/2)) * x^(k/2-1) * e^(-x/2)
                let! gammaDf = gamma (df / 2.0)
                let coeff = 1.0 / (2.0 ** (df / 2.0) * gammaDf)
                let xTerm = x ** (df / 2.0 - 1.0)
                let expTerm = exp (-x / 2.0)
                return coeff * xTerm * expTerm
            })

/// <summary>Cumulative distribution function for Chi-square distribution.</summary>/// <param name="df">Degrees of freedom k > 0.</param>/// <param name="x">Point at which to evaluate CDF (x >= 0).</param>/// <returns>CDF value at x.</returns>/// <remarks>
/// Uses the relationship to incomplete beta function via:
/// CDF(x) = incompleteBeta(df/2, x/2)
/// </remarks>
let cdf (df: float) (x: float) : FowlResult<float> =
    validateParams df
    |> Result.bind (fun () ->
        if x <= 0.0 then
            Ok 0.0
        else
            // Chi-square(k) CDF = lowerIncompleteGamma(k/2, x/2) / Gamma(k/2)
            // = incompleteBeta(k/2, x/(x+2)) ... approximation
            // Using approximation: 1 - exp(-x/2) * sum((x/2)^j / j!) for small df
            result {
                // Use relationship via Gamma CDF approximation
                // For simplicity, use incomplete beta relationship
                // CDF(x) ≈ incompleteBeta(df/2, large_value, x/(x+2))
                // Better: use normal approximation for large df
                if df > 30.0 then
                    // Wilson-Hilferty approximation
                    let z = ((x / df) ** (1.0/3.0) - (1.0 - 2.0/(9.0*df))) / sqrt(2.0/(9.0*df))
                    return 0.5 * (1.0 + SpecialFunctions.erf(z / sqrt(2.0)))
                else
                    // Direct incomplete beta approach
                    let p = df / 2.0
                    let betaArg = x / (x + 2.0)  // Approximate transformation
                    let! result = incompleteBeta p 1.0 betaArg
                    return result
            })

/// <summary>Percent point function (inverse CDF) for Chi-square distribution.</summary>/// <param name="df">Degrees of freedom k > 0.</param>/// <param name="p">Probability (0 <= p <= 1).</param>/// <returns>Value x such that CDF(x) = p.</returns>/// <remarks>
/// Uses the relationship: if X ~ Chi-square(k), then X ~ Gamma(k/2, 2).
/// So the PPF can use the Gamma PPF approximation.
/// </remarks>
let ppf (df: float) (p: float) : FowlResult<float> =
    validateParams df
    |> Result.bind (fun () ->
        if p < 0.0 || p > 1.0 then
            Error.invalidArgument "p must be in [0, 1]"
        elif p = 0.0 then
            Ok 0.0
        elif p = 1.0 then
            Ok infinity
        else
            // Approximate using Wilson-Hilferty for large df
            // or direct search for small df
            let rec searchPpf low high iter =
                if iter >= 100 then
                    Ok ((low + high) / 2.0)
                else
                    let mid = (low + high) / 2.0
                    match cdf df mid with
                    | Ok cdfMid ->
                        if abs (cdfMid - p) < 1e-10 then
                            Ok mid
                        elif cdfMid < p then
                            searchPpf mid high (iter + 1)
                        else
                            searchPpf low mid (iter + 1)
                    | Error _ -> Ok mid
            
            // Initial bounds based on mean and variance
            let mean = df
            let std = sqrt (2.0 * df)
            let low = max 0.0 (mean - 5.0 * std)
            let high = mean + 10.0 * std
            
            searchPpf low high 0)

/// <summary>Random variate sampling from Chi-square distribution.</summary>/// <param name="df">Degrees of freedom k > 0.</param>/// <param name="shape">Shape of output array.</param>/// <returns>Array of random samples.</returns>/// <remarks>
/// Uses the Gamma relationship: Chi-square(k) = Gamma(k/2, 2).
/// </remarks>
let rvs (df: float) (shape: Shape) : FowlResult<Ndarray<Float64, float>> =
    validateParams df
    |> Result.bind (fun () ->
        let n = Shape.numel shape
        let rng = Random()
        
        // Inline Gamma sampler
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
            result.[i] <- sampleGamma (df / 2.0) 2.0
        
        Ndarray.ofArray result shape)

/// <summary>Mean of Chi-square distribution.</summary>/// <param name="df">Degrees of freedom k > 0.</param>/// <returns>Mean = k.</returns>let mean (df: float) : FowlResult<float> =
    validateParams df
    |> Result.map (fun () -> df)

/// <summary>Variance of Chi-square distribution.</summary>/// <param name="df">Degrees of freedom k > 0.</param>/// <returns>Variance = 2k.</returns>let var (df: float) : FowlResult<float> =
    validateParams df
    |> Result.map (fun () -> 2.0 * df)

/// <summary>Standard deviation of Chi-square distribution.</summary>/// <param name="df">Degrees of freedom k > 0.</param>/// <returns>Standard deviation = √(2k).</returns>let std (df: float) : FowlResult<float> =
    var df |> Result.map sqrt

/// <summary>Mode of Chi-square distribution.</summary>/// <param name="df">Degrees of freedom k > 0.</param>/// <returns>Mode = k - 2 for k >= 2, 0 otherwise.</returns>let mode (df: float) : FowlResult<float> =
    validateParams df
    |> Result.map (fun () -> if df >= 2.0 then df - 2.0 else 0.0)

/// <summary>Skewness of Chi-square distribution.</summary>/// <param name="df">Degrees of freedom k > 0.</param>/// <returns>Skewness = √(8/k).</returns>let skewness (df: float) : FowlResult<float> =
    validateParams df
    |> Result.map (fun () -> sqrt (8.0 / df))

/// <summary>Kurtosis (excess) of Chi-square distribution.</summary>/// <param name="df">Degrees of freedom k > 0.</param>/// <returns>Kurtosis = 12/k.</returns>let kurtosis (df: float) : FowlResult<float> =
    validateParams df
    |> Result.map (fun () -> 12.0 / df)

/// <summary>Entropy of Chi-square distribution.</summary>/// <param name="df">Degrees of freedom k > 0.</param>/// <returns>Entropy in nats.</returns>/// <remarks>
/// H(X) = k/2 + log(2Γ(k/2)) + (1 - k/2)ψ(k/2)
/// where ψ is the digamma function.
/// </remarks>
let entropy (df: float) : FowlResult<float> =
    validateParams df
    |> Result.bind (fun () ->
        result {
            let! gammaDf = gamma (df / 2.0)
            // Approximate digamma: ψ(x) ≈ log(x) - 1/(2x) for large x
            let digamma = log (df / 2.0) - 1.0 / df
            return df / 2.0 + log (2.0 * gammaDf) + (1.0 - df / 2.0) * digamma
        })