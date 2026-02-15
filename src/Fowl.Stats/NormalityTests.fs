module Fowl.Stats.NormalityTests

open System
open Fowl
open Fowl.Core.Types
open Fowl.Stats.GaussianDistribution

/// <summary>Result from a normality test.
/// </summary>
type NormalityTestResult = {
    Statistic: float
    PValue: float
    IsNormal: bool  // True if p >= alpha (fail to reject null)
    Alpha: float
}

/// <summary>Helper to create normality test result.
/// Null hypothesis: data comes from normal distribution.
/// </summary>
let private makeResult (statistic: float) (pValue: float) (alpha: float) : NormalityTestResult =
    {
        Statistic = statistic
        PValue = pValue
        IsNormal = pValue >= alpha  // Fail to reject null = normal
        Alpha = alpha
    }

/// <summary>Shapiro-Wilk test for normality.
/// Powerful test for small to medium samples.
/// </summary>/// <param name="sample">Sample data.</param>/// <param name="alpha">Significance level (default: 0.05).</param>/// <returns>Normality test result.</returns>/// <remarks>
/// This is an approximation for larger samples (n > 50).
/// For exact values, use statistical tables or software.
/// </remarks>
let shapiroWilk (sample: float[]) (alpha: float) : FowlResult<NormalityTestResult> =
    let n = sample.Length
    
    if n < 3 then
        Error.invalidArgument "Sample must have at least 3 observations"
    elif alpha <= 0.0 || alpha >= 1.0 then
        Error.invalidArgument "Alpha must be in (0, 1)"
    else
        // Sort the sample
        let sorted = Array.sort sample
        
        // Calculate mean
        let mean = Array.average sorted
        
        // Calculate S² = sum((x_i - mean)²)
        let s2 = sorted |> Array.map (fun x -> (x - mean) ** 2.0) |> Array.sum
        
        // For small n, use simplified approximation
        // W = (sum(a_i * x_(i)))² / S²
        // where a_i are coefficients from Shapiro-Wilk tables
        
        // Approximate coefficients (simplified)
        let m = float n
        let coefficients = 
            Array.init n (fun i ->
                let k = float (i + 1)
                // Approximate using normal distribution of order statistics
                let pk = (k - 0.375) / (m + 0.25)
                match GaussianDistribution.ppf 0.0 1.0 pk with
                | Ok z -> z
                | Error _ -> 0.0)
        
        // Normalize coefficients
        let coeffSum = coefficients |> Array.map (fun x -> x * x) |> Array.sum |> sqrt
        let normalizedCoeffs = coefficients |> Array.map (fun c -> c / coeffSum)
        
        // Calculate b = sum(a_i * x_(i))
        let b = Array.map2 (fun a x -> a * x) normalizedCoeffs sorted |> Array.sum
        
        // W statistic
        let w = (b * b) / s2
        
        // Approximate p-value using transformation
        // For simplified implementation, use approximation
        let logW = log (1.0 - w)
        let mu = -0.0006714 * (float n) + 0.0226786
        let sigma = -0.0002442 * (float n) + 0.0063341
        let z = (logW - mu) / sigma
        
        // Convert z to approximate p-value
        result {
            let! cdfVal = GaussianDistribution.cdf 0.0 1.0 z
            let pValue = 1.0 - cdfVal
            return makeResult w (max 0.0 (min 1.0 pValue)) alpha
        }

/// <summary>Anderson-Darling test for normality.
/// Sensitive to tails of distribution.
/// </summary>/// <param name="sample">Sample data.</param>/// <param name="alpha">Significance level (default: 0.05).</param>/// <returns>Normality test result.</returns>let andersonDarling (sample: float[]) (alpha: float) : FowlResult<NormalityTestResult> =
    let n = sample.Length
    
    if n < 8 then
        Error.invalidArgument "Sample must have at least 8 observations for Anderson-Darling"
    elif alpha <= 0.0 || alpha >= 1.0 then
        Error.invalidArgument "Alpha must be in (0, 1)"
    else
        // Sort sample
        let sorted = Array.sort sample
        let mean = Array.average sorted
        let std = sqrt (sorted |> Array.map (fun x -> (x - mean) ** 2.0) |> Array.average)
        
        // Standardize to Z scores
        let zScores = sorted |> Array.map (fun x -> (x - mean) / std)
        
        // Calculate CDF values
        let cdfValues = 
            zScores
            |> Array.map (fun z -
                match GaussianDistribution.cdf 0.0 1.0 z with
                | Ok p -> p
                | Error _ -> 0.5)
        
        // A² = -n - (1/n) * sum((2i-1)[ln(p_i) + ln(1-p_{n+1-i})])
        let sumTerm = 
            cdfValues
            |> Array.mapi (fun i p -
                let k = float (i + 1)
                let pRev = cdfValues.[n - i - 1]
                (2.0 * k - 1.0) * (log p + log (1.0 - pRev)))
            |> Array.sum
        
        let a2 = -float n - sumTerm / float n
        
        // Small sample correction
        let a2Star = a2 * (1.0 + 0.75 / float n + 2.25 / float n / float n)
        
        // Approximate critical values
        // For alpha = 0.05, critical value ≈ 0.787
        let criticalValue = 0.787
        let pValue = exp (-a2Star * 0.5)  // Rough approximation
        
        Ok (makeResult a2Star (max 0.0 (min 1.0 pValue)) alpha)

/// <summary>Kolmogorov-Smirnov test for normality.
/// Tests if sample differs from normal distribution.
/// </summary>/// <param name="sample">Sample data.</param>/// <param name="alpha">Significance level (default: 0.05).</param>/// <returns>Normality test result.</returns>let kolmogorovSmirnov (sample: float[]) (alpha: float) : FowlResult<NormalityTestResult> =
    let n = sample.Length
    
    if n < 5 then
        Error.invalidArgument "Sample must have at least 5 observations"
    elif alpha <= 0.0 || alpha >= 1.0 then
        Error.invalidArgument "Alpha must be in (0, 1)"
    else
        // Estimate parameters from sample
        let mean = Array.average sample
        let std = sqrt (sample |> Array.map (fun x -> (x - mean) ** 2.0) |> Array.average)
        
        // Sort sample
        let sorted = Array.sort sample
        
        // Calculate empirical CDF and theoretical CDF
        let empiricalCDF = Array.init n (fun i -> float (i + 1) / float n)
        let theoreticalCDF = 
            sorted
            |> Array.map (fun x -
                match GaussianDistribution.cdf mean std x with
                | Ok p -> p
                | Error _ -> 0.5)
        
        // D = max(|F_n(x) - F(x)|)
        let dPlus = 
            Array.zip empiricalCDF theoreticalCDF
            |> Array.map (fun (emp, theo) -> abs (emp - theo))
            |> Array.max
        
        let dMinus =
            Array.zip (Array.init n (fun i -> float i / float n)) theoreticalCDF
            |> Array.map (fun (emp, theo) -> abs (emp - theo))
            |> Array.max
        
        let d = max dPlus dMinus
        
        // Modified KS statistic for normality (Lilliefors)
        let dn = d * (sqrt (float n) + 0.12 + 0.11 / sqrt (float n))
        
        // Approximate p-value
        // For D*sqrt(n) > 1, p ≈ 0
        let pValue = 
            if dn > 1.5 then 0.01
            elif dn > 1.0 then 0.05
            elif dn > 0.8 then 0.10
            else 0.20
        
        Ok (makeResult d pValue alpha)

/// <summary>Jarque-Bera test for normality.
/// Tests based on skewness and kurtosis.
/// </summary>/// <param name="sample">Sample data.</param>/// <param name="alpha">Significance level (default: 0.05).</param>/// <returns>Normality test result.</returns>let jarqueBera (sample: float[]) (alpha: float) : FowlResult<NormalityTestResult> =
    let n = sample.Length
    
    if n < 4 then
        Error.invalidArgument "Sample must have at least 4 observations"
    elif alpha <= 0.0 || alpha >= 1.0 then
        Error.invalidArgument "Alpha must be in (0, 1)"
    else
        let mean = Array.average sample
        
        // Calculate moments
        let m2 = sample |> Array.map (fun x -> (x - mean) ** 2.0) |> Array.average
        let m3 = sample |> Array.map (fun x -> (x - mean) ** 3.0) |> Array.average
        let m4 = sample |> Array.map (fun x -> (x - mean) ** 4.0) |> Array.average
        
        // Skewness and kurtosis
        let skew = m3 / (m2 ** 1.5)
        let kurt = m4 / (m2 ** 2.0)
        
        // JB statistic = n/6 * (skew² + (kurt-3)²/4)
        let jb = float n / 6.0 * (skew * skew + (kurt - 3.0) ** 2.0 / 4.0)
        
        // JB ~ Chi-square(2)
        result {
            let! cdfVal = ChiSquareDistribution.cdf 2.0 jb
            let pValue = 1.0 - cdfVal
            return makeResult jb pValue alpha
        }