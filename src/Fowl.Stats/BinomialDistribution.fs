module Fowl.Stats.BinomialDistribution

open System
open Fowl
open Fowl.Core.Types
open Fowl.Stats.SpecialFunctions

/// <summary>Binomial distribution with n trials and success probability p.</summary>/// <remarks>
/// The binomial distribution models the number of successes in n independent
/// Bernoulli trials with success probability p. It's the foundation for
/// hypothesis testing about proportions.
///
/// PMF: P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
/// where C(n,k) is the binomial coefficient.
/// </remarks>

/// <summary>Validate binomial distribution parameters.</summary>let validateParams (n: int) (p: float) : FowlResult<unit> =
    if n < 0 then
        Error.invalidArgument "n must be non-negative"
    elif p < 0.0 || p > 1.0 then
        Error.invalidArgument "p must be in [0, 1]"
    else
        Ok ()

/// <summary>Log binomial coefficient log(C(n,k)).</summary>let private logBinomial (n: int) (k: int) : float =
    if k < 0 || k > n then
        -infinity
    else
        // Use log gamma: log(C(n,k)) = log(n!) - log(k!) - log((n-k)!)
        // = logGamma(n+1) - logGamma(k+1) - logGamma(n-k+1)
        match SpecialFunctions.gamma (float n + 1.0) with
        | Ok gn1 ->
            match SpecialFunctions.gamma (float k + 1.0) with
            | Ok gk1 ->
                match SpecialFunctions.gamma (float (n - k) + 1.0) with
                | Ok gnk1 -> log gn1 - log gk1 - log gnk1
                | Error _ -> -infinity
            | Error _ -> -infinity
        | Error _ -> -infinity

/// <summary>Probability mass function for Binomial distribution.</summary>/// <param name="n">Number of trials (n >= 0).</param>/// <param name="p">Success probability (0 <= p <= 1).</param>/// <param name="k">Number of successes.</param>/// <returns>PMF value P(X = k).</returns>/// <example>
/// <code>
/// let pmf = BinomialDistribution.pmf 10 0.3 3  // P(X = 3)
/// </code>
/// </example>
let pmf (n: int) (p: float) (k: int) : FowlResult<float> =
    validateParams n p
    |> Result.bind (fun () ->
        if k < 0 || k > n then
            Ok 0.0
        elif p = 0.0 then
            Ok (if k = 0 then 1.0 else 0.0)
        elif p = 1.0 then
            Ok (if k = n then 1.0 else 0.0)
        else
            // Use log-space for numerical stability
            let logCoeff = logBinomial n k
            let logProb = float k * log p + float (n - k) * log (1.0 - p)
            Ok (exp (logCoeff + logProb)))

/// <summary>Cumulative distribution function for Binomial distribution.</summary>/// <param name="n">Number of trials (n >= 0).</param>/// <param name="p">Success probability (0 <= p <= 1).</param>/// <param name="k">Upper bound on number of successes.</param>/// <returns>CDF value P(X <= k).</returns>let cdf (n: int) (p: float) (k: int) : FowlResult<float> =
    validateParams n p
    |> Result.bind (fun () ->
        if k < 0 then
            Ok 0.0
        elif k >= n then
            Ok 1.0
        elif p = 0.0 then
            Ok 1.0  // P(X = 0) = 1
        elif p = 1.0 then
            Ok (if k >= n then 1.0 else 0.0)
        else
            // Use incomplete beta: P(X <= k) = I_{1-p}(n-k, k+1)
            let a = float (n - k)
            let b = float (k + 1)
            result {
                let! result = SpecialFunctions.incompleteBeta a b (1.0 - p)
                return result
            })

/// <summary>Percent point function (inverse CDF) for Binomial distribution.</summary>/// <param name="n">Number of trials (n >= 0).</param>/// <param name="p">Success probability (0 <= p <= 1).</param>/// <param name="prob">Probability (0 <= prob <= 1).</param>/// <returns>Smallest k such that P(X <= k) >= prob.</returns>let ppf (n: int) (p: float) (prob: float) : FowlResult<int> =
    validateParams n p
    |> Result.bind (fun () ->
        if prob < 0.0 || prob > 1.0 then
            Error.invalidArgument "prob must be in [0, 1]"
        elif prob = 0.0 then
            Ok 0
        elif prob = 1.0 then
            Ok n
        elif p = 0.0 then
            Ok 0
        elif p = 1.0 then
            Ok n
        else
            // Search for smallest k where CDF(k) >= prob
            let rec search k =
                if k > n then
                    Ok n
                else
                    match cdf n p k with
                    | Ok cdfVal ->
                        if cdfVal >= prob then Ok k else search (k + 1)
                    | Error e -> Error e
            
            // Start from mean for efficiency
            let mean = int (float n * p)
            search (max 0 (mean - 1)))

/// <summary>Random variate sampling from Binomial distribution.</summary>/// <param name="n">Number of trials (n >= 0).</param>/// <param name="p">Success probability (0 <= p <= 1).</param>/// <param name="shape">Shape of output array.</param>/// <returns>Array of random samples.</returns>/// <remarks>
/// Uses inverse transform sampling for small n, 
/// normal approximation for large n.
/// </remarks>let rvs (n: int) (p: float) (shape: Shape) : FowlResult<Ndarray<Float64, float>> =
    validateParams n p
    |> Result.bind (fun () ->
        let nSamples = Shape.numel shape
        let rng = Random()
        
        let result = Array.zeroCreate nSamples
        
        // For large n*p*(1-p), use normal approximation
        if float n * p * (1.0 - p) > 10.0 then
            // Normal approximation with continuity correction
            let mu = float n * p
            let sigma = sqrt (float n * p * (1.0 - p))
            
            for i = 0 to nSamples - 1 do
                let z = SpecialFunctions.randn rng
                let normalSample = mu + sigma * z
                // Round to nearest integer with bounds
                let binomialSample = int (round normalSample) |> max 0 |> min n
                result.[i] <- float binomialSample
        else
            // Direct Bernoulli sum
            for i = 0 to nSamples - 1 do
                let mutable successes = 0
                for _ = 1 to n do
                    if rng.NextDouble() < p then
                        successes <- successes + 1
                result.[i] <- float successes
        
        Ndarray.ofArray result shape)

/// <summary>Mean of Binomial distribution.</summary>/// <param name="n">Number of trials (n >= 0).</param>/// <param name="p">Success probability (0 <= p <= 1).</param>/// <returns>Mean = n * p.</returns>let mean (n: int) (p: float) : FowlResult<float> =
    validateParams n p
    |> Result.map (fun () -> float n * p)

/// <summary>Variance of Binomial distribution.</summary>/// <param name="n">Number of trials (n >= 0).</param>/// <param name="p">Success probability (0 <= p <= 1).</param>/// <returns>Variance = n * p * (1 - p).</returns>let var (n: int) (p: float) : FowlResult<float> =
    validateParams n p
    |> Result.map (fun () -> float n * p * (1.0 - p))

/// <summary>Standard deviation of Binomial distribution.</summary>/// <param name="n">Number of trials (n >= 0).</param>/// <param name="p">Success probability (0 <= p <= 1).</param>/// <returns>Standard deviation.</returns>let std (n: int) (p: float) : FowlResult<float> =
    var n p |> Result.map sqrt

/// <summary>Mode of Binomial distribution.</summary>/// <param name="n">Number of trials (n >= 0).</param>/// <param name="p">Success probability (0 <= p <= 1).</param>/// <returns>Mode = floor((n+1)p).</returns>let mode (n: int) (p: float) : FowlResult<int> =
    validateParams n p
    |> Result.map (fun () -> int (floor (float (n + 1) * p)))

/// <summary>Skewness of Binomial distribution.</summary>/// <param name="n">Number of trials (n >= 0).</param>/// <param name="p">Success probability (0 <= p <= 1).</param>/// <returns>Skewness = (1-2p) / √(np(1-p)).</returns>let skewness (n: int) (p: float) : FowlResult<float> =
    var n p
    |> Result.map (fun variance -> (1.0 - 2.0 * p) / sqrt variance)

/// <summary>Kurtosis (excess) of Binomial distribution.</summary>/// <param name="n">Number of trials (n >= 0).</param>/// <param name="p">Success probability (0 <= p <= 1).</param>/// <returns>Kurtosis = (1-6p(1-p)) / (np(1-p)).</returns>let kurtosis (n: int) (p: float) : FowlResult<float> =
    validateParams n p
    |> Result.map (fun () ->
        let q = 1.0 - p
        (1.0 - 6.0 * p * q) / (float n * p * q))