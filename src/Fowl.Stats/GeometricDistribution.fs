module Fowl.Stats.GeometricDistribution

open System
open Fowl
open Fowl.Core.Types

/// <summary>Geometric distribution with success probability p.</summary>/// <remarks>
/// Models the number of trials until the first success in a sequence of
/// independent Bernoulli trials. Two variants:
/// - Number of trials until first success (support: 1, 2, 3, ...)
/// - Number of failures before first success (support: 0, 1, 2, ...)
///
/// This implementation uses: P(X = k) = (1-p)^k * p for k = 0, 1, 2, ...
/// (number of failures before first success)
///
/// PMF: P(X = k) = (1-p)^k * p
/// CDF: P(X <= k) = 1 - (1-p)^(k+1)
/// </remarks>

/// <summary>Validate geometric distribution parameter.</summary>let validateParams (p: float) : FowlResult<unit> =
    if p <= 0.0 || p > 1.0 then
        Error.invalidArgument "p must be in (0, 1]"
    else
        Ok ()

/// <summary>Probability mass function for Geometric distribution.</summary>/// <param name="p">Success probability (0 < p <= 1).</param>/// <param name="k">Number of failures before first success (k >= 0).</param>/// <returns>PMF value P(X = k).</returns>/// <example>
/// <code>
/// let pmf = GeometricDistribution.pmf 0.3 2  // P(X = 2 failures)
/// </code>
/// </example>
let pmf (p: float) (k: int) : FowlResult<float> =
    validateParams p
    |> Result.bind (fun () ->
        if k < 0 then
            Ok 0.0
        elif p = 1.0 then
            Ok (if k = 0 then 1.0 else 0.0)
        else
            // P(X = k) = (1-p)^k * p
            let q = 1.0 - p
            Ok (q ** float k * p))

/// <summary>Cumulative distribution function for Geometric distribution.</summary>/// <param name="p">Success probability (0 < p <= 1).</param>/// <param name="k">Upper bound on failures (k >= 0).</param>/// <returns>CDF value P(X <= k).</returns>/// <remarks>
/// CDF(k) = 1 - (1-p)^(k+1)
/// </remarks>
let cdf (p: float) (k: int) : FowlResult<float> =
    validateParams p
    |> Result.bind (fun () ->
        if k < 0 then
            Ok 0.0
        elif p = 1.0 then
            Ok 1.0
        else
            // CDF(k) = 1 - (1-p)^(k+1)
            let q = 1.0 - p
            Ok (1.0 - q ** float (k + 1)))

/// <summary>Percent point function (inverse CDF) for Geometric distribution.</summary>/// <param name="p">Success probability (0 < p <= 1).</param>/// <param name="prob">Probability (0 <= prob <= 1).</param>/// <returns>Smallest k such that P(X <= k) >= prob.</returns>/// <remarks>
/// k = ceil(log(1-prob) / log(1-p)) - 1
/// </remarks>
let ppf (p: float) (prob: float) : FowlResult<int> =
    validateParams p
    |> Result.bind (fun () ->
        if prob < 0.0 || prob > 1.0 then
            Error.invalidArgument "prob must be in [0, 1]"
        elif prob = 0.0 then
            Ok 0
        elif prob = 1.0 then
            Ok System.Int32.MaxValue
        elif p = 1.0 then
            Ok 0
        else
            // k = ceil(log(1-prob) / log(1-p)) - 1
            let q = 1.0 - p
            let k = int (ceil (log (1.0 - prob) / log q)) - 1
            Ok (max 0 k))

/// <summary>Random variate sampling from Geometric distribution.</summary>/// <param name="p">Success probability (0 < p <= 1).</param>/// <param name="shape">Shape of output array.</param>/// <returns>Array of random samples.</returns>/// <remarks>
/// Uses inverse transform: k = floor(log(U) / log(1-p)) where U ~ Uniform(0,1)
/// Or: count trials until first success.
/// </remarks>
let rvs (p: float) (shape: Shape) : FowlResult<Ndarray<Float64, float>> =
    validateParams p
    |> Result.bind (fun () ->
        let n = Shape.numel shape
        let rng = Random()
        
        let result = Array.zeroCreate n
        
        if p > 0.1 then
            // Direct method: count trials
            for i = 0 to n - 1 do
                let mutable trials = 0
                while rng.NextDouble() > p do
                    trials <- trials + 1
                result.[i] <- float trials
        else
            // Inverse transform for small p (more efficient)
            // k = floor(log(U) / log(1-p))
            let logQ = log (1.0 - p)
            for i = 0 to n - 1 do
                let u = rng.NextDouble()
                let k = int (floor (log u / logQ))
                result.[i] <- float (max 0 k)
        
        Ndarray.ofArray result shape)

/// <summary>Mean of Geometric distribution.</summary>/// <param name="p">Success probability (0 < p <= 1).</param>/// <returns>Mean = (1-p)/p = q/p.</returns>let mean (p: float) : FowlResult<float> =
    validateParams p
    |> Result.map (fun () -> (1.0 - p) / p)

/// <summary>Variance of Geometric distribution.</summary>/// <param name="p">Success probability (0 < p <= 1).</param>/// <returns>Variance = (1-p)/p² = q/p².</returns>let var (p: float) : FowlResult<float> =
    validateParams p
    |> Result.map (fun () -> (1.0 - p) / (p * p))

/// <summary>Standard deviation of Geometric distribution.</summary>/// <param name="p">Success probability (0 < p <= 1).</param>/// <returns>Standard deviation.</returns>let std (p: float) : FowlResult<float> =
    var p |> Result.map sqrt

/// <summary>Mode of Geometric distribution.</summary>/// <param name="p">Success probability (0 < p <= 1).</param>/// <returns>Mode = 0 (always).</returns>let mode (p: float) : FowlResult<int> =
    validateParams p
    |> Result.map (fun () -> 0)

/// <summary>Median of Geometric distribution.</summary>/// <param name="p">Success probability (0 < p <= 1).</param>/// <returns>Median = ceil(-1/log₂(1-p)) - 1.</returns>let median (p: float) : FowlResult<int> =
    validateParams p
    |> Result.map (fun () ->
        let q = 1.0 - p
        int (ceil (-1.0 / log q / log 2.0)) - 1)

/// <summary>Skewness of Geometric distribution.</summary>/// <param name="p">Success probability (0 < p <= 1).</param>/// <returns>Skewness = (2-p)/√(1-p).</returns>let skewness (p: float) : FowlResult<float> =
    validateParams p
    |> Result.map (fun () -> (2.0 - p) / sqrt (1.0 - p))

/// <summary>Kurtosis (excess) of Geometric distribution.</summary>/// <param name="p">Success probability (0 < p <= 1).</param>/// <returns>Kurtosis = 6 + p²/(1-p).</returns>let kurtosis (p: float) : FowlResult<float> =
    validateParams p
    |> Result.map (fun () -> 6.0 + p * p / (1.0 - p))

/// <summary>Entropy of Geometric distribution.</summary>/// <param name="p">Success probability (0 < p <= 1).</param>/// <returns>Entropy in nats = (-(1-p)log(1-p) - plog(p))/p.</returns>let entropy (p: float) : FowlResult<float> =
    validateParams p
    |> Result.map (fun () ->
        if p = 1.0 then
            0.0
        else
            let q = 1.0 - p
            (-(q * log q) - (p * log p)) / p)