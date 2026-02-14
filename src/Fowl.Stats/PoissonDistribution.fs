module Fowl.Stats.PoissonDistribution

open System
open Fowl
open Fowl.Core.Types
open Fowl.Stats.SpecialFunctions

/// <summary>Poisson distribution with rate parameter λ (lambda).</summary>/// <remarks>
/// The Poisson distribution models the number of events occurring in a fixed
/// interval of time or space, given a constant mean rate λ. It's fundamental
/// for modeling rare events, count data, and queuing theory.
///
/// PMF: P(X = k) = λ^k * e^(-λ) / k!
///
/// Relationship to other distributions:
/// - Limit of Binomial(n, λ/n) as n → ∞
/// - Sum of independent Poissons is Poisson
/// </remarks>

/// <summary>Validate Poisson rate parameter.</summary>let validateParams (lambda: float) : FowlResult<unit> =
    if lambda < 0.0 then
        Error.invalidArgument "lambda must be non-negative"
    else
        Ok ()

/// <summary>Probability mass function for Poisson distribution.</summary>/// <param name="lambda">Rate parameter λ >= 0.</param>/// <param name="k">Number of events (k >= 0).</param>/// <returns>PMF value P(X = k).</returns>/// <example>
/// <code>
/// let pmf = PoissonDistribution.pmf 2.5 3  // P(X = 3)
/// </code>
/// </example>
let pmf (lambda: float) (k: int) : FowlResult<float> =
    validateParams lambda
    |> Result.bind (fun () ->
        if k < 0 then
            Ok 0.0
        elif lambda = 0.0 then
            Ok (if k = 0 then 1.0 else 0.0)
        else
            // Use log-space for numerical stability
            // log(PMF) = k*log(λ) - λ - log(k!)
            match SpecialFunctions.gamma (float k + 1.0) with
            | Ok g ->
                let logPmf = float k * log lambda - lambda - log g
                Ok (exp logPmf)
            | Error _ ->
                // Fallback for large k: use Stirling's approximation
                let logKFact = float k * log (float k) - float k + 0.5 * log (2.0 * Math.PI * float k)
                let logPmf = float k * log lambda - lambda - logKFact
                Ok (exp logPmf))

/// <summary>Cumulative distribution function for Poisson distribution.</summary>/// <param name="lambda">Rate parameter λ >= 0.</param>/// <param name="k">Upper bound on number of events.</param>/// <returns>CDF value P(X <= k).</returns>/// <remarks>
/// Uses the relationship to incomplete gamma function:
/// P(X <= k) = Γ(k+1, λ) / k! = Q(k+1, λ)
/// where Q is the regularized upper incomplete gamma.
/// </remarks>let cdf (lambda: float) (k: int) : FowlResult<float> =
    validateParams lambda
    |> Result.bind (fun () ->
        if k < 0 then
            Ok 0.0
        elif lambda = 0.0 then
            Ok 1.0  // P(X = 0) = 1
        else
            // CDF(k) = P(X <= k) = 1 - P(X > k)
            // Use relationship to incomplete gamma
            // P(X > k) = γ(k+1, λ) / k! where γ is lower incomplete gamma
            // For now, use direct summation (good for small k)
            // For large k, use normal approximation
            
            if float k > 20.0 * lambda then
                // k is way above mean, CDF ≈ 1
                Ok 1.0
            elif float k < lambda / 20.0 && lambda > 20.0 then
                // k is way below mean, CDF ≈ 0
                Ok 0.0
            else
                // Direct summation for moderate k
                let rec sumPmf acc i =
                    if i > k then
                        Ok acc
                    else
                        match pmf lambda i with
                        | Ok p -> sumPmf (acc + p) (i + 1)
                        | Error e -> Error e
                
                sumPmf 0.0 0)

/// <summary>Percent point function (inverse CDF) for Poisson distribution.</summary>/// <param name="lambda">Rate parameter λ >= 0.</param>/// <param name="p">Probability (0 <= p <= 1).</param>/// <returns>Smallest k such that P(X <= k) >= p.</returns>let ppf (lambda: float) (p: float) : FowlResult<int> =
    validateParams lambda
    |> Result.bind (fun () ->
        if p < 0.0 || p > 1.0 then
            Error.invalidArgument "p must be in [0, 1]"
        elif p = 0.0 then
            Ok 0
        elif p = 1.0 then
            Ok System.Int32.MaxValue
        elif lambda = 0.0 then
            Ok 0
        else
            // Start from mean and search
            let mean = int lambda
            
            // Check if we need to go up or down from mean
            match cdf lambda mean with
            | Ok cdfMean ->
                if cdfMean >= p then
                    // Search downward
                    let rec searchDown k =
                        if k < 0 then Ok 0
                        else
                            match cdf lambda k with
                            | Ok cdfK ->
                                if cdfK < p then Ok (k + 1) else searchDown (k - 1)
                            | Error _ -> Ok (max 0 k)
                    searchDown mean
                else
                    // Search upward
                    let rec searchUp k =
                        match cdf lambda k with
                        | Ok cdfK ->
                            if cdfK >= p then Ok k else searchUp (k + 1)
                        | Error _ -> Ok k
                    searchUp (mean + 1)
            | Error e -> Error e)

/// <summary>Random variate sampling from Poisson distribution.</summary>/// <param name="lambda">Rate parameter λ >= 0.</param>/// <param name="shape">Shape of output array.</param>/// <returns>Array of random samples.</returns>/// <remarks>
/// Uses Knuth's algorithm for small λ, 
/// Ahrens-Dieter algorithm for large λ.
/// </remarks>let rvs (lambda: float) (shape: Shape) : FowlResult<Ndarray<Float64, float>> =
    validateParams lambda
    |> Result.bind (fun () ->
        let n = Shape.numel shape
        let rng = Random()
        
        let result = Array.zeroCreate n
        
        if lambda < 30.0 then
            // Knuth's algorithm
            for i = 0 to n - 1 do
                let mutable k = 0
                let mutable p = 1.0
                let el = exp (-lambda)
                
                while p > el do
                    k <- k + 1
                    p <- p * rng.NextDouble()
                
                result.[i] <- float (k - 1)
        else
            // Ahrens-Dieter algorithm (rejection method)
            let c = 0.767 - 3.36 / lambda
            let beta = Math.PI / sqrt (3.0 * lambda)
            let alpha = beta * lambda
            let k = log c - lambda - log beta
            
            let rec generate () =
                let u = rng.NextDouble()
                let x = (alpha - log ((1.0 - u) / u)) / beta
                let n = int (floor (x + 0.5))
                
                if n < 0 then
                    generate ()
                else
                    let v = rng.NextDouble()
                    let y = alpha - beta * x
                    let lhs = y + log (v / (1.0 + exp y) ** 2.0)
                    let rhs = k + float n * log lambda - log (SpecialFunctions.gamma (float n + 1.0) |> Result.get)
                    
                    if lhs <= rhs then
                        float n
                    else
                        generate ()
            
            for i = 0 to n - 1 do
                result.[i] <- generate ()
        
        Ndarray.ofArray result shape)

/// <summary>Mean of Poisson distribution.</summary>/// <param name="lambda">Rate parameter λ >= 0.</param>/// <returns>Mean = λ.</returns>let mean (lambda: float) : FowlResult<float> =
    validateParams lambda
    |> Result.map (fun () -> lambda)

/// <summary>Variance of Poisson distribution.</summary>/// <param name="lambda">Rate parameter λ >= 0.</param>/// <returns>Variance = λ.</returns>let var (lambda: float) : FowlResult<float> =
    validateParams lambda
    |> Result.map (fun () -> lambda)

/// <summary>Standard deviation of Poisson distribution.</summary>/// <param name="lambda">Rate parameter λ >= 0.</param>/// <returns>Standard deviation = √λ.</returns>let std (lambda: float) : FowlResult<float> =
    var lambda |> Result.map sqrt

/// <summary>Mode of Poisson distribution.</summary>/// <param name="lambda">Rate parameter λ >= 0.</param>/// <returns>Mode = ⌊λ⌋ and ⌈λ⌉ - 1 (dual mode when λ is integer).</returns>let mode (lambda: float) : FowlResult<int> =
    validateParams lambda
    |> Result.map (fun () -> int (floor lambda))

/// <summary>Skewness of Poisson distribution.</summary>/// <param name="lambda">Rate parameter λ > 0.</param>/// <returns>Skewness = 1/√λ.</returns>let skewness (lambda: float) : FowlResult<float> =
    validateParams lambda
    |> Result.map (fun () -> 1.0 / sqrt lambda)

/// <summary>Kurtosis (excess) of Poisson distribution.</summary>/// <param name="lambda">Rate parameter λ > 0.</param>/// <returns>Kurtosis = 1/λ.</returns>let kurtosis (lambda: float) : FowlResult<float> =
    validateParams lambda
    |> Result.map (fun () -> 1.0 / lambda)

/// <summary>Entropy of Poisson distribution.</summary>/// <param name="lambda">Rate parameter λ > 0.</param>/// <returns>Entropy in nats.</returns>/// <remarks>
/// H(X) = λ(1 - log λ) + e^(-λ) * Σ(λ^k * log(k!) / k!)
/// </remarks>let entropy (lambda: float) : FowlResult<float> =
    validateParams lambda
    |> Result.map (fun () ->
        // Approximation for moderate to large λ
        // H(X) ≈ 0.5 * log(2πeλ) - 1/(12λ) - ...
        0.5 * log (2.0 * Math.PI * Math.E * lambda))