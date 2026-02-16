/// <summary>
/// Fowl.Stats - Statistical distributions and functions.
/// 
/// Based on Owl's stats module and numerical recipes.
/// All distributions provide: pdf, cdf, ppf, rvs, mean, var, std
/// </summary>
namespace Fowl

open System

// ============================================================================
// Special Mathematical Functions
// ============================================================================

/// <summary>Special mathematical functions for statistics.
/// </summary>
module Special =
    // ------------------------------------------------------------------------
    // Gamma Function
    // ------------------------------------------------------------------------
    
    /// <summary>Logarithm of the gamma function using Lanczos approximation.
    /// </summary>
    /// <param name="x">Input value (x > 0).</param>
    /// <returns>Log gamma of x.</returns>
    let gammaln (x: float) : float =
        // Lanczos coefficients
        let p = [|
            676.5203681218851
            -1259.1392167224028
            771.32342877765313
            -176.61502916214059
            12.507343278686905
            -0.13857109526572012
            9.9843695780195716e-6
            1.5056327351493116e-7
        |]
        let g = 7.0
        
        let x = x - 1.0
        let a = 
            (1.0, p) ||> Array.fold (fun acc p_i -> acc + p_i / (x + float (Array.findIndex ((=) p_i) p) + 1.0))
            |> fun a -> 0.99999999999980993 + (Array.fold2 (fun acc i p_i -> acc + p_i / (x + float i + 1.0)) 0.0 [|0..p.Length-1|] p)
        
        // Simpler implementation
        let rec loop i acc =
            if i >= p.Length then acc
            else loop (i + 1) (acc + p.[i] / (x + float i + 1.0))
        let a = 0.99999999999980993 + loop 0 0.0
        
        let t = x + g + 0.5
        0.5 * log (2.0 * Math.PI) + log a - t + (t - 0.5) * log t
    
    /// <summary>Gamma function.
    /// </summary>
    /// <param name="x">Input value.</param>
    /// <returns>Gamma of x.</returns>
    let rec gamma (x: float) : float =
        if x > 0.0 then exp (gammaln x)
        else Math.PI / (sin (Math.PI * x) * gamma (1.0 - x))
    
    // ------------------------------------------------------------------------
    // Error Function
    // ------------------------------------------------------------------------
    
    /// <summary>Error function using Abramowitz & Stegun approximation.
    /// </summary>
    /// <param name="x">Input value.</param>
    /// <returns>Erf of x.</returns>
    let erf (x: float) : float =
        // Abramowitz & Stegun formula 7.1.26
        let a1 =  0.254829592
        let a2 = -0.284496736
        let a3 =  1.421413741
        let a4 = -1.453152027
        let a5 =  1.061405429
        let p  =  0.3275911
        
        let sign = if x < 0.0 then -1.0 else 1.0
        let x = abs x
        
        let t = 1.0 / (1.0 + p * x)
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp (-x * x)
        
        sign * y
    
    /// <summary>Complementary error function.
    /// </summary>
    /// <param name="x">Input value.</param>
    /// <returns>Erfc of x.</returns>
    let erfc (x: float) : float = 1.0 - erf x
    
    /// <summary>Inverse complementary error function.
    /// </summary>
    /// <param name="p">Probability value.</param>
    /// <returns>Inverse erfc of p.</returns>
    let erfcinv (p: float) : float =
        // Rational approximation
        if p <= 0.0 then infinity
        elif p >= 2.0 then -infinity
        else
            let pp = if p < 1.0 then p else 2.0 - p
            let t = sqrt (-2.0 * log (pp / 2.0))
            
            let c0 = 2.515517
            let c1 = 0.802853
            let c2 = 0.010328
            let d1 = 1.432788
            let d2 = 0.189269
            let d3 = 0.001308
            
            let num = c0 + c1 * t + c2 * t * t
            let den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
            let x = t - num / den
            
            if p < 1.0 then x else -x

// ============================================================================
// Probability Distributions
// ============================================================================

/// <summary>Probability distributions.
/// </summary>
module Distributions =
    
    // --------------------------------------------------------------------
    // Normal Distribution
    // --------------------------------------------------------------------
    
    /// <summary>Normal (Gaussian) distribution.
    /// </summary>
    module Normal =
        /// <summary>Probability density function.
        /// </summary>
        /// <param name="mu">Mean.</param>
        /// <param name="sigma">Standard deviation.</param>
        /// <param name="x">Input value.</param>
        /// <returns>PDF at x.</returns>
        let pdf (mu: float) (sigma: float) (x: float) : float =
            let z = (x - mu) / sigma
            exp (-0.5 * z * z) / (sigma * sqrt (2.0 * Math.PI))
        
        /// <summary>Cumulative distribution function.
        /// </summary>
        /// <param name="mu">Mean.</param>
        /// <param name="sigma">Standard deviation.</param>
        /// <param name="x">Input value.</param>
        /// <returns>CDF at x.</returns>
        let cdf (mu: float) (sigma: float) (x: float) : float =
            0.5 * (1.0 + Special.erf ((x - mu) / (sigma * sqrt 2.0)))
        
        /// <summary>Percent point function (inverse CDF).
        /// </summary>
        /// <param name="mu">Mean.</param>
        /// <param name="sigma">Standard deviation.</param>
        /// <param name="p">Probability.</param>
        /// <returns>PPF at p.</returns>
        let ppf (mu: float) (sigma: float) (p: float) : float =
            if p <= 0.0 || p >= 1.0 then
                failwith "ppf: p must be in (0, 1)"
            let x = sqrt 2.0 * Special.erfcinv (2.0 * (1.0 - p))
            mu + sigma * x
        
        /// <summary>Random variate sampling (Box-Muller).
        /// </summary>
        /// <param name="mu">Mean.</param>
        /// <param name="sigma">Standard deviation.</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>Random sample.</returns>
        let rvs (mu: float) (sigma: float) (rng: Random) : float =
            let u1 = 1.0 - rng.NextDouble()
            let u2 = rng.NextDouble()
            let radius = sqrt (-2.0 * log u1)
            let theta = 2.0 * Math.PI * u2
            mu + sigma * radius * cos theta
        
        /// <summary>Mean of distribution.
        /// </summary>
        /// <param name="mu">Mean parameter.</param>
        /// <param name="_">Unused sigma.</param>
        /// <returns>Mean.</returns>
        let mean (mu: float) (_: float) : float = mu
        
        /// <summary>Variance of distribution.
        /// </summary>
        /// <param name="_">Unused mu.</param>
        /// <param name="sigma">Standard deviation.</param>
        /// <returns>Variance.</returns>
        let var (_: float) (sigma: float) : float = sigma * sigma
        
        /// <summary>Standard deviation of distribution.
        /// </summary>
        /// <param name="_">Unused mu.</param>
        /// <param name="sigma">Standard deviation parameter.</param>
        /// <returns>Standard deviation.</returns>
        let std (_: float) (sigma: float) : float = sigma
    
    // --------------------------------------------------------------------
    // Uniform Distribution
    // --------------------------------------------------------------------
    
    /// <summary>Uniform distribution.
    /// </summary>
    module Uniform =
        let pdf (a: float) (b: float) (x: float) : float =
            if x >= a && x <= b then 1.0 / (b - a) else 0.0
        
        let cdf (a: float) (b: float) (x: float) : float =
            if x < a then 0.0
            elif x > b then 1.0
            else (x - a) / (b - a)
        
        let ppf (a: float) (b: float) (p: float) : float =
            a + p * (b - a)
        
        let rvs (a: float) (b: float) (rng: Random) : float =
            a + rng.NextDouble() * (b - a)
        
        let mean (a: float) (b: float) : float = (a + b) / 2.0
        let var (a: float) (b: float) : float = (b - a) * (b - a) / 12.0
        let std (a: float) (b: float) : float = sqrt (var a b)
    
    // --------------------------------------------------------------------
    // Exponential Distribution
    // --------------------------------------------------------------------
    
    /// <summary>Exponential distribution.
    /// </summary>
    module Exponential =
        let pdf (lambda: float) (x: float) : float =
            if x < 0.0 then 0.0
            else lambda * exp (-lambda * x)
        
        let cdf (lambda: float) (x: float) : float =
            if x < 0.0 then 0.0
            else 1.0 - exp (-lambda * x)
        
        let ppf (lambda: float) (p: float) : float =
            -log (1.0 - p) / lambda
        
        let rvs (lambda: float) (rng: Random) : float =
            -log (1.0 - rng.NextDouble()) / lambda
        
        let mean (lambda: float) : float = 1.0 / lambda
        let var (lambda: float) : float = 1.0 / (lambda * lambda)
        let std (lambda: float) : float = 1.0 / lambda
    
    // --------------------------------------------------------------------
    // Gamma Distribution
    // --------------------------------------------------------------------
    
    /// <summary>Gamma distribution.
    /// </summary>
    module Gamma =
        let pdf (alpha: float) (beta: float) (x: float) : float =
            if x < 0.0 then 0.0
            elif x = 0.0 then
                if alpha < 1.0 then infinity
                elif alpha = 1.0 then beta
                else 0.0
            else
                exp (alpha * log beta + (alpha - 1.0) * log x - beta * x - Special.gammaln alpha)
        
        let mean (alpha: float) (beta: float) : float = alpha / beta
        let var (alpha: float) (beta: float) : float = alpha / (beta * beta)
        let std (alpha: float) (beta: float) : float = sqrt (var alpha beta)
    
    // --------------------------------------------------------------------
    // Beta Distribution
    // --------------------------------------------------------------------
    
    /// <summary>Beta distribution.
    /// </summary>
    module Beta =
        let pdf (alpha: float) (beta': float) (x: float) : float =
            if x < 0.0 || x > 1.0 then 0.0
            elif x = 0.0 || x = 1.0 then
                if alpha < 1.0 || beta' < 1.0 then infinity
                else 0.0
            else
                let logB = Special.gammaln alpha + Special.gammaln beta' - Special.gammaln (alpha + beta')
                exp ((alpha - 1.0) * log x + (beta' - 1.0) * log (1.0 - x) - logB)
        
        let mean (alpha: float) (beta': float) : float = alpha / (alpha + beta')
        let var (alpha: float) (beta': float) : float = 
            (alpha * beta') / ((alpha + beta') * (alpha + beta') * (alpha + beta' + 1.0))

// ============================================================================
// Summary Statistics
// ============================================================================

/// <summary>Summary statistics for datasets.
/// </summary>
module Summary =
    /// <summary>Calculate mean.
    /// </summary>
    /// <param name="data">Input data.</param>
    /// <returns>Mean value.</returns>
    /// <exception cref="System.Exception">Thrown when data is empty.</exception>
    let mean (data: float array) : float =
        if data.Length = 0 then failwith "mean: empty array"
        Array.sum data / float data.Length
    
    /// <summary>Calculate variance.
    /// </summary>
    /// <param name="ddof">Delta degrees of freedom (0 for population, 1 for sample).</param>
    /// <param name="data">Input data.</param>
    /// <returns>Variance.</returns>
    let var (ddof: int) (data: float array) : float =
        if data.Length <= ddof then failwith "var: insufficient data"
        let m = mean data
        let ss = data |> Array.sumBy (fun x -> (x - m) ** 2.0)
        ss / float (data.Length - ddof)
    
    /// <summary>Sample variance (ddof = 1).
    /// </summary>
    let varSample = var 1
    
    /// <summary>Population variance (ddof = 0).
    /// </summary>
    let varPopulation = var 0
    
    /// <summary>Standard deviation.
    /// </summary>
    /// <param name="ddof">Delta degrees of freedom.</param>
    /// <param name="data">Input data.</param>
    /// <returns>Standard deviation.</returns>
    let std (ddof: int) (data: float array) : float =
        sqrt (var ddof data)
    
    /// <summary>Sample standard deviation.
    /// </summary>
    let stdSample = std 1
    
    /// <summary>Population standard deviation.
    /// </summary>
    let stdPopulation = std 0
    
    /// <summary>Median.
    /// </summary>
    /// <param name="data">Input data.</param>
    /// <returns>Median value.</returns>
    let median (data: float array) : float =
        if data.Length = 0 then failwith "median: empty array"
        let sorted = Array.sort data
        let n = sorted.Length
        if n % 2 = 1 then
            sorted.[n / 2]
        else
            (sorted.[n / 2 - 1] + sorted.[n / 2]) / 2.0
    
    /// <summary>Quantile.
    /// </summary>
    /// <param name="q">Quantile to compute (0 to 1).</param>
    /// <param name="data">Input data.</param>
    /// <returns>Quantile value.</returns>
    let quantile (q: float) (data: float array) : float =
        if q < 0.0 || q > 1.0 then failwith "quantile: q must be in [0, 1]"
        let sorted = Array.sort data
        let n = sorted.Length
        let idx = q * float (n - 1)
        let i = int (floor idx)
        let frac = idx - float i
        if i >= n - 1 then sorted.[n - 1]
        else sorted.[i] * (1.0 - frac) + sorted.[i + 1] * frac

// ============================================================================
// Random Sampling
// ============================================================================

/// <summary>Random sampling utilities.
/// </summary>
module Sampling =
    /// <summary>Create RNG with seed.
    /// </summary>
    /// <param name="seed">Random seed.</param>
    /// <returns>Random number generator.</returns>
    let rngSeed (seed: int) = Random(seed)
    
    /// <summary>Create RNG with time-based seed.
    /// </summary>
    /// <returns>Random number generator.</returns>
    let rng () = Random()
    
    /// <summary>Sample from distribution.
    /// </summary>
    /// <param name="dist">Distribution function.</param>
    /// <param name="n">Number of samples.</param>
    /// <param name="rng">Random number generator.</param>
    /// <returns>Array of samples.</returns>
    let sample (dist: Random -> float) (n: int) (rng: Random) : float array =
        Array.init n (fun _ -> dist rng)
    
    /// <summary>Shuffle array in-place.
    /// </summary>
    /// <param name="rng">Random number generator.</param>
    /// <param name="arr">Array to shuffle.</param>
    let shuffle (rng: Random) (arr: 'T array) : unit =
        let n = arr.Length
        for i = n - 1 downto 1 do
            let j = rng.Next(i + 1)
            let tmp = arr.[i]
            arr.[i] <- arr.[j]
            arr.[j] <- tmp
    
    /// <summary>Sample without replacement.
    /// </summary>
    /// <param name="rng">Random number generator.</param>
    /// <param name="arr">Source array.</param>
    /// <param name="k">Number of samples.</param>
    /// <returns>Array of k samples.</returns>
    let sampleWithoutReplacement (rng: Random) (arr: 'T array) (k: int) : 'T array =
        if k > arr.Length then failwith "sampleWithoutReplacement: k > n"
        let copy = Array.copy arr
        shuffle rng copy
        Array.sub copy 0 k
