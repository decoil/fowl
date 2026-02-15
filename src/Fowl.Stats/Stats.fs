/// <summary>
/// Fowl.Stats - Statistical distributions and functions.
/// 
/// Based on:
/// - Owl's stats module
/// - Architecture of Advanced Numerical Analysis Systems
/// - Statistical Computing in Numerical Recipes
/// </summary>
namespace Fowl

open System

// ============================================================================
// Special Functions
// ============================================================================

/// <summary>Special mathematical functions for statistics.
/// </summary>
module Special =
    /// <summary>Logarithm of the gamma function.
    /// Uses Lanczos approximation.
    /// </summary>
    let gammaln (x: float) : float =
        // Lanczos coefficients
        let p = [|676.5203681218851; -1259.1392167224028; 771.32342877765313;
                  -176.61502916214059; 12.507343278686905; -0.13857109526572012;
                  9.9843695780195716e-6; 1.5056327351493116e-7|]
        let g = 7.0
        
        let x = x - 1.0
        let a = 0.99999999999980993
        let a = p |> Array.mapi (fun i p -> a + p / (x + float i + 1.0)) |> Array.last
        
        let t = x + g + 0.5
        0.5 * log (2.0 * Math.PI) + log a - t + (t - 0.5) * log t
    
    /// <summary>Gamma function.
    /// </summary>
    let gamma (x: float) : float =
        if x > 0.0 then
            exp (gammaln x)
        else
            Math.PI / (sin (Math.PI * x) * gamma (1.0 - x))
    
    /// <summary>Incomplete gamma function P(a,x).
    /// Regularized lower incomplete gamma.
    /// </summary>
    let gammainc (a: float) (x: float) : float =
        // Series representation for small x
        let rec series (gln: float) (sum: float) (del: float) (ap: float) (n: int) : float =
            if n > 10000 then
                failwith "gammainc: series did not converge"
            else
                let ap = ap + 1.0
                let del = del * x / ap
                let sum = sum + del
                if abs del < abs sum * 1e-7 then
                    sum * exp (-x + a * log x - gln)
                else
                    series gln sum del ap (n + 1)
        
        if x < 0.0 || a <= 0.0 then
            failwith "gammainc: invalid arguments"
        elif x < a + 1.0 then
            // Use series representation
            let gln = gammaln a
            series gln (1.0 / a) (1.0 / a) a 0
        else
            // Use continued fraction
            1.0 - gammaincc a x
    
    /// <summary>Complementary incomplete gamma function Q(a,x).
    /// </summary>
    and gammaincc (a: float) (x: float) : float =
        // Continued fraction representation
        let rec cfrac (gl: float) (a: float) (x: float) : float =
            let b = x + 1.0 - a
            let c = 1.0 / 1.0e-30
            let d = 1.0 / b
            let h = d
            
            let rec loop i d c h =
                if i > 100 then h
                else
                    let an = -float i * (float i - a)
                    let b = b + 2.0
                    let d = an * d + b
                    let d = if abs d < 1.0e-30 then 1.0e-30 else d
                    let c = b + an / c
                    let c = if abs c < 1.0e-30 then 1.0e-30 else c
                    let d = 1.0 / d
                    let del = d * c
                    let h = h * del
                    if abs (del - 1.0) < 1.0e-7 then
                        h
                    else
                        loop (i + 1) d c h
            
            loop 1 d c h
        
        let gln = gammaln a
        let ans = cfrac gln a x
        exp (-x + a * log x - gln) * ans
    
    /// <summary>Error function erf(x).
    /// </summary>
    let erf (x: float) : float =
        if x >= 0.0 then gammainc 0.5 (x * x)
        else -gammainc 0.5 (x * x)
    
    /// <summary>Complementary error function erfc(x).
    /// </summary>
    let erfc (x: float) : float = 1.0 - erf x
    
    /// <summary>Inverse complementary error function erfc^(-1)(x).
    /// Uses rational approximation.
    /// </summary>
    let erfcinv (p: float) : float =
        // Rational approximation for inverse erfc
        // Based on Abramowitz & Stegun formula 26.2.23
        if p <= 0.0 then infinity
        elif p >= 2.0 then -infinity
        else
            let pp = if p < 1.0 then p else 2.0 - p
            let t = sqrt (-2.0 * log (pp / 2.0))
            
            // Coefficients for rational approximation
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
    
    /// <summary>Beta function B(a,b).
    /// </summary>
    let beta (a: float) (b: float) : float =
        exp (gammaln a + gammaln b - gammaln (a + b))
    
    /// <summary>Log beta function.
    /// </summary>
    let betaln (a: float) (b: float) : float =
        gammaln a + gammaln b - gammaln (a + b)
    
    /// <summary>Incomplete beta function Ix(a,b).
    /// </summary>
    let betainc (a: float) (b: float) (x: float) : float =
        if x < 0.0 || x > 1.0 then
            failwith "betainc: x must be in [0, 1]"
        elif x = 0.0 || x = 1.0 then
            x
        else
            let bt = exp (-betaln a b + a * log x + b * log (1.0 - x))
            if x < (a + 1.0) / (a + b + 2.0) then
                bt * betacf a b x / a
            else
                1.0 - bt * betacf b a (1.0 - x) / b
    
    /// <summary>Continued fraction for incomplete beta.
    /// </summary>
    and betacf (a: float) (b: float) (x: float) : float =
        let rec loop m m2 d c h =
            if m > 10000 then h
            else
                let aa = float m * (b - float m) * x / ((a + m2 - 1.0) * (a + m2))
                let d = 1.0 + aa * d
                let d = if abs d < 1.0e-30 then 1.0e-30 else d
                let c = 1.0 + aa / c
                let c = if abs c < 1.0e-30 then 1.0e-30 else c
                let d = 1.0 / d
                let h = h * d * c
                let aa = -(a + float m) * (a + b + float m) * x / ((a + m2) * (a + m2 + 1.0))
                let d = 1.0 + aa * d
                let d = if abs d < 1.0e-30 then 1.0e-30 else d
                let c = 1.0 + aa / c
                let c = if abs c < 1.0e-30 then 1.0e-30 else c
                let d = 1.0 / d
                let del = d * c
                let h = h * del
                if abs (del - 1.0) < 1.0e-7 then
                    h
                else
                    loop (m + 1) (m2 + 2.0) d c h
        
        let m2 = 0.0
        let aa = 1.0
        let d = 1.0 - (a + b) * x / (a + 1.0)
        let d = if abs d < 1.0e-30 then 1.0e-30 else d
        let c = 1.0
        let h = aa / d
        loop 1 2.0 d c h

// ============================================================================
// Probability Distributions
// ============================================================================

/// <summary>Common probability distributions.
/// </summary>
module Distributions =
    
    /// <summary>Normal (Gaussian) distribution.
    /// </summary>
    module Normal =
        /// <summary>Probability density function.
        /// </summary>
        let pdf (mu: float) (sigma: float) (x: float) : float =
            let z = (x - mu) / sigma
            exp (-0.5 * z * z) / (sigma * sqrt (2.0 * Math.PI))
        
        /// <summary>Cumulative distribution function.
        /// </summary>
        let cdf (mu: float) (sigma: float) (x: float) : float =
            0.5 * (1.0 + Special.erf ((x - mu) / (sigma * sqrt 2.0)))
        
        /// <summary>Percent point function (inverse CDF).
        /// </summary>
        let ppf (mu: float) (sigma: float) (p: float) : float =
            if p <= 0.0 || p >= 1.0 then
                failwith "ppf: p must be in (0, 1)"
            // Approximation using inverse error function
            let x = sqrt 2.0 * Special.erfcinv (2.0 * (1.0 - p))
            mu + sigma * x
        
        /// <summary>Random sample generation (Box-Muller).
        /// </summary>
        let rvs (mu: float) (sigma: float) (rng: Random) : float =
            let u1 = 1.0 - rng.NextDouble()
            let u2 = rng.NextDouble()
            let radius = sqrt (-2.0 * log u1)
            let theta = 2.0 * Math.PI * u2
            mu + sigma * radius * cos theta
    
    /// <summary>Uniform distribution.
    /// </summary>
    module Uniform =
        /// <summary>PDF.
        /// </summary>
        let pdf (a: float) (b: float) (x: float) : float =
            if x >= a && x <= b then 1.0 / (b - a) else 0.0
        
        /// <summary>CDF.
        /// </summary>
        let cdf (a: float) (b: float) (x: float) : float =
            if x < a then 0.0
            elif x > b then 1.0
            else (x - a) / (b - a)
        
        /// <summary>PPF.
        /// </summary>
        let ppf (a: float) (b: float) (p: float) : float =
            a + p * (b - a)
        
        /// <summary>RVS.
        /// </summary>
        let rvs (a: float) (b: float) (rng: Random) : float =
            a + rng.NextDouble() * (b - a)
    
    /// <summary>Exponential distribution.
    /// </summary>
    module Exponential =
        /// <summary>PDF.
        /// </summary>
        let pdf (lambda: float) (x: float) : float =
            if x < 0.0 then 0.0
            else lambda * exp (-lambda * x)
        
        /// <summary>CDF.
        /// </summary>
        let cdf (lambda: float) (x: float) : float =
            if x < 0.0 then 0.0
            else 1.0 - exp (-lambda * x)
        
        /// <summary>PPF.
        /// </summary>
        let ppf (lambda: float) (p: float) : float =
            -log (1.0 - p) / lambda
        
        /// <summary>RVS (inverse transform sampling).
        /// </summary>
        let rvs (lambda: float) (rng: Random) : float =
            -log (1.0 - rng.NextDouble()) / lambda
    
    /// <summary>Gamma distribution.
    /// </summary>
    module Gamma =
        /// <summary>PDF.
        /// </summary>
        let pdf (alpha: float) (beta: float) (x: float) : float =
            if x < 0.0 then 0.0
            elif x = 0.0 then
                if alpha < 1.0 then infinity
                elif alpha = 1.0 then beta
                else 0.0
            else
                exp (alpha * log beta + (alpha - 1.0) * log x - beta * x - Special.gammaln alpha)
        
        /// <summary>CDF using incomplete gamma.
        /// </summary>
        let cdf (alpha: float) (beta: float) (x: float) : float =
            if x <= 0.0 then 0.0
            else Special.gammainc alpha (beta * x)
        
        /// <summary>RVS using Marsaglia-Tsang method.
        /// </summary>
        let rvs (alpha: float) (beta: float) (rng: Random) : float =
            if alpha >= 1.0 then
                let d = alpha - 1.0 / 3.0
                let c = 1.0 / sqrt (9.0 * d)
                
                let rec loop () =
                    let x = Normal.rvs 0.0 1.0 rng
                    let v = (1.0 + c * x) ** 3.0
                    if v <= 0.0 then
                        loop ()
                    else
                        let u = rng.NextDouble()
                        if log u < 0.5 * x * x + d * (1.0 - v + log v) then
                            d * v / beta
                        else
                            loop ()
                
                loop ()
            else
                // Use alpha + 1 and rejection
                let x = rvs (alpha + 1.0) beta rng
                let u = rng.NextDouble()
                x * (u ** (1.0 / alpha))
    
    /// <summary>Beta distribution.
    /// </summary>
    module Beta =
        /// <summary>PDF.
        /// </summary>
        let pdf (alpha: float) (beta': float) (x: float) : float =
            if x < 0.0 || x > 1.0 then 0.0
            elif x = 0.0 || x = 1.0 then
                if alpha < 1.0 || beta' < 1.0 then infinity
                else 0.0
            else
                exp ((alpha - 1.0) * log x + (beta' - 1.0) * log (1.0 - x) - Special.betaln alpha beta')
        
        /// <summary>CDF using incomplete beta.
        /// </summary>
        let cdf (alpha: float) (beta': float) (x: float) : float =
            if x <= 0.0 then 0.0
            elif x >= 1.0 then 1.0
            else Special.betainc alpha beta' x
        
        /// <summary>RVS using gamma relationship.
        /// </summary>
        let rvs (alpha: float) (beta': float) (rng: Random) : float =
            let x = Gamma.rvs alpha 1.0 rng
            let y = Gamma.rvs beta' 1.0 rng
            x / (x + y)
    
    /// <summary>Student's t-distribution.
    /// </summary>
    module StudentT =
        /// <summary>PDF.
        /// </summary>
        let pdf (df: float) (x: float) : float =
            let nu = df
            let logC = Special.gammaln ((nu + 1.0) / 2.0) - Special.gammaln (nu / 2.0) - 0.5 * log (Math.PI * nu)
            let logF = -(nu + 1.0) / 2.0 * log (1.0 + x * x / nu)
            exp (logC + logF)
        
        /// <summary>CDF using incomplete beta.
        /// </summary>
        let cdf (df: float) (x: float) : float =
            let x2 = x * x
            if x > 0.0 then
                1.0 - 0.5 * Special.betainc (df / 2.0) 0.5 (df / (df + x2))
            else
                0.5 * Special.betainc (df / 2.0) 0.5 (df / (df + x2))
        
        /// <summary>RVS using normal/chi-squared relationship.
        /// </summary>
        let rvs (df: float) (rng: Random) : float =
            let z = Normal.rvs 0.0 1.0 rng
            let v = 2.0 * Gamma.rvs (df / 2.0) 1.0 rng
            z / sqrt (v / df)
    
    /// <summary>Chi-squared distribution.
    /// </summary>
    module ChiSquare =
        /// <summary>PDF.
        /// </summary>
        let pdf (df: float) (x: float) : float =
            if x < 0.0 then 0.0
            else
                let k2 = df / 2.0
                exp (-x / 2.0 + (k2 - 1.0) * log x - k2 * log 2.0 - Special.gammaln k2)
        
        /// <summary>CDF using incomplete gamma.
        /// </summary>
        let cdf (df: float) (x: float) : float =
            if x <= 0.0 then 0.0
            else Special.gammainc (df / 2.0) (x / 2.0)
        
        /// <summary>RVS using gamma distribution.
        /// </summary>
        let rvs (df: float) (rng: Random) : float =
            2.0 * Gamma.rvs (df / 2.0) 1.0 rng
    
    /// <summary>F-distribution.
    /// </summary>
    module FDistribution =
        /// <summary>PDF.
        /// </summary>
        let pdf (d1: float) (d2: float) (x: float) : float =
            if x < 0.0 then 0.0
            else
                let logC = Special.gammaln ((d1 + d2) / 2.0) - Special.gammaln (d1 / 2.0) - Special.gammaln (d2 / 2.0)
                let logF = d1 / 2.0 * log d1 + d2 / 2.0 * log d2 + (d1 / 2.0 - 1.0) * log x
                let logDen = (d1 + d2) / 2.0 * log (d2 + d1 * x)
                exp (logC + logF - logDen)
        
        /// <summary>CDF using incomplete beta.
        /// </summary>
        let cdf (d1: float) (d2: float) (x: float) : float =
            if x <= 0.0 then 0.0
            else Special.betainc (d1 / 2.0) (d2 / 2.0) (d1 * x / (d2 + d1 * x))
        
        /// <summary>RVS using chi-squared relationship.
        /// </summary>
        let rvs (d1: float) (d2: float) (rng: Random) : float =
            let chi1 = ChiSquare.rvs d1 rng
            let chi2 = ChiSquare.rvs d2 rng
            (chi1 / d1) / (chi2 / d2)

// ============================================================================
// Summary Statistics
// ============================================================================

/// <summary>Summary statistics for datasets.
/// </summary>
module Summary =
    /// <summary>Calculate mean.
    /// </summary>
    let mean (data: float array) : float =
        if data.Length = 0 then failwith "mean: empty array"
        Array.sum data / float data.Length
    
    /// <summary>Calculate variance (sample or population).
    /// </summary>
    let var (ddof: int) (data: float array) : float =
        if data.Length <= ddof then failwith "var: insufficient data"
        let m = mean data
        let ss = data |> Array.sumBy (fun x -> (x - m) ** 2.0)
        ss / float (data.Length - ddof)
    
    /// <summary>Sample variance.
    /// </summary>
    let varSample = var 1
    
    /// <summary>Population variance.
    /// </summary>
    let varPopulation = var 0
    
    /// <summary>Standard deviation.
    /// </summary>
    let std (ddof: int) (data: float array) : float =
        sqrt (var ddof data)
    
    /// <summary>Sample standard deviation.
    /// </summary>
    let stdSample = std 1
    
    /// <summary>Median.
    /// </summary>
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
    let quantile (q: float) (data: float array) : float =
        if q < 0.0 || q > 1.0 then failwith "quantile: q must be in [0, 1]"
        let sorted = Array.sort data
        let n = sorted.Length
        let idx = q * float (n - 1)
        let i = int (floor idx)
        let frac = idx - float i
        if i >= n - 1 then sorted.[n - 1]
        else sorted.[i] * (1.0 - frac) + sorted.[i + 1] * frac
    
    /// <summary>Skewness.
    /// </summary>
    let skew (data: float array) : float =
        let n = float data.Length
        let m = mean data
        let s = stdSample data
        let m3 = data |> Array.sumBy (fun x -> ((x - m) / s) ** 3.0)
        m3 * n / (n - 1.0) / (n - 2.0)
    
    /// <summary>Kurtosis.
    /// </summary>
    let kurt (data: float array) : float =
        let n = float data.Length
        let m = mean data
        let s = stdSample data
        let m4 = data |> Array.sumBy (fun x -> ((x - m) / s) ** 4.0)
        m4 * n * (n + 1.0) / (n - 1.0) / (n - 2.0) / (n - 3.0) - 3.0 * (n - 1.0) ** 2.0 / (n - 2.0) / (n - 3.0)

// ============================================================================
// Hypothesis Tests
// ============================================================================

/// <summary>Hypothesis testing functions.
/// </summary>
module HypothesisTests =
    /// <summary>One-sample t-test.
    /// </summary>
    let ttest1samp (data: float array) (popmean: float) : float * float =
        let n = float data.Length
        let m = Summary.mean data
        let se = Summary.stdSample data / sqrt n
        let t = (m - popmean) / se
        let df = n - 1.0
        let p = 2.0 * (1.0 - abs (Distributions.StudentT.cdf df t))
        t, p
    
    /// <summary>Two-sample t-test (equal variances).
    /// </summary>
    let ttestInd (data1: float array) (data2: float array) : float * float =
        let n1 = float data1.Length
        let n2 = float data2.Length
        let v1 = Summary.varSample data1
        let v2 = Summary.varSample data2
        let m1 = Summary.mean data1
        let m2 = Summary.mean data2
        
        // Pooled standard error
        let se = sqrt ((v1 * (n1 - 1.0) + v2 * (n2 - 1.0)) / (n1 + n2 - 2.0) * (1.0 / n1 + 1.0 / n2))
        let t = (m1 - m2) / se
        let df = n1 + n2 - 2.0
        let p = 2.0 * (1.0 - abs (Distributions.StudentT.cdf df t))
        t, p
    
    /// <summary>Chi-square test for variance.
    /// </summary>
    let chisquareVar (data: float array) (popvar: float) : float * float =
        let n = float data.Length
        let s2 = Summary.varSample data
        let chi2 = (n - 1.0) * s2 / popvar
        let p = 1.0 - Distributions.ChiSquare.cdf (n - 1.0) chi2
        chi2, p
    
    /// <summary>Kolmogorov-Smirnov test (against normal).
    /// </summary>
    let kstest (data: float array) : float * float =
        let n = float data.Length
        let mu = Summary.mean data
        let sigma = Summary.stdSample data
        let sorted = Array.sort data
        
        // Calculate empirical CDF and theoretical CDF
        let d = sorted |> Array.mapi (fun i x ->
            let empirical = float (i + 1) / n
            let theoretical = Distributions.Normal.cdf mu sigma x
            max (abs (empirical - theoretical)) (abs (theoretical - float i / n))
        ) |> Array.max
        
        // Approximate p-value
        let en = sqrt n
        let p = min 1.0 (2.0 * exp (-2.0 * en * en * d * d))
        d, p

// ============================================================================
// Random Sampling
// ============================================================================

/// <summary>Random sampling utilities.
/// </summary>
module Sampling =
    /// <summary>Random number generator with seed.
    /// </summary>
    let rngSeed (seed: int) = Random(seed)
    
    /// <summary>Random number generator with time-based seed.
    /// </summary>
    let rng () = Random()
    
    /// <summary>Sample from distribution.
    /// </summary>
    let sample (dist: Random -> float) (n: int) (rng: Random) : float array =
        Array.init n (fun _ -> dist rng)
    
    /// <summary>Shuffle array in-place.
    /// </summary>
    let shuffle (rng: Random) (arr: 'T array) : unit =
        let n = arr.Length
        for i = n - 1 downto 1 do
            let j = rng.Next(i + 1)
            let tmp = arr.[i]
            arr.[i] <- arr.[j]
            arr.[j] <- tmp
    
    /// <summary>Sample without replacement.
    /// </summary>
    let sampleWithoutReplacement (rng: Random) (arr: 'T array) (k: int) : 'T array =
        if k > arr.Length then failwith "sampleWithoutReplacement: k > n"
        let copy = Array.copy arr
        shuffle rng copy
        Array.sub copy 0 k
