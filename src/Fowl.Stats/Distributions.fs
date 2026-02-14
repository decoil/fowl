module Fowl.Stats.Distributions

open System
open Fowl

/// Gaussian (Normal) distribution
module Gaussian =
    /// Probability density function
    let pdf (mu: float) (sigma: float) (x: float) : float =
        if sigma <= 0.0 then failwith "sigma must be positive"
        let coeff = 1.0 / (sigma * sqrt (2.0 * Math.PI))
        let expArg = -0.5 * ((x - mu) / sigma) ** 2.0
        coeff * exp expArg
    
    /// Cumulative distribution function (using error function approximation)
    let cdf (mu: float) (sigma: float) (x: float) : float =
        if sigma <= 0.0 then failwith "sigma must be positive"
        0.5 * (1.0 + SpecialFunctions.erf ((x - mu) / (sigma * sqrt 2.0)))
    
    /// Percent point function (inverse CDF)
    let ppf (mu: float) (sigma: float) (p: float) : float =
        if p < 0.0 || p > 1.0 then failwith "p must be in [0, 1]"
        mu + sigma * sqrt 2.0 * SpecialFunctions.erfcinv (2.0 * (1.0 - p))
    
    /// Random variate sampling
    let rvs (mu: float) (sigma: float) (shape: Shape) : Ndarray<Float64, float> =
        let rng = Random()
        let n = Shape.numel shape
        let data = Array.zeroCreate n
        // Box-Muller transform
        for i in 0 .. 2 .. n - 1 do
            let u1 = rng.NextDouble()
            let u2 = rng.NextDouble()
            let z1 = sqrt (-2.0 * log u1) * cos (2.0 * Math.PI * u2)
            let z2 = sqrt (-2.0 * log u1) * sin (2.0 * Math.PI * u2)
            data.[i] <- mu + sigma * z1
            if i + 1 < n then
                data.[i + 1] <- mu + sigma * z2
        Ndarray.ofArray data shape
    
    /// Log of PDF (more numerically stable for small values)
    let logpdf (mu: float) (sigma: float) (x: float) : float =
        if sigma <= 0.0 then failwith "sigma must be positive"
        let coeff = -log (sigma * sqrt (2.0 * Math.PI))
        let expArg = -0.5 * ((x - mu) / sigma) ** 2.0
        coeff + expArg

/// Uniform distribution
module Uniform =
    /// Probability density function
    let pdf (a: float) (b: float) (x: float) : float =
        if a >= b then failwith "a must be less than b"
        if x >= a && x <= b then 1.0 / (b - a) else 0.0
    
    /// Cumulative distribution function
    let cdf (a: float) (b: float) (x: float) : float =
        if a >= b then failwith "a must be less than b"
        if x < a then 0.0
        elif x > b then 1.0
        else (x - a) / (b - a)
    
    /// Percent point function (inverse CDF)
    let ppf (a: float) (b: float) (p: float) : float =
        if a >= b then failwith "a must be less than b"
        if p < 0.0 || p > 1.0 then failwith "p must be in [0, 1]"
        a + p * (b - a)
    
    /// Random variate sampling
    let rvs (a: float) (b: float) (shape: Shape) : Ndarray<Float64, float> =
        let rng = Random()
        let n = Shape.numel shape
        let data = Array.init n (fun _ -> a + rng.NextDouble() * (b - a))
        Ndarray.ofArray data shape
    
    /// Mean of distribution
    let mean (a: float) (b: float) : float =
        (a + b) / 2.0
    
    /// Variance of distribution
    let var (a: float) (b: float) : float =
        (b - a) ** 2.0 / 12.0

/// Exponential distribution
module Exponential =
    /// Probability density function
    let pdf (lambda: float) (x: float) : float =
        if lambda <= 0.0 then failwith "lambda must be positive"
        if x < 0.0 then 0.0
        else lambda * exp (-lambda * x)
    
    /// Cumulative distribution function
    let cdf (lambda: float) (x: float) : float =
        if lambda <= 0.0 then failwith "lambda must be positive"
        if x < 0.0 then 0.0
        else 1.0 - exp (-lambda * x)
    
    /// Percent point function (inverse CDF)
    let ppf (lambda: float) (p: float) : float =
        if lambda <= 0.0 then failwith "lambda must be positive"
        if p < 0.0 || p > 1.0 then failwith "p must be in [0, 1]"
        -log (1.0 - p) / lambda
    
    /// Random variate sampling (inverse transform)
    let rvs (lambda: float) (shape: Shape) : Ndarray<Float64, float> =
        let rng = Random()
        let n = Shape.numel shape
        let data = Array.init n (fun _ -> -log (1.0 - rng.NextDouble()) / lambda)
        Ndarray.ofArray data shape
    
    /// Mean (1/lambda)
    let mean (lambda: float) : float = 1.0 / lambda
    
    /// Variance (1/lambda^2)
    let var (lambda: float) : float = 1.0 / (lambda ** 2.0)

/// Gamma distribution
module Gamma =
    /// Probability density function (using shape k and scale theta)
    let pdf (shape: float) (scale: float) (x: float) : float =
        if shape <= 0.0 || scale <= 0.0 then failwith "shape and scale must be positive"
        if x < 0.0 then 0.0
        else
            let z = x / scale
            (z ** (shape - 1.0)) * exp (-z) / (scale * SpecialFunctions.gamma shape)
    
    /// Random variate sampling (Marsaglia and Tsang method)
    let rvs (k: float) (theta: float) (shape: Shape) : Ndarray<Float64, float> =
        if k < 1.0 then failwith "Gamma shape < 1 not implemented"
        let rng = Random()
        let n = Shape.numel shape
        let data = Array.zeroCreate n
        
        let d = k - 1.0 / 3.0
        let c = 1.0 / sqrt (9.0 * d)
        
        for i = 0 to n - 1 do
            let mutable found = false
            while not found do
                let z = SpecialFunctions.randn rng
                let u = rng.NextDouble()
                let v = (1.0 + c * z) ** 3.0
                if z > -1.0 / c && log u < 0.5 * z * z + d - d * v + d * log v then
                    data.[i] <- theta * d * v
                    found <- true
        Ndarray.ofArray data shape
    
    /// Mean (k * theta)
    let mean (k: float) (theta: float) : float = k * theta
    
    /// Variance (k * theta^2)
    let var (k: float) (theta: float) : float = k * (theta ** 2.0)

/// Beta distribution
module Beta =
    /// Probability density function
    let pdf (alpha: float) (beta: float) (x: float) : float =
        if alpha <= 0.0 || beta <= 0.0 then failwith "alpha and beta must be positive"
        if x < 0.0 || x > 1.0 then 0.0
        else
            let norm = SpecialFunctions.gamma (alpha + beta) / (SpecialFunctions.gamma alpha * SpecialFunctions.gamma beta)
            norm * (x ** (alpha - 1.0)) * ((1.0 - x) ** (beta - 1.0))
    
    /// Random variate sampling (using gamma)
    let rvs (alpha: float) (beta: float) (shape: Shape) : Ndarray<Float64, float> =
        let x = Gamma.rvs alpha 1.0 shape
        let y = Gamma.rvs beta 1.0 shape
        Ndarray.map2 (fun xi yi -> xi / (xi + yi)) x y
    
    /// Mean (alpha / (alpha + beta))
    let mean (alpha: float) (beta: float) : float = alpha / (alpha + beta)
    
    /// Variance
    let var (alpha: float) (beta: float) : float =
        (alpha * beta) / ((alpha + beta) ** 2.0 * (alpha + beta + 1.0))
