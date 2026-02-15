/// <summary>Fowl.Stats - Statistical distributions and functions.
/// </summary>
namespace Fowl

open System

/// <summary>Error function.
/// </summary>
module Special =
    let erf (x: float) : float =
        // Abramowitz & Stegun approximation
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

/// <summary>Probability distributions.
/// </summary>
module Distributions =
    
    /// <summary>Normal (Gaussian) distribution.
    /// </summary>
    module Normal =
        let pdf (mu: float) (sigma: float) (x: float) : float =
            let z = (x - mu) / sigma
            exp (-0.5 * z * z) / (sigma * sqrt (2.0 * Math.PI))
        
        let cdf (mu: float) (sigma: float) (x: float) : float =
            0.5 * (1.0 + Special.erf ((x - mu) / (sigma * sqrt 2.0)))
        
        let rvs (mu: float) (sigma: float) (rng: Random) : float =
            let u1 = 1.0 - rng.NextDouble()
            let u2 = rng.NextDouble()
            let radius = sqrt (-2.0 * log u1)
            let theta = 2.0 * Math.PI * u2
            mu + sigma * radius * cos theta
    
    /// <summary>Uniform distribution.
    /// </summary>
    module Uniform =
        let pdf (a: float) (b: float) (x: float) : float =
            if x >= a && x <= b then 1.0 / (b - a) else 0.0
        
        let cdf (a: float) (b: float) (x: float) : float =
            if x < a then 0.0
            elif x > b then 1.0
            else (x - a) / (b - a)
        
        let rvs (a: float) (b: float) (rng: Random) : float =
            a + rng.NextDouble() * (b - a)
    
    /// <summary>Exponential distribution.
    /// </summary>
    module Exponential =
        let pdf (lambda: float) (x: float) : float =
            if x < 0.0 then 0.0
            else lambda * exp (-lambda * x)
        
        let cdf (lambda: float) (x: float) : float =
            if x < 0.0 then 0.0
            else 1.0 - exp (-lambda * x)
        
        let rvs (lambda: float) (rng: Random) : float =
            -log (1.0 - rng.NextDouble()) / lambda

/// <summary>Error function.
/// </summary>
and Special =
    static member erf (x: float) : float =
        // Abramowitz & Stegun approximation
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

/// <summary>Summary statistics.
/// </summary>
module Summary =
    let mean (data: float array) : float =
        if data.Length = 0 then failwith "mean: empty array"
        Array.sum data / float data.Length
    
    let varSample (data: float array) : float =
        if data.Length < 2 then failwith "var: insufficient data"
        let m = mean data
        let ss = data |> Array.sumBy (fun x -> (x - m) ** 2.0)
        ss / float (data.Length - 1)
    
    let stdSample (data: float array) : float =
        sqrt (varSample data)
    
    let median (data: float array) : float =
        if data.Length = 0 then failwith "median: empty array"
        let sorted = Array.sort data
        let n = sorted.Length
        if n % 2 = 1 then
            sorted.[n / 2]
        else
            (sorted.[n / 2 - 1] + sorted.[n / 2]) / 2.0

/// <summary>Random sampling.
/// </summary>
module Sampling =
    let rngSeed (seed: int) = Random(seed)
    let rng () = Random()
    
    let sample (dist: Random -> float) (n: int) (rng: Random) : float array =
        Array.init n (fun _ -> dist rng)
    
    let shuffle (rng: Random) (arr: 'T array) : unit =
        let n = arr.Length
        for i = n - 1 downto 1 do
            let j = rng.Next(i + 1)
            let tmp = arr.[i]
            arr.[i] <- arr.[j]
            arr.[j] <- tmp
    
    let sampleWithoutReplacement (rng: Random) (arr: 'T array) (k: int) : 'T array =
        if k > arr.Length then failwith "sampleWithoutReplacement: k > n"
        let copy = Array.copy arr
        shuffle rng copy
        Array.sub copy 0 k
