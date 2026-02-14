module Fowl.Stats.Distributions

open System
open Fowl
open Fowl.Core.Types
open Fowl.Stats.Random

/// Gaussian (Normal) distribution
module Gaussian =
    /// Validate parameters
    let validateParams (mu: float) (sigma: float) : FowlResult<unit> =
        if sigma <= 0.0 then
            Error.invalidArgument "sigma must be positive"
        else
            Ok ()
    
    /// Probability density function
    let pdf (mu: float) (sigma: float) (x: float) : FowlResult<float> =
        validateParams mu sigma
        |> Result.map (fun () ->
            let coeff = 1.0 / (sigma * sqrt (2.0 * Math.PI))
            let expArg = -0.5 * ((x - mu) / sigma) ** 2.0
            coeff * exp expArg)
    
    /// Cumulative distribution function
    let cdf (mu: float) (sigma: float) (x: float) : FowlResult<float> =
        validateParams mu sigma
        |> Result.map (fun () ->
            0.5 * (1.0 + SpecialFunctions.erf ((x - mu) / (sigma * sqrt 2.0))))
    
    /// Percent point function (inverse CDF)
    let ppf (mu: float) (sigma: float) (p: float) : FowlResult<float> =
        validateParams mu sigma
        |> Result.bind (fun () ->
            if p < 0.0 || p > 1.0 then
                Error.invalidArgument "p must be in [0, 1]"
            else
                Ok (mu + sigma * sqrt 2.0 * SpecialFunctions.erfcinv (2.0 * (1.0 - p))))
    
    /// Random variate sampling with explicit random state
    let rvsWithState (mu: float) (sigma: float) (shape: Shape) (state: RandomState) : FowlResult<Ndarray<Float64, float> * RandomState> =
        result {
            let! _ = validateParams mu sigma
            let n = Shape.numel shape
            let! data, newState = nextStandardNormals n state |> Ok
            let scaledData = data |> Array.map (fun z -> mu + sigma * z)
            let! arr = Ndarray.ofArray scaledData shape
            return arr, newState
        }
    
    /// Random variate sampling with default random state
    let rvs (mu: float) (sigma: float) (shape: Shape) : FowlResult<Ndarray<Float64, float>> =
        rvsWithState mu sigma shape (init()) |> Result.map fst

/// Uniform distribution
module Uniform =
    let validateParams (a: float) (b: float) : FowlResult<unit> =
        if a >= b then
            Error.invalidArgument "a must be less than b"
        else
            Ok ()
    
    let pdf (a: float) (b: float) (x: float) : FowlResult<float> =
        validateParams a b
        |> Result.map (fun () ->
            if x >= a && x <= b then 1.0 / (b - a) else 0.0)
    
    let cdf (a: float) (b: float) (x: float) : FowlResult<float> =
        validateParams a b
        |> Result.map (fun () ->
            if x < a then 0.0
            elif x > b then 1.0
            else (x - a) / (b - a))
    
    let ppf (a: float) (b: float) (p: float) : FowlResult<float> =
        validateParams a b
        |> Result.bind (fun () ->
            if p < 0.0 || p > 1.0 then
                Error.invalidArgument "p must be in [0, 1]"
            else
                Ok (a + p * (b - a)))
    
    let rvsWithState (a: float) (b: float) (shape: Shape) (state: RandomState) : FowlResult<Ndarray<Float64, float> * RandomState> =
        result {
            let! _ = validateParams a b
            let n = Shape.numel shape
            let! values, newState = nextFloats n state |> Ok
            let scaled = values |> Array.map (fun u -> a + u * (b - a))
            let! arr = Ndarray.ofArray scaled shape
            return arr, newState
        }

    let rvs (a: float) (b: float) (shape: Shape) : FowlResult<Ndarray<Float64, float>> =
        rvsWithState a b shape (init()) |> Result.map fst

/// Exponential distribution
module Exponential =
    let validateLambda (lambda: float) : FowlResult<unit> =
        if lambda <= 0.0 then
            Error.invalidArgument "lambda must be positive"
        else
            Ok ()
    
    let pdf (lambda: float) (x: float) : FowlResult<float> =
        validateLambda lambda
        |> Result.map (fun () ->
            if x < 0.0 then 0.0
            else lambda * exp (-lambda * x))
    
    let cdf (lambda: float) (x: float) : FowlResult<float> =
        validateLambda lambda
        |> Result.map (fun () ->
            if x < 0.0 then 0.0
            else 1.0 - exp (-lambda * x))
    
    let ppf (lambda: float) (p: float) : FowlResult<float> =
        validateLambda lambda
        |> Result.bind (fun () ->
            if p < 0.0 || p > 1.0 then
                Error.invalidArgument "p must be in [0, 1]"
            else
                Ok (-log (1.0 - p) / lambda))
    
    let rvsWithState (lambda: float) (shape: Shape) (state: RandomState) : FowlResult<Ndarray<Float64, float> * RandomState> =
        result {
            let! _ = validateLambda lambda
            let n = Shape.numel shape
            let! values, newState = nextFloats n state |> Ok
            let expSamples = values |> Array.map (fun u -> -log (1.0 - u) / lambda)
            let! arr = Ndarray.ofArray expSamples shape
            return arr, newState
        }

    let rvs (lambda: float) (shape: Shape) : FowlResult<Ndarray<Float64, float>> =
        rvsWithState lambda shape (init()) |> Result.map fst

    let mean (lambda: float) : FowlResult<float> =
        validateLambda lambda |> Result.map (fun () -> 1.0 / lambda)
    
    let var (lambda: float) : FowlResult<float> =
        validateLambda lambda |> Result.map (fun () -> 1.0 / (lambda ** 2.0))
