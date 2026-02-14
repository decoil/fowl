module Fowl.Stats.Descriptive

open Fowl

/// Calculate mean of array
let mean (arr: Ndarray<'K, float>) : float =
    let sum = arr |> Ndarray.toArray |> Array.sum
    sum / float (Ndarray.numel arr)

/// Calculate mean along specified axis
let meanAxis (axis: int) (arr: Ndarray<'K, float>) : Ndarray<'K, float> =
    Fowl.Core.Matrix.sum axis arr
    |> Ndarray.map (fun x -> x / float (Ndarray.shape arr).[axis])

/// Calculate variance (population variance by default)
let var (?ddof: int) (arr: Ndarray<'K, float>) : float =
    let ddof = defaultArg ddof 0
    let mu = mean arr
    let data = Ndarray.toArray arr
    let n = float (Array.length data - ddof)
    data |> Array.sumBy (fun x -> (x - mu) ** 2.0) |> fun s -> s / n

/// Calculate standard deviation
let std (?ddof: int) (arr: Ndarray<'K, float>) : float =
    sqrt (var ?ddof=ddof arr)

/// Calculate median
let median (arr: Ndarray<'K, float>) : float =
    let data = arr |> Ndarray.toArray |> Array.sort
    let n = data.Length
    if n % 2 = 1 then
        data.[n / 2]
    else
        (data.[n / 2 - 1] + data.[n / 2]) / 2.0

/// Calculate p-th percentile (0 <= p <= 100)
let percentile (p: float) (arr: Ndarray<'K, float>) : FowlResult<float> =
    if p < 0.0 || p > 100.0 then
        Error.invalidArgument "Percentile must be between 0 and 100"
    else
        let data = arr |> Ndarray.toArray |> Array.sort
        let n = float data.Length
        let idx = (p / 100.0) * (n - 1.0)
        let lower = int (floor idx)
        let upper = int (ceil idx)
        let frac = idx - float lower
        if lower = upper then
            Ok data.[lower]
        else
            Ok (data.[lower] * (1.0 - frac) + data.[upper] * frac)

/// Calculate skewness (measure of asymmetry)
let skewness (arr: Ndarray<'K, float>) : float =
    let data = Ndarray.toArray arr
    let n = float data.Length
    let mu = mean arr
    let sigma = std arr
    let m3 = data |> Array.sumBy (fun x -> (x - mu) ** 3.0) / n
    m3 / (sigma ** 3.0)

/// Calculate kurtosis (measure of tail heaviness)
let kurtosis (arr: Ndarray<'K, float>) : float =
    let data = Ndarray.toArray arr
    let n = float data.Length
    let mu = mean arr
    let sigma = std arr
    let m4 = data |> Array.sumBy (fun x -> (x - mu) ** 4.0) / n
    m4 / (sigma ** 4.0) - 3.0  // Excess kurtosis (Fisher definition)

/// Calculate n-th central moment
let moment (n: int) (arr: Ndarray<'K, float>) : FowlResult<float> =
    if n < 0 then
        Error.invalidArgument "Moment order must be non-negative"
    else
        let data = Ndarray.toArray arr
        let mu = mean arr
        Ok (data |> Array.sumBy (fun x -> (x - mu) ** float n) / float data.Length)

/// Minimum value
let min (arr: Ndarray<'K, float>) : float =
    arr |> Ndarray.toArray |> Array.min

/// Maximum value
let max (arr: Ndarray<'K, float>) : float =
    arr |> Ndarray.toArray |> Array.max

/// Range (max - min)
let range (arr: Ndarray<'K, float>) : float =
    max arr - min arr

/// Sum of all elements
let sum (arr: Ndarray<'K, float>) : float =
    arr |> Ndarray.toArray |> Array.sum

/// Product of all elements
let prod (arr: Ndarray<'K, float>) : float =
    arr |> Ndarray.toArray |> Array.reduce (*)

/// Cumulative sum
let cumsum (arr: Ndarray<'K, float>) : Ndarray<'K, float> =
    let data = Ndarray.toArray arr
    let result = Array.zeroCreate data.Length
    let mutable acc = 0.0
    for i = 0 to data.Length - 1 do
        acc <- acc + data.[i]
        result.[i] <- acc
    Ndarray.ofArray result (Ndarray.shape arr)

/// Cumulative product
let cumprod (arr: Ndarray<'K, float>) : Ndarray<'K, float> =
    let data = Ndarray.toArray arr
    let result = Array.zeroCreate data.Length
    let mutable acc = 1.0
    for i = 0 to data.Length - 1 do
        acc <- acc * data.[i]
        result.[i] <- acc
    Ndarray.ofArray result (Ndarray.shape arr)
