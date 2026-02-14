module Fowl.Linalg.Core

open Fowl
open Fowl.Core.Types

/// Identity matrix
let eye (n: int) : FowlResult<Ndarray<'K, float>> =
    if n <= 0 then
        Error.invalidArgument "eye requires positive dimension"
    else
        let data = Array.zeroCreate (n * n)
        for i = 0 to n - 1 do
            data.[i * n + i] <- 1.0
        Ndarray.ofArray data [|n; n|]

/// Create diagonal matrix from vector
let diag (v: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
    let shape = Ndarray.shape v
    match shape with
    | [|n|] | [|n; 1|] | [|1; n|] ->
        let data = Array.zeroCreate (n * n)
        let values = Ndarray.toArray v
        for i = 0 to n - 1 do
            data.[i * n + i] <- values.[i]
        Ndarray.ofArray data [|n; n|]
    | _ -> Error.invalidArgument "diag requires 1D array"

/// Extract diagonal from matrix
let getDiag (a: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
    match Ndarray.shape a with
    | [|m; n|] ->
        let k = min m n
        let data = Array.init k (fun i -> 
            match Ndarray.get a [|i; i|] with
            | Ok v -> v
            | Error _ -> 0.0)
        Ndarray.ofArray data [|k|]
    | _ -> Error.invalidArgument "getDiag requires 2D matrix"

/// Trace of matrix (sum of diagonal)
let trace (a: Ndarray<'K, float>) : FowlResult<float> =
    getDiag a
    |> Result.map (fun diag -> Ndarray.toArray diag |> Array.sum)

/// Frobenius norm
let normFrobenius (a: Ndarray<'K, float>) : float =
    a
    |> Ndarray.toArray
    |> Array.sumBy (fun x -> x * x)
    |> sqrt

/// Matrix 1-norm (max column sum)
let norm1 (a: Ndarray<'K, float>) : FowlResult<float> =
    match Ndarray.shape a with
    | [|m; n|] ->
        let mutable maxSum = 0.0
        for j = 0 to n - 1 do
            let mutable colSum = 0.0
            for i = 0 to m - 1 do
                match Ndarray.get a [|i; j|] with
                | Ok v -> colSum <- colSum + abs v
                | Error _ -> ()
            maxSum <- max maxSum colSum
        Ok maxSum
    | _ -> Error.invalidArgument "norm1 requires 2D matrix"

/// Matrix infinity-norm (max row sum)
let normInf (a: Ndarray<'K, float>) : FowlResult<float> =
    match Ndarray.shape a with
    | [|m; n|] ->
        let mutable maxSum = 0.0
        for i = 0 to m - 1 do
            let mutable rowSum = 0.0
            for j = 0 to n - 1 do
                match Ndarray.get a [|i; j|] with
                | Ok v -> rowSum <- rowSum + abs v
                | Error _ -> ()
            maxSum <- max maxSum rowSum
        Ok maxSum
    | _ -> Error.invalidArgument "normInf requires 2D matrix"
