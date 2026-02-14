module Fowl.Linalg.Core

open Fowl

/// Identity matrix
let eye (n: int) : Ndarray<'K, float> =
    let data = Array.zeroCreate (n * n)
    for i = 0 to n - 1 do
        data.[i * n + i] <- 1.0
    Ndarray.ofArray data [|n; n|]

/// Create diagonal matrix from vector
let diag (v: Ndarray<'K, float>) : Ndarray<'K, float> =
    let shape = Ndarray.shape v
    match shape with
    | [|n|] | [|n; 1|] | [|1; n|] ->
        let data = Array.zeroCreate (n * n)
        let values = Ndarray.toArray v
        for i = 0 to n - 1 do
            data.[i * n + i] <- values.[i]
        Ndarray.ofArray data [|n; n|]
    | _ -> failwith "diag requires 1D array"

/// Extract diagonal from matrix
let getDiag (a: Ndarray<'K, float>) : Ndarray<'K, float> =
    match Ndarray.shape a with
    | [|m; n|] ->
        let k = min m n
        let data = Array.init k (fun i -> Ndarray.get a [|i; i|])
        Ndarray.ofArray data [|k|]
    | _ -> failwith "getDiag requires 2D matrix"

/// Trace of matrix (sum of diagonal)
let trace (a: Ndarray<'K, float>) : float =
    getDiag a |> Ndarray.toArray |> Array.sum

/// Upper triangular part (k=0 includes diagonal, k>0 above diagonal)
let triu (k: int) (a: Ndarray<'K, float>) : Ndarray<'K, float> =
    match a with
    | Dense da ->
        match da.Shape with
        | [|m; n|] ->
            let out = Ndarray.zeros<'K> [|m; n|]
            for i = 0 to m - 1 do
                let startJ = max 0 (i + k)
                for j = startJ to n - 1 do
                    Ndarray.set out [|i; j|] (Ndarray.get a [|i; j|])
            out
        | _ -> failwith "triu requires 2D matrix"
    | _ -> failwith "triu not implemented for sparse"

/// Lower triangular part (k=0 includes diagonal, k<0 below diagonal)
let tril (k: int) (a: Ndarray<'K, float>) : Ndarray<'K, float> =
    match a with
    | Dense da ->
        match da.Shape with
        | [|m; n|] ->
            let out = Ndarray.zeros<'K> [|m; n|]
            for i = 0 to m - 1 do
                let endJ = min (n - 1) (i + k)
                for j = 0 to endJ do
                    Ndarray.set out [|i; j|] (Ndarray.get a [|i; j|])
            out
        | _ -> failwith "tril requires 2D matrix"
    | _ -> failwith "tril not implemented for sparse"

/// Frobenius norm
let normFrobenius (a: Ndarray<'K, float>) : float =
    a
    |> Ndarray.toArray
    |> Array.sumBy (fun x -> x * x)
    |> sqrt

/// Matrix 1-norm (max column sum)
let norm1 (a: Ndarray<'K, float>) : float =
    match Ndarray.shape a with
    | [|m; n|] ->
        let mutable maxSum = 0.0
        for j = 0 to n - 1 do
            let mutable colSum = 0.0
            for i = 0 to m - 1 do
                colSum <- colSum + abs (Ndarray.get a [|i; j|])
            maxSum <- max maxSum colSum
        maxSum
    | _ -> failwith "norm1 requires 2D matrix"

/// Matrix infinity-norm (max row sum)
let normInf (a: Ndarray<'K, float>) : float =
    match Ndarray.shape a with
    | [|m; n|] ->
        let mutable maxSum = 0.0
        for i = 0 to m - 1 do
            let mutable rowSum = 0.0
            for j = 0 to n - 1 do
                rowSum <- rowSum + abs (Ndarray.get a [|i; j|])
            maxSum <- max maxSum rowSum
        maxSum
    | _ -> failwith "normInf requires 2D matrix"
