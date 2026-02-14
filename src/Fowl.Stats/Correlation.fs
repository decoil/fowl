module Fowl.Stats.Correlation

open Fowl

/// Calculate covariance between two arrays
let covariance (x: Ndarray<'K, float>) (y: Ndarray<'K, float>) : float =
    if Ndarray.shape x <> Ndarray.shape y then
        failwith "Arrays must have same shape"
    
    let xData = Ndarray.toArray x
    let yData = Ndarray.toArray y
    let n = float xData.Length
    
    let meanX = Array.average xData
    let meanY = Array.average yData
    
    Array.zip xData yData
    |> Array.sumBy (fun (xi, yi) -> (xi - meanX) * (yi - meanY))
    |> fun s -> s / n

/// Calculate Pearson correlation coefficient
let pearsonCorrelation (x: Ndarray<'K, float>) (y: Ndarray<'K, float>) : float =
    let cov = covariance x y
    let varX = Descriptive.var x
    let varY = Descriptive.var y
    cov / sqrt (varX * varY)

/// Calculate correlation matrix for 2D array (variables in columns)
let correlationMatrix (arr: Ndarray<'K, float>) : Ndarray<'K, float> =
    match Ndarray.shape arr with
    | [|n; m|] ->
        let result = Ndarray.zeros<'K> [|m; m|]
        for i = 0 to m - 1 do
            for j = 0 to m - 1 do
                let colI = Fowl.Core.Slice.slice arr [|SliceSpec.All; SliceSpec.Index i|]
                let colJ = Fowl.Core.Slice.slice arr [|SliceSpec.All; SliceSpec.Index j|]
                let corr = pearsonCorrelation colI colJ
                Ndarray.set result [|i; j|] corr
        result
    | _ -> failwith "correlationMatrix requires 2D array"
