namespace Fowl.Stats

open System
open Fowl
open Fowl.Core.Types

/// <summary>
/// Rank correlation coefficients.
/// Spearman and Kendall for non-parametric correlation.
/// </summary>
module RankCorrelation =
    
    /// <summary>
    /// Rank data (assign ranks with tie handling).
    /// Returns array of ranks.
    /// </summary>
    let private rankData (data: float[]) : float[] =
        let n = data.Length
        let indexed = data |> Array.mapi (fun i x -> (i, x))
        let sorted = indexed |> Array.sortBy snd
        
        let ranks = Array.zeroCreate n
        let mutable i = 0
        
        while i < n do
            let currentVal = snd sorted.[i]
            let mutable j = i + 1
            while j < n && snd sorted.[j] = currentVal do
                j <- j + 1
            
            // Average rank for ties
            let avgRank = float (i + j - 1) / 2.0 + 1.0
            for k = i to j - 1 do
                let (origIdx, _) = sorted.[k]
                ranks.[origIdx] <- avgRank
            
            i <- j
        
        ranks
    
    /// <summary>
    /// Spearman rank correlation coefficient.
    /// Non-parametric measure of rank correlation.
    /// ρ = 1 - (6Σd²)/(n(n²-1))
    /// where d = rank(x) - rank(y)
    /// </summary>
    let spearman (x: float[]) (y: float[]) : FowlResult<float> =
        result {
            if x.Length <> y.Length then
                return! Error.invalidArgument "Spearman requires arrays of same length"
            if x.Length < 2 then
                return! Error.invalidArgument "Spearman requires at least 2 observations"
            
            let n = float x.Length
            
            // Rank both arrays
            let rankX = rankData x
            let rankY = rankData y
            
            // Calculate sum of squared rank differences
            let sumSqDiff = 
                Array.map2 (fun rx ry -> (rx - ry) ** 2.0) rankX rankY
                |> Array.sum
            
            // Spearman's formula with tie correction would be better,
            // but this is the standard formula
            let rho = 1.0 - (6.0 * sumSqDiff) / (n * (n * n - 1.0))
            
            return rho
        }
    
    /// <summary>
    /// Spearman correlation with p-value.
    /// </summary>
    let spearmanWithPValue (x: float[]) (y: float[]) : FowlResult<(float * float)> =
        result {
            let! rho = spearman x y
            let n = float x.Length
            
            // t-statistic for significance test
            // t = r * sqrt((n-2)/(1-r²))
            let t = rho * sqrt ((n - 2.0) / (1.0 - rho * rho))
            
            // P-value from t-distribution with n-2 df
            let! pValueTwoTailed = StudentTDistribution.cdf (n - 2.0) (abs t)
            let pValue = 2.0 * (1.0 - pValueTwoTailed)
            
            return (rho, pValue)
        }
    
    /// <summary>
    /// Kendall tau correlation coefficient.
    /// Measures ordinal association.
    /// τ = (n_c - n_d) / √(n_0 * n_1)
    /// where n_c = concordant pairs, n_d = discordant pairs
    /// </summary>
    let kendall (x: float[]) (y: float[]) : FowlResult<float> =
        result {
            if x.Length <> y.Length then
                return! Error.invalidArgument "Kendall requires arrays of same length"
            if x.Length < 2 then
                return! Error.invalidArgument "Kendall requires at least 2 observations"
            
            let n = x.Length
            let mutable concordant = 0
            let mutable discordant = 0
            let mutable tiesX = 0
            let mutable tiesY = 0
            
            // Count concordant and discordant pairs
            for i = 0 to n - 2 do
                for j = i + 1 to n - 1 do
                    let dx = x.[j] - x.[i]
                    let dy = y.[j] - y.[i]
                    
                    if dx = 0.0 && dy = 0.0 then
                        tiesX <- tiesX + 1
                        tiesY <- tiesY + 1
                    elif dx = 0.0 then
                        tiesX <- tiesX + 1
                    elif dy = 0.0 then
                        tiesY <- tiesY + 1
                    elif (dx > 0.0 && dy > 0.0) || (dx < 0.0 && dy < 0.0) then
                        concordant <- concordant + 1
                    else
                        discordant <- discordant + 1
            
            let n0 = float (n * (n - 1) / 2)  // Total pairs
            let n1 = float (n * (n - 1) / 2 - tiesX)  // Adjusted for x ties
            let n2 = float (n * (n - 1) / 2 - tiesY)  // Adjusted for y ties
            
            let tau = float (concordant - discordant) / sqrt (n1 * n2)
            
            return tau
        }
    
    /// <summary>
    /// Kendall tau with p-value.
    /// </summary>
    let kendallWithPValue (x: float[]) (y: float[]) : FowlResult<(float * float)> =
        result {
            let! tau = kendall x y
            let n = float x.Length
            
            // Variance of Kendall's tau
            let var = (2.0 * (2.0 * n + 5.0)) / (9.0 * n * (n - 1.0))
            
            // z-score
            let z = tau / sqrt var
            
            // P-value from standard normal
            let! pValueTwoTailed = GaussianDistribution.cdf 0.0 1.0 (abs z)
            let pValue = 2.0 * (1.0 - pValueTwoTailed)
            
            return (tau, pValue)
        }
    
    /// <summary>
    /// Point-biserial correlation.
    /// Correlation between continuous and dichotomous variables.
    /// </summary>
    let pointBiserial (continuous: float[]) (dichotomous: float[]) : FowlResult<float> =
        result {
            if continuous.Length <> dichotomous.Length then
                return! Error.invalidArgument "Point-biserial requires arrays of same length"
            
            // Check dichotomous values are 0 or 1
            if dichotomous |> Array.exists (fun x -> x <> 0.0 && x <> 1.0) then
                return! Error.invalidArgument "Point-biserial requires dichotomous values to be 0 or 1"
            
            let n = float continuous.Length
            let n1 = dichotomous |> Array.filter ((=) 1.0) |> Array.length |> float
            let n0 = n - n1
            
            if n1 = 0.0 || n0 = 0.0 then
                return! Error.invalidState "Point-biserial requires both groups to have observations"
            
            let mean1 = 
                Array.map2 (fun c d -> if d = 1.0 then Some c else None) continuous dichotomous
                |> Array.choose id
                |> Array.average
            
            let mean0 = 
                Array.map2 (fun c d -> if d = 0.0 then Some c else None) continuous dichotomous
                |> Array.choose id
                |> Array.average
            
            let! std = Descriptive.std continuous
            
            // Point-biserial formula
            let rpb = (mean1 - mean0) / std * sqrt (n1 * n0 / (n * n))
            
            return rpb
        }
    
    /// <summary>
    /// Rank correlation matrix for multiple variables.
    /// </summary>
    let spearmanMatrix (data: float[][]) : FowlResult<float[,]> =
        result {
            let n = data.Length
            if n = 0 then
                return! Error.invalidArgument "spearmanMatrix requires at least one variable"
            
            let corrMatrix = Array2D.zeroCreate n n
            
            for i = 0 to n - 1 do
                for j = 0 to n - 1 do
                    if i = j then
                        corrMatrix.[i, j] <- 1.0
                    else
                        let! corr = spearman data.[i] data.[j]
                        corrMatrix.[i, j] <- corr
            
            return corrMatrix
        }