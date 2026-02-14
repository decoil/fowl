namespace Fowl.Stats

open System
open Fowl
open Fowl.Core.Types

/// <summary>Non-parametric tests for comparing groups.
/// Use when data doesn't meet normality assumptions.
/// </summary>module NonParametricTests =
    
    /// <summary>Mann-Whitney U test (Wilcoxon rank-sum test).
/// Non-parametric test comparing two independent groups.
/// H0: Distributions are equal (medians equal under shift alternative).
/// </summary>type MannWhitneyResult = {
        /// U statistic (minimum of U1 and U2)
        UStatistic: float
        /// Z-score (normalized U)
        ZScore: float
        /// Two-tailed p-value
        PValue: float
        /// Sample 1 rank sum
        RankSum1: float
        /// Sample 2 rank sum
        RankSum2: float
        /// U1 statistic
        U1: float
        /// U2 statistic
        U2: float
        /// Effect size (r = Z / sqrt(N))
        EffectSize: float
    }
    
    /// <summary>Perform Mann-Whitney U test.
/// </summary>let mannWhitneyU (sample1: float[]) (sample2: float[]) 
                     : FowlResult<MannWhitneyResult> =
        result {
            let n1 = sample1.Length
            let n2 = sample2.Length
            
            if n1 = 0 || n2 = 0 then
                return! Error.invalidArgument "Mann-Whitney U requires non-empty samples"
            if n1 < 3 || n2 < 3 then
                return! Error.invalidArgument "Mann-Whitney U requires at least 3 observations per sample"
            
            // Combine and rank all observations
            let combined = 
                Array.append 
                    (sample1 |> Array.map (fun x -> (x, 1)))  // Tag as group 1
                    (sample2 |> Array.map (fun x -> (x, 2)))  // Tag as group 2
            
            // Sort by value
            let sorted = combined |> Array.sortBy fst
            
            // Assign ranks (handling ties with average rank)
            let n = n1 + n2
            let mutable ranks = Array.zeroCreate n
            let mutable i = 0
            
            while i < n do
                let currentVal = fst sorted.[i]
                // Find all tied values
                let tieEnd = 
                    sorted
                    |> Array.tryFindIndex (fun j -> j > i && fst sorted.[j] > currentVal)
                    |> Option.defaultValue n
                
                let tieCount = tieEnd - i
                let avgRank = float (i + tieEnd - 1) / 2.0 + 1.0  // 1-based ranking
                
                for j = i to tieEnd - 1 do
                    ranks.[j] <- avgRank
                
                i <- tieEnd
            
            // Calculate rank sums for each group
            let mutable rankSum1 = 0.0
            let mutable rankSum2 = 0.0
            
            for i = 0 to n - 1 do
                let (_, group) = sorted.[i]
                if group = 1 then
                    rankSum1 <- rankSum1 + ranks.[i]
                else
                    rankSum2 <- rankSum2 + ranks.[i]
            
            // Calculate U statistics
            let u1 = rankSum1 - float (n1 * (n1 + 1)) / 2.0
            let u2 = rankSum2 - float (n2 * (n2 + 1)) / 2.0
            
            // Use minimum U for test
            let u = min u1 u2
            
            // Calculate tie correction factor
            let tieCorrection =
                let mutable tSum = 0.0
                let mutable i = 0
                while i < n do
                    let currentVal = fst sorted.[i]
                    let tieEnd = 
                        sorted
                        |> Array.tryFindIndex (fun j -> j > i && fst sorted.[j] > currentVal)
                        |> Option.defaultValue n
                    let t = float (tieEnd - i)
                    if t > 1.0 then
                        tSum <- tSum + (t ** 3.0 - t)
                    i <- int tieEnd
                tSum
            
            // Standard deviation with tie correction
            let sdU = 
                sqrt (
                    (float (n1 * n2) / 12.0) *
                    (float (n + 1) - tieCorrection / (float (n * (n - 1))))
                )
            
            // Mean of U under H0
            let meanU = float (n1 * n2) / 2.0
            
            // Z-score (with continuity correction)
            let z = (u - meanU + 0.5) / sdU
            
            // Two-tailed p-value from standard normal
            let! pValueNormal = GaussianDistribution.cdf 0.0 1.0 (abs z)
            let pValue = 2.0 * (1.0 - pValueNormal)
            
            // Effect size: r = Z / sqrt(N)
            let effectSize = abs z / sqrt (float n)
            
            return {
                UStatistic = u
                ZScore = z
                PValue = pValue
                RankSum1 = rankSum1
                RankSum2 = rankSum2
                U1 = u1
                U2 = u2
                EffectSize = effectSize
            }
        }
    
    /// <summary>Wilcoxon signed-rank test.
/// Non-parametric test for paired samples.
/// H0: Median difference is zero.
/// </summary>type WilcoxonResult = {
        /// W+ statistic (sum of positive ranks)
        WPlus: float
        /// W- statistic (sum of negative ranks)
        WMinus: float
        /// Test statistic (minimum of W+, W-)
        WStatistic: float
        /// Z-score
        ZScore: float
        /// Two-tailed p-value
        PValue: float
        /// Number of non-zero differences
        N: int
    }
    
    /// <summary>Perform Wilcoxon signed-rank test.
/// </summary>let wilcoxonSignedRank (before: float[]) (after: float[]) 
                         : FowlResult<WilcoxonResult> =
        result {
            if before.Length <> after.Length then
                return! Error.invalidArgument "Wilcoxon requires paired samples of equal length"
            if before.Length < 5 then
                return! Error.invalidArgument "Wilcoxon requires at least 5 pairs"
            
            // Calculate differences
            let diffs = Array.map2 (-) after before
            
            // Remove zero differences
            let nonZeroDiffs = 
                diffs
                |> Array.mapi (fun i d -> (i, d))
                |> Array.filter (fun (_, d) -> d <> 0.0)
            
            let n = nonZeroDiffs.Length
            
            if n < 5 then
                return! Error.invalidArgument "Wilcoxon requires at least 5 non-zero differences"
            
            // Rank absolute differences
            let absDiffs = nonZeroDiffs |> Array.map (fun (i, d) -> (i, abs d))
            let sortedByAbs = absDiffs |> Array.sortBy snd
            
            // Assign ranks (handling ties)
            let mutable ranks = Array.zeroCreate n
            let mutable i = 0
            while i < n do
                let currentAbs = snd sortedByAbs.[i]
                let tieEnd = 
                    sortedByAbs
                    |> Array.tryFindIndex (fun j -> j > i && snd sortedByAbs.[j] > currentAbs)
                    |> Option.defaultValue n
                
                let avgRank = float (i + tieEnd - 1) / 2.0 + 1.0
                for j = i to tieEnd - 1 do
                    ranks.[j] <- avgRank
                i <- tieEnd
            
            // Map ranks back to original indices
            let rankMap = 
                sortedByAbs
                |> Array.mapi (fun j (origIdx, _) -> (origIdx, ranks.[j]))
                |> Map.ofArray
            
            // Calculate W+ and W-
            let mutable wPlus = 0.0
            let mutable wMinus = 0.0
            
            for (origIdx, d) in nonZeroDiffs do
                let r = rankMap.[origIdx]
                if d > 0.0 then
                    wPlus <- wPlus + r
                else
                    wMinus <- wMinus + r
            
            let w = min wPlus wMinus
            
            // Normal approximation
            let meanW = float (n * (n + 1)) / 4.0
            let varW = float (n * (n + 1) * (2 * n + 1)) / 24.0
            let sdW = sqrt varW
            
            let z = (w - meanW + 0.5) / sdW
            
            let! pValueNormal = GaussianDistribution.cdf 0.0 1.0 (abs z)
            let pValue = 2.0 * (1.0 - pValueNormal)
            
            return {
                WPlus = wPlus
                WMinus = wMinus
                WStatistic = w
                ZScore = z
                PValue = pValue
                N = n
            }
        }
    
    /// <summary>Interpret non-parametric test result.
/// </summary>let interpretMannWhitney (result: MannWhitneyResult) (alpha: float) : string =
        if result.PValue < alpha then
            sprintf "Reject H0: Distributions differ significantly (U=%.1f, p=%.4f, r=%.3f)"
                result.UStatistic result.PValue result.EffectSize
        else
            sprintf "Fail to reject H0: No significant difference (U=%.1f, p=%.4f)"
                result.UStatistic result.PValue
    
    /// <summary>Interpret Wilcoxon test result.
/// </summary>let interpretWilcoxon (result: WilcoxonResult) (alpha: float) : string =
        if result.PValue < alpha then
            sprintf "Reject H0: Paired samples differ significantly (W=%.1f, p=%.4f)"
                result.WStatistic result.PValue
        else
            sprintf "Fail to reject H0: No significant difference (W=%.1f, p=%.4f)"
                result.WStatistic result.PValue