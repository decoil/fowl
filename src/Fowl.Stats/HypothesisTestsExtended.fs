namespace Fowl.Stats

open System
open Fowl
open Fowl.Core.Types

/// <summary>Additional hypothesis tests.
/// Kruskal-Wallis, Levene, Bartlett tests.
/// </summary>module HypothesisTestsExtended =
    
    /// <summary>Kruskal-Wallis H-test.
/// Non-parametric ANOVA for comparing 3+ independent groups.
/// H0: All groups have same distribution.
/// </summary>type KruskalWallisResult = {
        /// H-statistic
        HStatistic: float
        /// P-value
        PValue: float
        /// Degrees of freedom
        DF: int
        /// Mean rank per group
        MeanRanks: float[]
        /// Overall mean rank
        OverallMeanRank: float
    }
    
    /// <summary>Kruskal-Wallis test for 3+ groups.
/// </summary>let kruskalWallis (groups: float[][]) : FowlResult<KruskalWallisResult> =
        result {
            let k = groups.Length
            if k < 3 then
                return! Error.invalidArgument "Kruskal-Wallis requires at least 3 groups"
            
            // Filter out empty groups
            let validGroups = groups |> Array.filter (fun g -> g.Length > 0)
            if validGroups.Length < 3 then
                return! Error.invalidArgument "Kruskal-Wallis requires at least 3 non-empty groups"
            
            // Combine all data with group labels
            let labeledData = 
                validGroups
                |> Array.mapi (fun groupIdx groupValues -
                    groupValues |> Array.map (fun v -> (v, groupIdx)))
                |> Array.concat
            
            let n = labeledData.Length
            
            // Sort by value and assign ranks
            let sorted = labeledData |> Array.sortBy fst
            
            // Assign ranks with tie handling
            let mutable ranks = Array.zeroCreate n
            let mutable i = 0
            while i < n do
                let currentVal = fst sorted.[i]
                // Find end of tie group
                let mutable j = i + 1
                while j < n && fst sorted.[j] = currentVal do
                    j <- j + 1
                
                // Average rank for tied values
                let avgRank = float (i + j - 1) / 2.0 + 1.0
                for k = i to j - 1 do
                    ranks.[k] <- avgRank
                
                i <- j
            
            // Calculate tie correction factor
            let mutable tieCorrection = 0.0
            i <- 0
            while i < n do
                let currentVal = fst sorted.[i]
                let mutable j = i + 1
                while j < n && fst sorted.[j] = currentVal do
                    j <- j + 1
                let t = float (j - i)
                if t > 1.0 then
                    tieCorrection <- tieCorrection + (t ** 3.0 - t)
                i <- j
            
            tieCorrection <- 1.0 - tieCorrection / (float (n * n * n - n))
            
            // Sum of ranks per group
            let rankSums = Array.zeroCreate validGroups.Length
            for i = 0 to n - 1 do
                let (_, groupIdx) = sorted.[i]
                rankSums.[groupIdx] <- rankSums.[groupIdx] + ranks.[i]
            
            // Calculate H statistic
            let mutable h = 0.0
            for i = 0 to validGroups.Length - 1 do
                let ni = float validGroups.[i].Length
                h <- h + (rankSums.[i] * rankSums.[i]) / ni
            
            let nFloat = float n
            h <- (12.0 / (nFloat * (nFloat + 1.0))) * h - 3.0 * (nFloat + 1.0)
            
            // Apply tie correction
            if tieCorrection > 0.0 then
                h <- h / tieCorrection
            
            // Degrees of freedom
            let df = validGroups.Length - 1
            
            // P-value from chi-square
            let! pValue = ChiSquareDistribution.cdf (float df) h
            let pValue = 1.0 - pValue
            
            // Mean ranks
            let meanRanks = 
                Array.map2 (fun sum count -> sum / float count) rankSums (validGroups |> Array.map (fun g -> g.Length))
            
            let overallMeanRank = (nFloat + 1.0) / 2.0
            
            return {
                HStatistic = h
                PValue = pValue
                DF = df
                MeanRanks = meanRanks
                OverallMeanRank = overallMeanRank
            }
        }
    
    /// <summary>Levene's test for equality of variances.
/// Tests if multiple groups have equal variances.
/// Robust to non-normality.
/// </summary>type LeveneResult = {
        /// W-statistic
        WStatistic: float
        /// P-value
        PValue: float
        /// Degrees of freedom 1
        DF1: int
        /// Degrees of freedom 2
        DF2: int
    }
    
    /// <summary>Levene's test.
/// Uses absolute deviations from group medians (Brown-Forsythe variant).
/// </summary>let levene (groups: float[][]) : FowlResult<LeveneResult> =
        result {
            let k = groups.Length
            if k < 2 then
                return! Error.invalidArgument "Levene test requires at least 2 groups"
            
            // Compute group medians and absolute deviations
            let groupMedians = 
                groups
                |> Array.map (fun g -
                    if g.Length = 0 then None
                    else Descriptive.median g |> Result.toOption)
                |> Array.choose id
            
            if groupMedians.Length < 2 then
                return! Error.invalidArgument "Levene test requires at least 2 non-empty groups"
            
            // Compute absolute deviations from medians
            let absDeviations = 
                groups
                |> Array.mapi (fun i g -
                    let med = groupMedians.[i]
                    g |> Array.map (fun x -> abs (x - med)))
            
            // Perform ANOVA on absolute deviations
            let! anovaResult = Anova.oneWay absDeviations
            
            return {
                WStatistic = anovaResult.FStatistic
                PValue = anovaResult.PValue
                DF1 = anovaResult.DFBetween
                DF2 = anovaResult.DFWithin
            }
        }
    
    /// <summary>Bartlett's test for equality of variances.
/// More sensitive than Levene but assumes normality.
/// </summary>type BartlettResult = {
        /// T-statistic (chi-square)
        TStatistic: float
        /// P-value
        PValue: float
        /// Degrees of freedom
        DF: int
    }
    
    /// <summary>Bartlett's test.
/// Tests homogeneity of variances.
/// </summary>let bartlett (groups: float[][]) : FowlResult<BartlettResult> =
        result {
            let k = groups.Length
            if k < 2 then
                return! Error.invalidArgument "Bartlett test requires at least 2 groups"
            
            // Filter empty groups and compute sample variances
            let validGroups = groups |> Array.filter (fun g -> g.Length > 1)  // Need at least 2 for variance
            
            if validGroups.Length < 2 then
                return! Error.invalidArgument "Bartlett test requires at least 2 groups with 2+ observations"
            
            let groupSizes = validGroups |> Array.map (fun g -> float g.Length)
            let groupVars = 
                validGroups
                |> Array.map (fun g -> Descriptive.var g |> Result.defaultValue 0.0)
            
            let nTotal = Array.sum groupSizes
            let kFloat = float validGroups.Length
            
            // Pooled variance
            let pooledVar = 
                Array.map2 (fun n var -> (n - 1.0) * var) groupSizes groupVars
                |> Array.sum
                |> fun sum -> sum / (nTotal - kFloat)
            
            // Bartlett's T statistic
            let mutable t = 0.0
            for i = 0 to validGroups.Length - 1 do
                let ni = groupSizes.[i]
                let si2 = groupVars.[i]
                if si2 > 0.0 then
                    t <- t + (ni - 1.0) * log (si2 / pooledVar)
            
            // Correction factor
            let correction = 
                1.0 + (1.0 / (3.0 * (kFloat - 1.0))) * 
                (Array.sumBy (fun n -> 1.0 / (n - 1.0)) groupSizes - 1.0 / (nTotal - kFloat))
            
            let tCorrected = -t / correction
            
            // Degrees of freedom
            let df = validGroups.Length - 1
            
            // P-value from chi-square
            let! pValue = ChiSquareDistribution.cdf (float df) tCorrected
            let pValue = 1.0 - pValue
            
            return {
                TStatistic = tCorrected
                PValue = pValue
                DF = df
            }
        }
    
    /// <summary>Friedman test.
/// Non-parametric repeated measures ANOVA.
/// For comparing 3+ related samples.
/// </summary>type FriedmanResult = {
        /// Chi-square statistic
        ChiSquare: float
        /// P-value
        PValue: float
        /// Degrees of freedom
        DF: int
    }
    
    /// <summary>Friedman test for repeated measures.
/// </summary>let friedman (data: float[][]) : FowlResult<FriedmanResult> =
        result {
            let k = data.Length  // Number of treatments
            
            if k < 3 then
                return! Error.invalidArgument "Friedman test requires at least 3 treatments"
            
            // Check all groups have same size
            let n = data.[0].Length
            if data |> Array.exists (fun g -> g.Length <> n) then
                return! Error.invalidArgument "Friedman test requires equal sample sizes"
            if n = 0 then
                return! Error.invalidArgument "Friedman test requires non-empty samples"
            
            // Rank treatments within each block (subject)
            let mutable rankSums = Array.zeroCreate k
            
            for block = 0 to n - 1 do
                // Get values for this block
                let blockValues = data |> Array.map (fun g -> g.[block])
                
                // Rank within block (handle ties with average rank)
                let indexed = blockValues |> Array.mapi (fun i v -> (i, v))
                let sorted = indexed |> Array.sortBy snd
                
                // Assign ranks
                let mutable i = 0
                while i < k do
                    let currentVal = snd sorted.[i]
                    let mutable j = i + 1
                    while j < k && snd sorted.[j] = currentVal do
                        j <- j + 1
                    
                    let avgRank = float (i + j - 1) / 2.0 + 1.0
                    for idx = i to j - 1 do
                        let (origIdx, _) = sorted.[idx]
                        rankSums.[origIdx] <- rankSums.[origIdx] + avgRank
                    
                    i <- j
            
            // Calculate chi-square statistic
            let nFloat = float n
            let kFloat = float k
            let meanRank = (kFloat + 1.0) / 2.0
            
            let chiSq = 
                (12.0 / (nFloat * kFloat * (kFloat + 1.0))) *
                (rankSums |> Array.sumBy (fun r -> (r - nFloat * meanRank) ** 2.0))
            
            let df = k - 1
            
            let! pValue = ChiSquareDistribution.cdf (float df) chiSq
            let pValue = 1.0 - pValue
            
            return {
                ChiSquare = chiSq
                PValue = pValue
                DF = df
            }
        }