namespace Fowl.Stats

open System
open Fowl
open Fowl.Core.Types

/// <summary>ANOVA (Analysis of Variance) tests.
/// Compare means across multiple groups.
/// </summary>module Anova =
    
    /// <summary>Result of one-way ANOVA.
/// </summary>type AnovaResult = {
        /// F-statistic
        FStatistic: float
        /// P-value
        PValue: float
        /// Between-group sum of squares
        SSBetween: float
        /// Within-group sum of squares (error)
        SSWithin: float
        /// Total sum of squares
        SSTotal: float
        /// Between-group degrees of freedom
        DFBetween: int
        /// Within-group degrees of freedom
        DFWithin: int
        /// Mean square between groups
        MSBetween: float
        /// Mean square within groups (error)
        MSWithin: float
        /// Group means
        GroupMeans: float[]
        /// Grand mean
        GrandMean: float
    }
    
    /// <summary>One-way ANOVA.
/// Tests if means of 3+ groups are equal.
/// H0: μ₁ = μ₂ = ... = μₖ
/// </summary>let oneWay (groups: float[][]) : FowlResult<AnovaResult> =
        result {
            let k = groups.Length
            if k < 2 then
                return! Error.invalidArgument "ANOVA requires at least 2 groups"
            
            // Compute group statistics
            let groupStats = 
                groups
                |> Array.map (fun g -
                    if g.Length = 0 then
                        None
                    else
                        Some (Array.average g, g.Length, Array.sumBy (fun x -> x * x) g))
                |> Array.choose id
            
            if groupStats.Length < 2 then
                return! Error.invalidArgument "ANOVA requires at least 2 non-empty groups"
            
            let groupMeans = groupStats |> Array.map (fun (m, _, _) -> m)
            let groupSizes = groupStats |> Array.map (fun (_, n, _) -> n)
            let groupSumsSq = groupStats |> Array.map (fun (_, _, ss) -> ss)
            
            let nTotal = Array.sum groupSizes
            let grandMean = 
                Array.map2 (*) groupMeans (groupSizes |> Array.map float)
                |> Array.sum
                |> fun s -> s / float nTotal
            
            // Sum of squares between groups
            let ssBetween = 
                Array.map2 (fun m n -> float n * (m - grandMean) ** 2.0) groupMeans groupSizes
                |> Array.sum
            
            // Sum of squares within groups (error)
            let ssWithin = 
                Array.map2 (fun ss (m, n) -
                    ss - float n * m * m) groupSumsSq (Array.zip groupMeans groupSizes)
                |> Array.sum
            
            let ssTotal = ssBetween + ssWithin
            
            // Degrees of freedom
            let dfBetween = k - 1
            let dfWithin = nTotal - k
            
            // Mean squares
            let msBetween = ssBetween / float dfBetween
            let msWithin = ssWithin / float dfWithin
            
            if msWithin = 0.0 then
                return! Error.invalidState "ANOVA MSWithin is zero (no variation within groups)"
            
            // F-statistic
            let fStat = msBetween / msWithin
            
            // P-value from F-distribution
            let! pValue = FDistribution.cdf (float dfBetween) (float dfWithin) fStat
            let pValue = 1.0 - pValue  // Upper tail
            
            return {
                FStatistic = fStat
                PValue = pValue
                SSBetween = ssBetween
                SSWithin = ssWithin
                SSTotal = ssTotal
                DFBetween = dfBetween
                DFWithin = dfWithin
                MSBetween = msBetween
                MSWithin = msWithin
                GroupMeans = groupMeans
                GrandMean = grandMean
            }
        }
    
    /// <summary>Interpret ANOVA result.
/// Returns significance at α = 0.05 level.
/// </summary>let interpret (result: AnovaResult) (alpha: float) : string =
        if result.PValue < alpha then
            sprintf "Reject H0: Significant difference between group means (F=%.4f, p=%.4f)" 
                result.FStatistic result.PValue
        else
            sprintf "Fail to reject H0: No significant difference between group means (F=%.4f, p=%.4f)"
                result.FStatistic result.PValue
    
    /// <summary>Effect size (eta-squared): proportion of variance explained.
/// η² = SSBetween / SSTotal
/// </summary>let etaSquared (result: AnovaResult) : float =
        result.SSBetween / result.SSTotal
    
    /// <summary>Adjusted effect size (omega-squared).
/// Less biased than eta-squared.
/// </summary>let omegaSquared (result: AnovaResult) : float =
        let dfBetween = float result.DFBetween
        let dfWithin = float result.DFWithin
        let msWithin = result.MSWithin
        let ssBetween = result.SSBetween
        let ssTotal = result.SSTotal
        
        (ssBetween - dfBetween * msWithin) / (ssTotal + msWithin)
    
    /// <summary>Post-hoc Tukey HSD test.
/// Compare all pairs of groups after significant ANOVA.
/// </summary>let tukeyHSD (groups: float[][]) (result: AnovaResult) 
                 (alpha: float) : FowlResult<(string * string * float * bool)[]> =
        result {
            let k = groups.Length
            let n = Array.sumBy (fun (g: float[]) -> g.Length) groups
            let df = n - k
            
            let msError = result.MSWithin
            let stderr x1 x2 = sqrt (msError * (1.0/float x1 + 1.0/float x2))
            
            // Studentized range distribution critical value
            // Approximate using q = sqrt(2) * t for large df
            let! tCrit = StudentTDistribution.ppf (float df) (1.0 - alpha/2.0)
            let qCrit = sqrt(2.0) * tCrit
            
            let comparisons =
                [for i = 0 to k - 1 do
                 for j = i + 1 to k - 1 do
                    let meanDiff = abs (result.GroupMeans.[i] - result.GroupMeans.[j])
                    let se = stderr groups.[i].Length groups.[j].Length
                    let q = meanDiff / se
                    let significant = q > qCrit
                    yield (sprintf "Group%d" i, sprintf "Group%d" j, q, significant)]
            
            return Array.ofList comparisons
        }