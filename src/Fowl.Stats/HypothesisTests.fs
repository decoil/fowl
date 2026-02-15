module Fowl.Stats.HypothesisTests

open System
open Fowl
open Fowl.Core.Types
open Fowl.Stats.StudentTDistribution
open Fowl.Stats.ChiSquareDistribution
open Fowl.Stats.FDistribution

/// <summary>Test result containing statistic, p-value, and conclusion.
/// </summary>
type TestResult = {
    Statistic: float
    PValue: float
    Significant: bool  // True if p < alpha
    Alpha: float
}

/// <summary>Helper to create test result.
/// </summary>
let private makeResult (statistic: float) (pValue: float) (alpha: float) : TestResult =
    {
        Statistic = statistic
        PValue = pValue
        Significant = pValue < alpha
        Alpha = alpha
    }

/// <summary>One-sample t-test.
/// Tests if the mean of a sample differs from a known value.
/// </summary>/// <param name="sample">Sample data.</param>/// <param name="popMean">Population mean to test against.</param>/// <param name="alpha">Significance level (default: 0.05).</param>/// <returns>Test result with t-statistic and p-value.</returns>/// <example>
/// <code>
/// let sample = [|5.2; 4.8; 5.5; 5.0; 5.3|]
/// let result = ttestOneSample sample 5.0 0.05
/// // result.Significant indicates if mean differs from 5.0
/// </code>
/// </example>let ttestOneSample (sample: float[]) (popMean: float) (alpha: float) : FowlResult<TestResult> =
    if sample.Length < 2 then
        Error.invalidArgument "Sample must have at least 2 observations"
    elif alpha <= 0.0 || alpha >= 1.0 then
        Error.invalidArgument "Alpha must be in (0, 1)"
    else
        let n = float sample.Length
        let sampleMean = Array.average sample
        let sampleVar = 
            sample 
            |> Array.map (fun x -> (x - sampleMean) ** 2.0)
            |> Array.sum
            |> fun s -> s / (n - 1.0)
        let sampleStd = sqrt sampleVar
        
        // t-statistic = (sample_mean - pop_mean) / (sample_std / sqrt(n))
        let se = sampleStd / sqrt n
        let tStat = (sampleMean - popMean) / se
        let df = n - 1.0
        
        // Two-tailed p-value
        result {
            let! cdfVal = cdf df (abs tStat)
            let pValue = 2.0 * (1.0 - cdfVal)
            return makeResult tStat pValue alpha
        }

/// <summary>Two-sample independent t-test (Welch's t-test).
/// Tests if means of two independent samples differ.
/// Does not assume equal variances.
/// </summary>/// <param name="sample1">First sample.</param>/// <param name="sample2">Second sample.</param>/// <param name="alpha">Significance level.</param>/// <returns>Test result.</returns>let ttestIndependent (sample1: float[]) (sample2: float[]) (alpha: float) : FowlResult<TestResult> =
    if sample1.Length < 2 || sample2.Length < 2 then
        Error.invalidArgument "Both samples must have at least 2 observations"
    elif alpha <= 0.0 || alpha >= 1.0 then
        Error.invalidArgument "Alpha must be in (0, 1)"
    else
        let n1 = float sample1.Length
        let n2 = float sample2.Length
        
        let mean1 = Array.average sample1
        let mean2 = Array.average sample2
        
        let var1 = sample1 |> Array.map (fun x -> (x - mean1) ** 2.0) |> Array.sum |> fun s -> s / (n1 - 1.0)
        let var2 = sample2 |> Array.map (fun x -> (x - mean2) ** 2.0) |> Array.sum |> fun s -> s / (n2 - 1.0)
        
        // Welch's t-test (doesn't assume equal variances)
        let se = sqrt (var1 / n1 + var2 / n2)
        let tStat = (mean1 - mean2) / se
        
        // Welch-Satterthwaite degrees of freedom
        let num = (var1 / n1 + var2 / n2) ** 2.0
        let den = (var1 / n1) ** 2.0 / (n1 - 1.0) + (var2 / n2) ** 2.0 / (n2 - 1.0)
        let df = num / den
        
        result {
            let! cdfVal = cdf df (abs tStat)
            let pValue = 2.0 * (1.0 - cdfVal)
            return makeResult tStat pValue alpha
        }

/// <summary>Chi-square goodness of fit test.
/// Tests if observed frequencies match expected frequencies.
/// </summary>/// <param name="observed">Observed frequencies.</param>/// <param name="expected">Expected frequencies.</param>/// <param name="alpha">Significance level.</param>/// <returns>Test result.</returns>let chiSquareGoodnessOfFit (observed: int[]) (expected: float[]) (alpha: float) : FowlResult<TestResult> =
    if observed.Length <= 0 || observed.Length <> expected.Length then
        Error.invalidArgument "Observed and expected must have same non-zero length"
    elif alpha <= 0.0 || alpha >= 1.0 then
        Error.invalidArgument "Alpha must be in (0, 1)"
    elif Array.exists (fun e -> e <= 0.0) expected then
        Error.invalidArgument "All expected frequencies must be positive"
    else
        // Chi-square statistic = sum((O - E)² / E)
        let chi2Stat = 
            Array.zip observed expected
            |> Array.map (fun (o, e) -> (float o - e) ** 2.0 / e)
            |> Array.sum
        
        let df = float (observed.Length - 1)
        
        result {
            let! cdfVal = ChiSquareDistribution.cdf df chi2Stat
            let pValue = 1.0 - cdfVal
            return makeResult chi2Stat pValue alpha
        }

/// <summary>Chi-square test of independence.
/// Tests if two categorical variables are independent.
/// </summary>/// <param name="contingencyTable">2D array: rows = variable 1, cols = variable 2.</param>/// <param name="alpha">Significance level.</param>/// <returns>Test result.</returns>let chiSquareIndependence (contingencyTable: int[,]) (alpha: float) : FowlResult<TestResult> =
    let rows = contingencyTable.GetLength(0)
    let cols = contingencyTable.GetLength(1)
    
    if rows < 2 || cols < 2 then
        Error.invalidArgument "Contingency table must be at least 2x2"
    elif alpha <= 0.0 || alpha >= 1.0 then
        Error.invalidArgument "Alpha must be in (0, 1)"
    else
        // Calculate row and column totals
        let rowTotals = Array.init rows (fun i ->
            Array.init cols (fun j -> contingencyTable.[i, j])
            |> Array.sum |> float)
        
        let colTotals = Array.init cols (fun j ->
            Array.init rows (fun i -> contingencyTable.[i, j])
            |> Array.sum |> float)
        
        let total = Array.sum rowTotals
        
        // Calculate expected frequencies
        let expected = 
            Array2D.init rows cols (fun i j -> rowTotals.[i] * colTotals.[j] / total)
        
        // Chi-square statistic
        let chi2Stat = 
            seq {
                for i = 0 to rows - 1 do
                    for j = 0 to cols - 1 do
                        let o = float contingencyTable.[i, j]
                        let e = expected.[i, j]
                        yield (o - e) ** 2.0 / e
            }
            |> Seq.sum
        
        let df = float ((rows - 1) * (cols - 1))
        
        result {
            let! cdfVal = ChiSquareDistribution.cdf df chi2Stat
            let pValue = 1.0 - cdfVal
            return makeResult chi2Stat pValue alpha
        }

/// <summary>F-test for comparing variances of two samples.
/// </summary>/// <param name="sample1">First sample.</param>/// <param name="sample2">Second sample.</param>/// <param name="alpha">Significance level.</param>/// <returns>Test result.</returns>let fTestVariances (sample1: float[]) (sample2: float[]) (alpha: float) : FowlResult<TestResult> =
    if sample1.Length < 2 || sample2.Length < 2 then
        Error.invalidArgument "Both samples must have at least 2 observations"
    elif alpha <= 0.0 || alpha >= 1.0 then
        Error.invalidArgument "Alpha must be in (0, 1)"
    else
        let n1 = float sample1.Length
        let n2 = float sample2.Length
        
        let mean1 = Array.average sample1
        let mean2 = Array.average sample2
        
        let var1 = sample1 |> Array.map (fun x -> (x - mean1) ** 2.0) |> Array.sum |> fun s -> s / (n1 - 1.0)
        let var2 = sample2 |> Array.map (fun x -> (x - mean2) ** 2.0) |> Array.sum |> fun s -> s / (n2 - 1.0)
        
        // F-statistic = larger variance / smaller variance
        let fStat = max var1 var2 / min var1 var2
        let df1 = n1 - 1.0
        let df2 = n2 - 1.0
        
        result {
            let! cdfVal = FDistribution.cdf df1 df2 fStat
            let pValue = 2.0 * (1.0 - cdfVal)  // Two-tailed
            return makeResult fStat pValue alpha
        }