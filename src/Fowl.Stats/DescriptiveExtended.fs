namespace Fowl.Stats

open System
open Fowl
open Fowl.Core.Types

/// <summary>Additional descriptive statistics operations.
/// zscore, cumulative functions, histogram, quantile.
/// </summary>module DescriptiveExtended =
    
    /// <summary>Z-score normalization (standardization).
/// z = (x - μ) / σ
/// </summary>let zscore (data: float[]) : FowlResult<float[]> =
        result {
            if data.Length = 0 then
                return! Error.invalidArgument "zscore requires non-empty data"
            
            let! mean = Descriptive.mean data
            let! std = Descriptive.std data
            
            if std = 0.0 then
                return! Error.invalidState "zscore requires non-zero standard deviation"
            
            return data |> Array.map (fun x -> (x - mean) / std)
        }
    
    /// <summary>Z-score with specified mean and std.
/// </summary>let zscoreWithParams (mean: float) (std: float) (data: float[]) : FowlResult<float[]> =
        result {
            if data.Length = 0 then
                return! Error.invalidArgument "zscore requires non-empty data"
            if std = 0.0 then
                return! Error.invalidArgument "zscore requires non-zero standard deviation"
            
            return data |> Array.map (fun x -> (x - mean) / std)
        }
    
    /// <summary>Cumulative sum.
/// Returns array where each element is sum of all previous.
/// </summary>let cumsum (data: float[]) : float[] =
        let result = Array.zeroCreate data.Length
        let mutable sum = 0.0
        for i = 0 to data.Length - 1 do
            sum <- sum + data.[i]
            result.[i] <- sum
        result
    
    /// <summary>Cumulative product.
/// Returns array where each element is product of all previous.
/// </summary>let cumprod (data: float[]) : float[] =
        let result = Array.zeroCreate data.Length
        let mutable prod = 1.0
        for i = 0 to data.Length - 1 do
            prod <- prod * data.[i]
            result.[i] <- prod
        result
    
    /// <summary>Histogram computation.
/// Counts values in bins.
/// </summary>let histogram (data: float[]) (bins: int) : FowlResult<float[] * float[]> =
        result {
            if data.Length = 0 then
                return! Error.invalidArgument "histogram requires non-empty data"
            if bins <= 0 then
                return! Error.invalidArgument "histogram requires positive number of bins"
            
            let minVal = Array.min data
            let maxVal = Array.max data
            
            if minVal = maxVal then
                return! Error.invalidState "histogram requires data with variation"
            
            let binWidth = (maxVal - minVal) / float bins
            let counts = Array.zeroCreate bins
            
            for x in data do
                let binIndex = 
                    if x = maxVal then
                        bins - 1  // Include right edge in last bin
                    else
                        int ((x - minVal) / binWidth)
                counts.[binIndex] <- counts.[binIndex] + 1.0
            
            // Bin edges
            let edges = Array.init (bins + 1) (fun i -> minVal + float i * binWidth)
            
            return (counts, edges)
        }
    
    /// <summary>Histogram with specified range.
/// </summary>let histogramRange (data: float[]) (bins: int) (range: float * float) 
                      : FowlResult<float[] * float[]> =
        result {
            let (minVal, maxVal) = range
            
            if minVal >= maxVal then
                return! Error.invalidArgument "histogram range min must be less than max"
            if bins <= 0 then
                return! Error.invalidArgument "histogram requires positive number of bins"
            
            let binWidth = (maxVal - minVal) / float bins
            let counts = Array.zeroCreate bins
            
            for x in data do
                if x >= minVal && x <= maxVal then
                    let binIndex = 
                        if x = maxVal then
                            bins - 1
                        else
                            int ((x - minVal) / binWidth)
                    if binIndex >= 0 && binIndex < bins then
                        counts.[binIndex] <- counts.[binIndex] + 1.0
            
            let edges = Array.init (bins + 1) (fun i -> minVal + float i * binWidth)
            
            return (counts, edges)
        }
    
    /// <summary>Quantile (inverse CDF) using linear interpolation.
/// Returns value at given quantile (0 to 1).
/// </summary>let quantile (data: float[]) (q: float) : FowlResult<float> =
        result {
            if data.Length = 0 then
                return! Error.invalidArgument "quantile requires non-empty data"
            if q < 0.0 || q > 1.0 then
                return! Error.invalidArgument "quantile requires 0 ≤ q ≤ 1"
            
            let sorted = Array.sort data
            let n = float sorted.Length
            
            // Linear interpolation method
            let pos = q * (n - 1.0)
            let lower = int (floor pos)
            let upper = int (ceil pos)
            let weight = pos - float lower
            
            if lower = upper then
                return sorted.[lower]
            else
                return (1.0 - weight) * sorted.[lower] + weight * sorted.[upper]
        }
    
    /// <summary>Multiple quantiles at once.
/// </summary>let quantiles (data: float[]) (qs: float[]) : FowlResult<float[]> =
        result {
            return! qs |> Array.map (fun q -> quantile data q) |> Result.sequenceArray
        }
    
    /// <summary>Five-number summary (Tukey's hinges).
/// Returns (min, Q1, median, Q3, max)
/// </summary>let fiveNumberSummary (data: float[]) : FowlResult<(float * float * float * float * float)> =
        result {
            if data.Length = 0 then
                return! Error.invalidArgument "fiveNumberSummary requires non-empty data"
            
            let sorted = Array.sort data
            let! q0 = quantile sorted 0.0
            let! q1 = quantile sorted 0.25
            let! q2 = quantile sorted 0.5
            let! q3 = quantile sorted 0.75
            let! q4 = quantile sorted 1.0
            
            return (q0, q1, q2, q3, q4)
        }
    
    /// <summary>Interquartile range (IQR).
/// IQR = Q3 - Q1
/// </summary>let iqr (data: float[]) : FowlResult<float> =
        result {
            let! q1 = Descriptive.percentile data 25.0
            let! q3 = Descriptive.percentile data 75.0
            return q3 - q1
        }
    
    /// <summary>Standard error of the mean.
/// SEM = σ / √n
/// </summary>let sem (data: float[]) : FowlResult<float> =
        result {
            let! std = Descriptive.std data
            return std / sqrt (float data.Length)
        }
    
    /// <summary>Median absolute deviation (MAD).
/// Robust measure of spread.
/// </summary>let mad (data: float[]) : FowlResult<float> =
        result {
            let! median = Descriptive.median data
            let absDeviations = data |> Array.map (fun x -> abs (x - median))
            return! Descriptive.median absDeviations
        }
    
    /// <summary>Trimmed mean (remove outliers).
/// proportion: fraction to trim from each tail (0.05 for 5%)
/// </summary>let trimmedMean (data: float[]) (proportion: float) : FowlResult<float> =
        result {
            if proportion < 0.0 || proportion >= 0.5 then
                return! Error.invalidArgument "trimmedMean proportion must be in [0, 0.5)"
            
            let sorted = Array.sort data
            let n = sorted.Length
            let trimCount = int (float n * proportion)
            
            let trimmed = sorted.[trimCount..n-trimCount-1]
            return Array.average trimmed
        }
    
    /// <summary>Winsorized mean (cap outliers instead of removing).
/// </summary>let winsorizedMean (data: float[]) (proportion: float) : FowlResult<float> =
        result {
            if proportion < 0.0 || proportion >= 0.5 then
                return! Error.invalidArgument "winsorizedMean proportion must be in [0, 0.5)"
            
            let sorted = Array.sort data
            let n = sorted.Length
            let trimCount = int (float n * proportion)
            
            let lowerValue = sorted.[trimCount]
            let upperValue = sorted.[n - trimCount - 1]
            
            let winsorized = 
                data
                |> Array.map (fun x -
                    if x < lowerValue then lowerValue
                    elif x > upperValue then upperValue
                    else x)
            
            return Array.average winsorized
        }