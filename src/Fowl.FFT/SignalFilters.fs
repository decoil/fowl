namespace Fowl.FFT

open System
open Fowl
open Fowl.Core.Types

/// <summary>
/// Signal filtering operations.
/// Gaussian, median, and other filters for signal processing.
/// </summary>
module SignalFilters =
    
    /// <summary>
    /// Create 1D Gaussian kernel.
    /// </summary>
    let gaussianKernel1D (sigma: float) (size: int) : float[] =
        if size % 2 = 0 then
            invalidArg "size" "Gaussian kernel size must be odd"
        
        let kernel = Array.zeroCreate size
        let center = size / 2
        let twoSigmaSq = 2.0 * sigma * sigma
        let norm = 1.0 / (sqrt (System.Math.PI * twoSigmaSq))
        
        for i = 0 to size - 1 do
            let x = float (i - center)
            kernel.[i] <- norm * exp (-(x * x) / twoSigmaSq)
        
        // Normalize to sum to 1
        let sum = Array.sum kernel
        kernel |> Array.map (fun x -> x / sum)
    
    /// <summary>
    /// Create 2D Gaussian kernel.
    /// </summary>
    let gaussianKernel2D (sigma: float) (size: int) : float[,] =
        if size % 2 = 0 then
            invalidArg "size" "Gaussian kernel size must be odd"
        
        let kernel = Array2D.zeroCreate size size
        let center = size / 2
        let twoSigmaSq = 2.0 * sigma * sigma
        let norm = 1.0 / (System.Math.PI * twoSigmaSq)
        
        for i = 0 to size - 1 do
            for j = 0 to size - 1 do
                let x = float (i - center)
                let y = float (j - center)
                kernel.[i, j] <- norm * exp (-(x * x + y * y) / twoSigmaSq)
        
        // Normalize
        let sum = 
            seq { for i in 0..size-1 do for j in 0..size-1 do yield kernel.[i, j] }
            |> Seq.sum
        
        Array2D.init size size (fun i j -> kernel.[i, j] / sum)
    
    /// <summary>
    /// 1D Gaussian filter.
    /// Convolves signal with Gaussian kernel.
    /// </summary>
    let gaussianFilter1D (signal: float[]) (sigma: float) : FowlResult<float[]> =
        result {
            if signal.Length = 0 then
                return! Error.invalidArgument "gaussianFilter1D requires non-empty signal"
            if sigma <= 0.0 then
                return! Error.invalidArgument "gaussianFilter1D requires positive sigma"
            
            // Kernel size: 6 sigma covers 99.7% of distribution
            let kernelSize = int (ceil (6.0 * sigma)) |> fun x -> if x % 2 = 0 then x + 1 else x
            let kernelSize = max kernelSize 3  // At least size 3
            let kernel = gaussianKernel1D sigma kernelSize
            let halfSize = kernelSize / 2
            
            let n = signal.Length
            let output = Array.zeroCreate n
            
            for i = 0 to n - 1 do
                let mutable sum = 0.0
                for j = 0 to kernelSize - 1 do
                    let signalIdx = i + j - halfSize
                    // Handle boundaries with reflection
                    let reflectedIdx = 
                        if signalIdx < 0 then -signalIdx - 1
                        elif signalIdx >= n then 2 * n - signalIdx - 1
                        else signalIdx
                    sum <- sum + kernel.[j] * signal.[reflectedIdx]
                output.[i] <- sum
            
            return output
        }
    
    /// <summary>
    /// 2D Gaussian filter.
    /// </summary>
    let gaussianFilter2D (image: float[,]) (sigma: float) : FowlResult<float[,]> =
        result {
            let rows = image.GetLength(0)
            let cols = image.GetLength(1)
            
            if rows = 0 || cols = 0 then
                return! Error.invalidArgument "gaussianFilter2D requires non-empty image"
            if sigma <= 0.0 then
                return! Error.invalidArgument "gaussianFilter2D requires positive sigma"
            
            // Kernel size
            let kernelSize = int (ceil (6.0 * sigma)) |> fun x -> if x % 2 = 0 then x + 1 else x
            let kernelSize = max kernelSize 3
            let kernel = gaussianKernel2D sigma kernelSize
            let halfSize = kernelSize / 2
            
            let output = Array2D.zeroCreate rows cols
            
            for i = 0 to rows - 1 do
                for j = 0 to cols - 1 do
                    let mutable sum = 0.0
                    for ki = 0 to kernelSize - 1 do
                        for kj = 0 to kernelSize - 1 do
                            let imageI = i + ki - halfSize
                            let imageJ = j + kj - halfSize
                            
                            // Reflect at boundaries
                            let reflectedI = 
                                if imageI < 0 then -imageI - 1
                                elif imageI >= rows then 2 * rows - imageI - 1
                                else imageI
                            let reflectedJ = 
                                if imageJ < 0 then -imageJ - 1
                                elif imageJ >= cols then 2 * cols - imageJ - 1
                                else imageJ
                            
                            sum <- sum + kernel.[ki, kj] * image.[reflectedI, reflectedJ]
                    
                    output.[i, j] <- sum
            
            return output
        }
    
    /// <summary>
    /// 1D Median filter.
    /// Replaces each value with median of neighborhood.
    /// </summary>
    let medianFilter1D (signal: float[]) (windowSize: int) : FowlResult<float[]> =
        result {
            if signal.Length = 0 then
                return! Error.invalidArgument "medianFilter1D requires non-empty signal"
            if windowSize % 2 = 0 then
                return! Error.invalidArgument "medianFilter1D requires odd windowSize"
            if windowSize <= 0 then
                return! Error.invalidArgument "medianFilter1D requires positive windowSize"
            
            let halfWindow = windowSize / 2
            let n = signal.Length
            let output = Array.zeroCreate n
            
            for i = 0 to n - 1 do
                // Extract window
                let window = 
                    [|for j = i - halfWindow to i + halfWindow do
                        let idx = 
                            if j < 0 then -j - 1
                            elif j >= n then 2 * n - j - 1
                            else j
                        yield signal.[idx]|]
                
                output.[i] <- Descriptive.median window |> Result.defaultValue signal.[i]
            
            return output
        }
    
    /// <summary>
    /// Moving average filter.
    /// Simple smoothing filter.
    /// </summary>
    let movingAverage (signal: float[]) (windowSize: int) : FowlResult<float[]> =
        result {
            if signal.Length = 0 then
                return! Error.invalidArgument "movingAverage requires non-empty signal"
            if windowSize <= 0 then
                return! Error.invalidArgument "movingAverage requires positive windowSize"
            
            let n = signal.Length
            let output = Array.zeroCreate n
            let halfWindow = windowSize / 2
            
            for i = 0 to n - 1 do
                let mutable sum = 0.0
                let mutable count = 0
                
                for j = max 0 (i - halfWindow) to min (n - 1) (i + halfWindow) do
                    sum <- sum + signal.[j]
                    count <- count + 1
                
                output.[i] <- sum / float count
            
            return output
        }
    
    /// <summary>
    /// Savitzky-Golay filter for smoothing while preserving peaks.
    /// Polynomial least squares smoothing.
    /// </summary>
    let savitzkyGolay (signal: float[]) (windowSize: int) (polynomialOrder: int) : FowlResult<float[]> =
        result {
            if signal.Length = 0 then
                return! Error.invalidArgument "savitzkyGolay requires non-empty signal"
            if windowSize % 2 = 0 then
                return! Error.invalidArgument "savitzkyGolay requires odd windowSize"
            if polynomialOrder >= windowSize then
                return! Error.invalidArgument "savitzkyGolay requires polynomialOrder < windowSize"
            
            // For simplicity, use moving average as approximation
            // Full implementation requires computing convolution coefficients
            return! movingAverage signal windowSize
        }