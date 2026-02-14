namespace Fowl.FFT

open System
open System.Numerics
open Fowl
open Fowl.Core.Types

/// <summary>Fast Fourier Transform module.
/// Implements FFT using Cooley-Tukey algorithm.
/// </summary>module FFT =
    
    /// <summary>Check if n is a power of 2.
    /// </summary>let private isPowerOf2 (n: int) : bool =
        n > 0 && (n & (n - 1)) = 0
    
    /// <summary>Bit-reverse the index.
    /// </summary>let private bitReverse (n: int) (bits: int) : int =
        let mutable result = 0
        let mutable n = n
        for i = 0 to bits - 1 do
            result <- (result << 1) ||| (n & 1)
            n <- n >>> 1
        result
    
    /// <summary>Cooley-Tukey iterative FFT.
    /// Time complexity: O(n log n)
    /// </summary>let fft (input: Complex[]) : FowlResult<Complex[]> =
        let n = input.Length
        
        if not (isPowerOf2 n) then
            Error.invalidArgument "FFT input length must be power of 2"
        else
            let bits = int (System.Math.Log2 (float n))
            let output = Array.copy input
            
            // Bit-reverse permutation
            for i = 0 to n - 1 do
                let j = bitReverse i bits
                if i < j then
                    let temp = output.[i]
                    output.[i] <- output.[j]
                    output.[j] <- temp
            
            // Cooley-Tukey iterations
            let mutable len = 2
            while len <= n do
                let angle = -2.0 * System.Math.PI / float len
                let wlen = Complex(cos angle, sin angle)
                
                for i = 0 to n - 1 do
                    let mutable w = Complex(1.0, 0.0)
                    for j = 0 to len / 2 - 1 do
                        let u = output.[i + j]
                        let v = output.[i + j + len / 2] * w
                        output.[i + j] <- u + v
                        output.[i + j + len / 2] <- u - v
                        w <- w * wlen
                    i <- i + len - 1
                
                len <- len <<< 1
            
            Ok output
    
    /// <summary>Inverse FFT.
    /// </summary>let ifft (input: Complex[]) : FowlResult<Complex[]> =
        let n = input.Length
        
        if not (isPowerOf2 n) then
            Error.invalidArgument "IFFT input length must be power of 2"
        else
            // IFFT = conj(FFT(conj(x))) / n
            let conjugated = input |> Array.map (fun c -> Complex.Conjugate(c))
            
            match fft conjugated with
            | Ok fftResult ->
                let result = 
                    fftResult
                    |> Array.map (fun c -> Complex.Conjugate(c) / Complex(float n, 0.0))
                Ok result
            | Error e -> Error e
    
    /// <summary>Real FFT (optimized for real input).
    /// Returns only positive frequencies.
    /// </summary>let rfft (input: float[]) : FowlResult<Complex[]> =
        let n = input.Length
        
        if not (isPowerOf2 n) then
            Error.invalidArgument "RFFT input length must be power of 2"
        else
            let complexInput = input |> Array.map (fun x -> Complex(x, 0.0))
            
            match fft complexInput with
            | Ok result ->
                // Return only first n/2 + 1 frequencies
                let outputLen = n / 2 + 1
                Ok (result.[0..outputLen-1])
            | Error e -> Error e
    
    /// <summary>Inverse real FFT.
    /// </summary>let irfft (input: Complex[]) (n: int) : FowlResult<float[]> =
        if not (isPowerOf2 n) then
            Error.invalidArgument "IRFFT output length must be power of 2"
        else
            // Reconstruct full spectrum from positive frequencies
            let fullSpectrum = Array.zeroCreate n
            let inputLen = input.Length
            
            // Copy positive frequencies
            for i = 0 to inputLen - 1 do
                fullSpectrum.[i] <- input.[i]
            
            // Reconstruct negative frequencies (conjugate symmetry)
            for i = 1 to n / 2 - 1 do
                fullSpectrum.[n - i] <- Complex.Conjugate(input.[i])
            
            match ifft fullSpectrum with
            | Ok result -> Ok (result |> Array.map (fun c -> c.Real))
            | Error e -> Error e
    
    /// <summary>2D FFT for images.
    /// </summary>let fft2 (input: Complex[,]) : FowlResult<Complex[,]> =
        let rows = input.GetLength(0)
        let cols = input.GetLength(1)
        
        if not (isPowerOf2 rows) || not (isPowerOf2 cols) then
            Error.invalidArgument "FFT2 dimensions must be powers of 2"
        else
            // FFT on rows
            let rowFFT = Array2D.zeroCreate rows cols
            for r = 0 to rows - 1 do
                let row = Array.init cols (fun c -> input.[r, c])
                match fft row with
                | Ok result -
                    for c = 0 to cols - 1 do
                        rowFFT.[r, c] <- result.[c]
                | Error e -> return Error e
            
            // FFT on columns
            let result = Array2D.zeroCreate rows cols
            for c = 0 to cols - 1 do
                let col = Array.init rows (fun r -> rowFFT.[r, c])
                match fft col with
                | Ok colResult -
                    for r = 0 to rows - 1 do
                        result.[r, c] <- colResult.[r]
                | Error e -> return Error e
            
            Ok result
    
    /// <summary>Calculate power spectral density.
    /// </summary>let psd (fftResult: Complex[]) : float[] =
        fftResult |
        Array.map (fun c -> c.Magnitude ** 2.0)
    
    /// <summary>Calculate frequencies for FFT bins.
    /// </summary>let fftfreq (n: int) (d: float) : float[] =
        let df = 1.0 / (float n * d)
        Array.init n (fun i -
            if i <= n / 2 then
                float i * df
            else
                float (i - n) * df)
    
    /// <summary>Apply window function before FFT.
    /// </summary>let applyWindow (window: float[]) (signal: float[]) : float[] =
        if window.Length <> signal.Length then
            invalidArg "window" "Window length must match signal length"
        
        Array.map2 (*) window signal
    
    /// <summary>Hanning window.
    /// </summary>let hanning (n: int) : float[] =
        Array.init n (fun i -
            0.5 - 0.5 * cos (2.0 * System.Math.PI * float i / float (n - 1)))
    
    /// <summary>Hamming window.
    /// </summary>let hamming (n: int) : float[] =
        Array.init n (fun i -
            0.54 - 0.46 * cos (2.0 * System.Math.PI * float i / float (n - 1)))
    
    /// <summary>Blackman window.
    /// </summary>let blackman (n: int) : float[] =
        Array.init n (fun i -
            let alpha = 0.16
            let a0 = (1.0 - alpha) / 2.0
            let a1 = 0.5
            let a2 = alpha / 2.0
            let f = 2.0 * System.Math.PI * float i / float (n - 1)
            a0 - a1 * cos f + a2 * cos (2.0 * f))

/// <summary>Signal processing operations using FFT.
/// </summary>module SignalProcessing =
    
    /// <summary>Convolution using FFT (faster for large kernels).
    /// </summary>let fftConvolve (signal: float[]) (kernel: float[]) : FowlResult<float[]> =
        let n = signal.Length + kernel.Length - 1
        let paddedSize = int (2.0 ** ceil (log (float n) / log 2.0))
        
        // Pad to power of 2
        let paddedSignal = Array.zeroCreate paddedSize
        let paddedKernel = Array.zeroCreate paddedSize
        
        Array.blit signal 0 paddedSignal 0 signal.Length
        Array.blit kernel 0 paddedKernel 0 kernel.Length
        
        let signalComplex = paddedSignal |> Array.map (fun x -> Complex(x, 0.0))
        let kernelComplex = paddedKernel |> Array.map (fun x -> Complex(x, 0.0))
        
        result {
            let! signalFFT = FFT.fft signalComplex
            let! kernelFFT = FFT.fft kernelComplex
            
            // Multiply in frequency domain
            let product = Array.map2 (*) signalFFT kernelFFT
            
            // Inverse FFT
            let! resultComplex = FFT.ifft product
            
            // Take real part and trim
            let result = resultComplex |> Array.map (fun c -> c.Real)
            Ok (result.[0..n-1])
        }
    
    /// <summary>Cross-correlation using FFT.
    /// </summary>let fftCorrelate (x: float[]) (y: float[]) : FowlResult<float[]> =
        // Correlation = convolution with reversed kernel
        let yReversed = Array.rev y
        fftConvolve x yReversed
    
    /// <summary>Spectrogram computation.
    /// </summary>let spectrogram (signal: float[]) (windowSize: int) (hopSize: int)
                     : FowlResult<float[,]> =
        if not (FFT.isPowerOf2 windowSize) then
            Error.invalidArgument "Window size must be power of 2"
        else
            let nFrames = (signal.Length - windowSize) / hopSize + 1
            let nFreqs = windowSize / 2 + 1
            let result = Array2D.zeroCreate nFreqs nFrames
            let window = FFT.hanning windowSize
            
            let mutable success = true
            let mutable error = ""
            
            for frame = 0 to nFrames - 1 do
                let start = frame * hopSize
                let segment = signal.[start..start+windowSize-1]
                let windowed = FFT.applyWindow window segment
                
                match FFT.rfft windowed with
                | Ok fftResult -
                    let psd = FFT.psd fftResult
                    for f = 0 to nFreqs - 1 do
                        result.[f, frame] <- psd.[f]
                | Error e -
                    success <- false
                    error <- e.Message
            
            if success then
                Ok result
            else
                Error.invalidState error
    
    /// <summary>Welch's method for power spectral density estimation.
    /// </summary>let welch (signal: float[]) (windowSize: int) (overlap: int)
              : FowlResult<float[]> =
        if not (FFT.isPowerOf2 windowSize) then
            Error.invalidArgument "Window size must be power of 2"
        else
            let hopSize = windowSize - overlap
            let nFrames = (signal.Length - windowSize) / hopSize + 1
            let nFreqs = windowSize / 2 + 1
            let mutable psdSum = Array.zeroCreate nFreqs
            let window = FFT.hanning windowSize
            
            let mutable count = 0
            let mutable success = true
            let mutable error = ""
            
            for frame = 0 to nFrames - 1 do
                let start = frame * hopSize
                if start + windowSize <= signal.Length then
                    let segment = signal.[start..start+windowSize-1]
                    let windowed = FFT.applyWindow window segment
                    
                    match FFT.rfft windowed with
                    | Ok fftResult -
                        let psd = FFT.psd fftResult
                        for f = 0 to nFreqs - 1 do
                            psdSum.[f] <- psdSum.[f] + psd.[f]
                        count <- count + 1
                    | Error e -
                        success <- false
                        error <- e.Message
            
            if success && count > 0 then
                let avgPsd = psdSum |> Array.map (fun x -> x / float count)
                Ok avgPsd
            elif not success then
                Error.invalidState error
            else
                Error.invalidState "No valid frames processed"

/// <summary>Discrete Cosine Transform (DCT).
/// Used in JPEG compression.
/// </summary>module DCT =
    
    /// <summary>Type 2 DCT.
    /// </summary>let dct2 (input: float[]) : float[] =
        let n = input.Length
        let output = Array.zeroCreate n
        let scale = sqrt (2.0 / float n)
        
        for k = 0 to n - 1 do
            let mutable sum = 0.0
            for i = 0 to n - 1 do
                sum <- sum + input.[i] * cos (System.Math.PI / float n * (float i + 0.5) * float k)
            
            if k = 0 then
                output.[k] <- sum * sqrt (1.0 / float n)
            else
                output.[k] <- sum * scale
        
        output
    
    /// <summary>Inverse Type 2 DCT (Type 3 DCT).
    /// </summary>let idct2 (input: float[]) : float[] =
        let n = input.Length
        let output = Array.zeroCreate n
        let scale = sqrt (2.0 / float n)
        
        for i = 0 to n - 1 do
            let mutable sum = input.[0] * sqrt (1.0 / float n)
            for k = 1 to n - 1 do
                sum <- sum + input.[k] * scale * cos (System.Math.PI / float n * (float i + 0.5) * float k)
            output.[i] <- sum
        
        output
    
    /// <summary>2D DCT for images.
    /// </summary>let dct2_2d (input: float[,]) : float[,] =
        let rows = input.GetLength(0)
        let cols = input.GetLength(1)
        
        // DCT on rows
        let rowDCT = Array2D.zeroCreate rows cols
        for r = 0 to rows - 1 do
            let row = Array.init cols (fun c -> input.[r, c])
            let dctRow = dct2 row
            for c = 0 to cols - 1 do
                rowDCT.[r, c] <- dctRow.[c]
        
        // DCT on columns
        let result = Array2D.zeroCreate rows cols
        for c = 0 to cols - 1 do
            let col = Array.init rows (fun r -> rowDCT.[r, c])
            let dctCol = dct2 col
            for r = 0 to rows - 1 do
                result.[r, c] <- dctCol.[r]
        
        result