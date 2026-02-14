/// Signal Processing Example
/// FFT, filtering, and spectral analysis

module SignalProcessingExample

open System
open System.Numerics
open Fowl
open Fowl.Core
open Fowl.Stats

/// Discrete Fourier Transform (naive O(n²))
let dft (signal: Complex[]) : Complex[] =
    let n = signal.Length
    let output = Array.zeroCreate n
    
    for k = 0 to n - 1 do
        let mutable sum = Complex.Zero
        for t = 0 to n - 1 do
            let angle = -2.0 * Math.PI * float t * float k / float n
            let w = Complex(cos angle, sin angle)
            sum <- sum + signal.[t] * w
        output.[k] <- sum
    
    output

/// Generate sine wave
let generateSineWave (frequency: float) (sampleRate: float) 
                      (duration: float) : float[] =
    let nSamples = int (sampleRate * duration)
    let omega = 2.0 * Math.PI * frequency / sampleRate
    Array.init nSamples (fun i -> sin (omega * float i))

/// Generate square wave (sum of odd harmonics)
let generateSquareWave (fundamental: float) (sampleRate: float)
                       (duration: float) : float[] =
    let sine = generateSineWave fundamental sampleRate duration
    let harmonic3 = generateSineWave (fundamental * 3.0) sampleRate duration
    let harmonic5 = generateSineWave (fundamental * 5.0) sampleRate duration
    
    Array.map3 (fun s h3 h5 -
        s + h3 / 3.0 + h5 / 5.0) sine harmonic3 harmonic5

/// Simple moving average filter
let movingAverageFilter (signal: float[]) (windowSize: int) : float[] =
    let n = signal.Length
    Array.init (n - windowSize + 1) (fun i -
        signal.[i..i+windowSize-1] |> Array.average)

/// Exponential moving average filter
let exponentialFilter (signal: float[]) (alpha: float) : float[] =
    let n = signal.Length
    let result = Array.zeroCreate n
    result.[0] <- signal.[0]
    
    for i = 1 to n - 1 do
        result.[i] <- alpha * signal.[i] + (1.0 - alpha) * result.[i-1]
    
    result

/// Calculate power spectral density
let calculatePSD (fftResult: Complex[]) : float[] =
    fftResult |
    Array.map (fun c -> c.Magnitude ** 2.0)

/// Find dominant frequencies
let findPeaks (psd: float[]) (sampleRate: float) (nPeaks: int) : (float * float)[] =
    let n = psd.Length
    let freqs = Array.init (n/2) (fun i -
        float i * sampleRate / float n)
    
    // Simple peak detection
    psd.[1..n/2-2]
    |> Array.mapi (fun i p -> (freqs.[i+1], p))
    |> Array.filter (fun (f, p) -
        let idx = int (f / sampleRate * float n)
        p > psd.[idx-1] && p > psd.[idx+1])
    |> Array.sortByDescending snd
    |> Array.take (min nPeaks (n/2 - 1))

/// Run signal processing example
let runSignalProcessing() : unit =
    printfn "=== Signal Processing Example ==="
    printfn ""
    
    let sampleRate = 1000.0  // Hz
    let duration = 1.0       // second
    
    // Generate signals
    printfn "Generating signals..."
    let sine50Hz = generateSineWave 50.0 sampleRate duration
    let sine120Hz = generateSineWave 120.0 sampleRate duration
    let squareWave = generateSquareWave 50.0 sampleRate duration
    
    // Combine signals with noise
    let rng = Random()
    let noisySignal = 
        Array.init sine50Hz.Length (fun i -
            sine50Hz.[i] + 0.5 * sine120Hz.[i] + 
            rng.NextDouble() * 0.2 - 0.1)
    
    printfn "  50 Hz sine wave"
    printfn "  120 Hz sine wave (half amplitude)"
    printfn "  Gaussian noise"
    printfn ""
    
    // Convert to complex and apply DFT
    printfn "Applying DFT (Discrete Fourier Transform)..."
    let complexSignal = noisySignal |> Array.map (fun x -> Complex(x, 0.0))
    let fft = dft complexSignal
    let psd = calculatePSD fft
    
    // Find dominant frequencies
    let peaks = findPeaks psd sampleRate 5
    
    printfn ""
    printfn "Dominant Frequencies:"
    peaks |> Array.iteri (fun i (freq, power) -
        printfn "  %d. %.1f Hz (power: %.2f)" (i+1) freq power)
    printfn ""
    
    // Apply filters
    printfn "Applying Filters:"
    
    // Moving average
    let maFiltered = movingAverageFilter noisySignal 10
    let! maStd = Descriptive.std maFiltered
    let! origStd = Descriptive.std noisySignal
    printfn "  Moving Average (window=10):"
    printfn "    Std reduction: %.2f → %.2f (%.1f%% reduction)"
            origStd maStd ((1.0 - maStd/origStd) * 100.0)
    
    // Exponential filter
    let expFiltered = exponentialFilter noisySignal 0.3
    let! expStd = Descriptive.std expFiltered
    printfn "  Exponential Filter (α=0.3):"
    printfn "    Std reduction: %.2f → %.2f (%.1f%% reduction)"
            origStd expStd ((1.0 - expStd/origStd) * 100.0)
    printfn ""
    
    // Spectral analysis of square wave
    printfn "Square Wave Spectral Analysis:"
    let complexSquare = squareWave |> Array.map (fun x -> Complex(x, 0.0))
    let fftSquare = dft complexSquare
    let psdSquare = calculatePSD fftSquare
    let squarePeaks = findPeaks psdSquare sampleRate 5
    
    printfn "  Fundamental frequency: 50 Hz"
    printfn "  Expected harmonics: 150 Hz, 250 Hz, 350 Hz..."
    printfn "  Detected peaks:"
    squarePeaks |> Array.iter (fun (freq, power) -
        printfn "    %.1f Hz (power: %.2f)" freq power)
    printfn ""
    
    // Note about FFT performance
    printfn "Note: This example uses naive O(n²) DFT."
    printfn "For production use, implement Cooley-Tukey FFT (O(n log n))"
    printfn "or integrate with FFTW library."
    printfn ""
    
    printfn "=== Signal Processing Complete ==="

[<EntryPoint>]
let main argv =
    runSignalProcessing()
    0