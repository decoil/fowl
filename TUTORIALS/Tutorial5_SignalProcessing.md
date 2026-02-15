# Tutorial 5: Signal Processing

Analyze and process signals using FFT, filtering, and spectral analysis with Fowl.

## Overview

Signal processing is fundamental to audio analysis, image processing, and scientific computing. Fowl provides comprehensive FFT operations, filtering tools, and spectral analysis capabilities.

## Learning Objectives

- Compute Fast Fourier Transforms
- Analyze frequency content of signals
- Apply digital filters
- Perform spectral analysis
- Work with real-world signals

## Setup

```fsharp
open Fowl
open Fowl.Core.Types
open Fowl.FFT
open Fowl.FFT.SignalFilters
open System.Numerics

let unwrap = function Ok x -> x | Error e -> failwith e.Message
```

## Fast Fourier Transform (FFT)

### 1D FFT

```fsharp
// Create a simple signal: sum of two sine waves
let sampleRate = 1000.0  // 1 kHz
let duration = 1.0       // 1 second
let t = [|0.0 .. 1.0 / sampleRate .. duration|] |> Array.take 1000

// Signal: 50 Hz + 120 Hz
let signal = Array.map2 (fun t _ - 
    sin (2.0 * System.Math.PI * 50.0 * t) + 
    0.5 * sin (2.0 * System.Math.PI * 120.0 * t)) t t

// Convert to complex
let signalComplex = Array.map (fun x - Complex(x, 0.0)) signal

// Compute FFT
let fftResult = FFT.fft signalComplex |> unwrap

// Get magnitude spectrum
let magnitude = Array.map (fun c - c.Magnitude) fftResult

printfn "First 10 magnitudes: %A" (Array.take 10 magnitude)
```

### Inverse FFT

```fsharp
// Reconstruct signal from FFT
let reconstructed = FFT.ifft fftResult |> unwrap

// Convert back to real (imaginary should be ~0)
let reconstructedReal = Array.map (fun c - c.Real) reconstructed

// Verify reconstruction
let error = 
    signal
    |> Array.zip reconstructedReal
    |> Array.map (fun (orig, rec) - abs (orig - rec))
    |> Array.average

printfn "Reconstruction error: %.6f" error  // Should be very small (~1e-10)
```

### Real FFT (Optimized)

```fsharp
// For real-valued signals, use rfft (faster)
let realSignal = Array.map (fun t - sin (2.0 * System.Math.PI * 50.0 * t)) t
let rfftResult = FFT.rfft realSignal |> unwrap

// Result is half the size (due to symmetry)
printfn "FFT length: %d, rFFT length: %d" fftResult.Length rfftResult.Length
```

### FFT Frequency Bins

```fsharp
// Compute frequency bins
let freqs = FFT.fftfreq signal.Length (1.0 / sampleRate) |> unwrap

// Find dominant frequencies
let threshold = 0.1
let dominantFreqs = 
    magnitude
    |> Array.zip freqs
    |> Array.filter (fun (f, m) - m > threshold)
    |> Array.map fst

printfn "Dominant frequencies: %A Hz" dominantFreqs
```

### 2D FFT (Images)

```fsharp
// Create a simple 2D pattern
let image = Array2D.init 64 64 (fun i j - 
    sin (2.0 * System.Math.PI * 5.0 * float i / 64.0) * 
    cos (2.0 * System.Math.PI * 5.0 * float j / 64.0))

// Convert to complex
let imageComplex = Array2D.map (fun x - Complex(x, 0.0)) image

// 2D FFT
let fft2Result = FFT.fft2 imageComplex |> unwrap

// Magnitude spectrum (shifted for display)
let magnitude2D = 
    fft2Result
    |> Array2D.map (fun c - log (c.Magnitude + 1.0))

printfn "2D FFT shape: %d x %d" magnitude2D.GetLength(0) magnitude2D.GetLength(1)
```

## Spectral Analysis

### Power Spectral Density

```fsharp
// Compute PSD
let psd = FFT.psd fftResult

// Find peak frequencies
let maxIndex = Array.findIndex (fun m - m = Array.max psd) psd
let peakFreq = freqs.[maxIndex]

printfn "Peak frequency: %.1f Hz" peakFreq  // Should be close to 50 Hz

// Integrate PSD to get power in frequency bands
let bandStart = 40.0
let bandEnd = 60.0
let bandIndices = 
    freqs
    |> Array.mapi (fun i f - (i, f))
    |> Array.filter (fun (_, f) - f >= bandStart && f <= bandEnd)
    |> Array.map fst

let bandPower = 
    bandIndices
    |> Array.map (fun i - psd.[i])
    |> Array.sum

printfn "Power in %.0f-%.0f Hz band: %.2f" bandStart bandEnd bandPower
```

### Spectrogram

```fsharp
// Compute spectrogram (time-frequency analysis)
let fftSize = 256
let overlap = 128

let spectro = SignalProcessing.spectrogram signal fftSize overlap |> unwrap

// spectro is [time_windows; frequency_bins]
printfn "Spectrogram shape: %d x %d" spectro.Length spectro.[0].Length

// Find frequencies at specific times
let timeIndex = 50
let spectrumAtTime = spectro.[timeIndex]

// Plot or analyze spectrum
let peakFreqAtTime = 
    spectrumAtTime
    |> Array.mapi (fun i p - (i, p))
    |> Array.maxBy snd
    |> fst
    |> fun i - FFT.fftfreq fftSize (1.0 / sampleRate) |> unwrap
    |> fun f - f.[i]

printfn "Dominant frequency at time %d: %.1f Hz" timeIndex peakFreqAtTime
```

### Windowing

```fsharp
// Apply window function before FFT to reduce spectral leakage
// Hanning window
let hanningWindow = SignalProcessing.hanning fftSize

// Apply window to signal segment
let segment = signal.[0 .. fftSize - 1]
let windowedSegment = Array.map2 (*) segment hanningWindow

// Compute FFT with windowing
let windowedComplex = Array.map (fun x - Complex(x, 0.0)) windowedSegment
let windowedFFT = FFT.fft windowedComplex |> unwrap

// Compare with non-windowed FFT
let nonWindowedFFT = FFT.fft (Array.take fftSize signalComplex) |> unwrap

printfn "Windowed peak magnitude: %.2f" (Array.max (Array.map (fun c - c.Magnitude) windowedFFT))
printfn "Non-windowed peak magnitude: %.2f" (Array.max (Array.map (fun c - c.Magnitude) nonWindowedFFT))
```

## Filtering

### Gaussian Filter (Smoothing)

```fsharp
// Create noisy signal
let cleanSignal = Array.map (fun t - sin (2.0 * System.Math.PI * 5.0 * t)) t
let noise = Array.map (fun _ - 0.2 * (Random.Shared.NextDouble() - 0.5)) cleanSignal
let noisySignal = Array.map2 (+) cleanSignal noise

// Apply Gaussian filter (1D)
let sigma = 2.0
let smoothed = gaussianFilter1D noisySignal sigma |> unwrap

// Compute SNR improvement
let originalSNR = 
    cleanSignal
    |> Array.zip noise
    |> Array.map (fun (s, n) - 20.0 * log10 (abs s / (abs n + 1e-10)))
    |> Array.average

let filteredSNR = 
    cleanSignal
    |> Array.zip (Array.map2 (-) noisySignal smoothed)
    |> Array.map (fun (s, e) - 20.0 * log10 (abs s / (abs e + 1e-10)))
    |> Array.average

printfn "Original SNR: %.2f dB" originalSNR
printfn "Filtered SNR: %.2f dB" filteredSNR
```

### Gaussian Filter (2D)

```fsharp
// Create noisy image
let cleanImage = Array2D.init 64 64 (fun i j - 
    if i > 20 && i < 44 && j > 20 && j < 44 then 1.0 else 0.0)
let noise2D = Array2D.map (fun _ - 0.1 * (Random.Shared.NextDouble() - 0.5)) cleanImage
let noisyImage = Array2D.map2 (+) cleanImage noise2D

// Convert to array of arrays for filtering
let noisyImageRows = Array.init 64 (fun i - Array.init 64 (fun j - noisyImage.[i, j]))

// Apply Gaussian filter (2D)
let smoothedImage = gaussianFilter2D noisyImageRows sigma |> unwrap

// Convert back to 2D array
let smoothedImage2D = Array2D.init 64 64 (fun i j - smoothedImage.[i].[j])

printfn "Clean image mean: %.3f" (cleanImage |> Array2D.cast<float> |> Array.average)
printfn "Noisy image mean: %.3f" (noisyImage |> Array2D.cast<float> |> Array.average)
printfn "Smoothed image mean: %.3f" (smoothedImage2D |> Array2D.cast<float> |> Array.average)
```

### Median Filter (Outlier Removal)

```fsharp
// Create signal with spikes
let signalWithSpikes = 
    noisySignal
    |> Array.mapi (fun i x - 
        if i % 100 = 0 then x + 1.0  // Add spikes every 100 samples
        else x)

// Apply median filter (good for removing spikes)
let kernelSize = 5
let cleanSignal = medianFilter1D signalWithSpikes kernelSize |> unwrap

// Count remaining spikes
let spikesOriginal = 
    signalWithSpikes
    |> Array.filter (fun x - abs x > 1.5)
    |> Array.length

let spikesClean = 
    cleanSignal
    |> Array.filter (fun x - abs x > 1.5)
    |> Array.length

printfn "Spikes before: %d, after: %d" spikesOriginal spikesClean
```

### Moving Average Filter

```fsharp
// Simple moving average
let windowSize = 10
let averaged = movingAverage signalWithSpikes windowSize |> unwrap

// Compute signal smoothness (variance)
let smoothness signal =
    let diffs = 
        signal
    |> Array.pairwise
    |> Array.map (fun (x1, x2) - abs (x2 - x1))
    diffs |> Array.average

printfn "Original smoothness: %.4f" (smoothness signalWithSpikes)
printfn "Averaged smoothness: %.4f" (smoothness averaged)
```

### Custom Convolution Filter

```fsharp
// Apply custom kernel via convolution
let kernel = [|0.25; 0.5; 0.25|]  // Simple smoothing kernel
let convolved = convolve1D signalWithSpikes kernel |> unwrap

// Or use edge detection kernel
let edgeKernel = [|-1.0; 0.0; 1.0|]
let edges = convolve1D noisySignal edgeKernel |> unwrap

printfn "Edge detection max: %.3f" (Array.max (Array.map abs edges))
```

## Practical Examples

### Example 1: Audio Analysis

```fsharp
// Simulate audio signal (speech-like)
let sampleRate = 16000.0  // 16 kHz
let duration = 2.0
let t = [|0.0 .. 1.0 / sampleRate .. duration|] |> Array.take 32000

// Create signal with formants
let audio = Array.map (fun t - 
    0.5 * sin (2.0 * System.Math.PI * 500.0 * t) +
    0.3 * sin (2.0 * System.Math.PI * 1500.0 * t) +
    0.2 * sin (2.0 * System.Math.PI * 2500.0 * t)) t

// Add noise
let noisyAudio = Array.map2 (+) audio (Array.init audio.Length (fun _ - 0.05 * (Random.Shared.NextDouble() - 0.5)))

// Analyze spectrum
let audioFFT = FFT.fft (Array.map (fun x - Complex(x, 0.0)) noisyAudio |> unwrap) |> unwrap
let audioPSD = FFT.psd audioFFT
let audioFreqs = FFT.fftfreq audio.Length (1.0 / sampleRate) |> unwrap

// Find formant frequencies (peaks)
let findPeaks threshold arr =
    arr
    |> Array.mapi (fun i x - (i, x))
    |> Array.filter (fun (i, x) - 
        x > threshold && 
        (i = 0 || x > arr.[i - 1]) && 
        (i = arr.Length - 1 || x >= arr.[i + 1]))
    |> Array.map fst

let peakIndices = findPeaks (Array.average audioPSD * 2.0) audioPSD
let formants = peakIndices |> Array.map (fun i - audioFreqs.[i])

printfn "Detected formants: %A Hz" formants
```

### Example 2: ECG Analysis

```fsharp
// Simulate ECG signal (heartbeat)
let ecg = 
    Array.mapi (fun i t - 
        let beat = System.Math.IEEERemainder(t * 1.2, 1.0)
        if beat >= 0.0 && beat < 0.1 then 0.0  // P wave
        elif beat >= 0.1 && beat < 0.15 then 0.1  // PR segment
        elif beat >= 0.15 && beat < 0.25 then 1.0  // QRS complex
        elif beat >= 0.25 && beat < 0.4 then -0.3  // ST segment
        elif beat >= 0.4 && beat < 0.6 then 0.2  // T wave
        else 0.0
    ) t

// Add baseline wander
let baselineWander = 
    Array.mapi (fun i _ - 0.1 * sin (2.0 * System.Math.PI * 0.5 * float i / ecg.Length)) ecg

let ecgWithBaseline = Array.map2 (+) ecg baselineWander

// Remove baseline wander using high-pass filter
let cutoff = 0.5  // Hz
let ecgFiltered = highPassFilter ecgWithBaseline sampleRate cutoff |> unwrap

// Detect R-peaks
let detectRPeaks signal threshold =
    signal
    |> Array.mapi (fun i x - (i, x))
    |> Array.filter (fun (i, x) - 
        x > threshold && 
        (i = 0 || x >= signal.[i - 1]) && 
        (i = signal.Length - 1 || x > signal.[i + 1]))
    |> Array.map fst

let rPeaks = detectRPeaks ecgFiltered 0.8
let heartRate = float rPeaks.Length * 60.0 / duration  // BPM

printfn "Detected R-peaks: %d" rPeaks.Length
printfn "Heart rate: %.1f BPM" heartRate
```

### Example 3: Image Frequency Analysis

```fsharp
// Create patterned image
let pattern = Array2D.init 128 128 (fun i j - 
    sin (2.0 * System.Math.PI * 10.0 * float i / 128.0) * 
    cos (2.0 * System.Math.PI * 15.0 * float j / 128.0))

// 2D FFT
let patternComplex = Array2D.map (fun x - Complex(x, 0.0)) pattern
let patternFFT = FFT.fft2 patternComplex |> unwrap

// Magnitude spectrum (log scale)
let patternMag = patternFFT |> Array2D.map (fun c - log (c.Magnitude + 1.0))

// Find dominant frequencies
let findDominantFreqs fftData threshold =
    fftData
    |> Array2D.mapi (fun i j x - (i, j, x))
    |> Array.filter (fun (_, _, x) - x > threshold)
    |> Array.sortByDescending (fun (_, _, x) - x)
    |> Array.take 10
    |> Array.map (fun (i, j, _) - (i, j))

let domFreqs = findDominantFreqs patternMag (Array2D.cast<float> patternMag |> Array.average * 2.0)

printfn "Dominant frequency indices: %A" domFreqs
// Should be near (10, 15) and symmetric positions
```

## Exercises

1. Create a signal with 3 frequency components and extract them using FFT
2. Design a band-pass filter to extract specific frequency range
3. Compute and plot the spectrogram of a chirp signal
4. Remove 60 Hz power line interference from simulated EEG
5. Implement a simple peak detector for R-wave detection

## Solutions

```fsharp
// Exercise 1: Multi-frequency signal
let multiFreqSignal = Array.map (fun t - 
    1.0 * sin (2.0 * System.Math.PI * 50.0 * t) +
    0.5 * sin (2.0 * System.Math.PI * 120.0 * t) +
    0.3 * sin (2.0 * System.Math.PI * 200.0 * t)) t

let multiFreqFFT = FFT.fft (Array.map (fun x - Complex(x, 0.0)) multiFreqSignal |> unwrap) |> unwrap
let freqs = FFT.fftfreq multiFreqSignal.Length (1.0 / sampleRate) |> unwrap

let peaks = findPeaks (Array.average (Array.map (fun c - c.Magnitude) multiFreqFFT) * 5.0) (Array.map (fun c - c.Magnitude) multiFreqFFT)
let componentFreqs = peaks |> Array.map (fun i - freqs.[i])

// Exercise 4: 60 Hz interference removal
let eeg = Array.map (fun t - 0.5 * sin (2.0 * System.Math.PI * 10.0 * t) + 0.3 * sin (2.0 * System.Math.PI * 60.0 * t)) t
let eegFFT = FFT.fft (Array.map (fun x - Complex(x, 0.0)) eeg |> unwrap) |> unwrap

// Zero out 60 Hz component
let eegFFTFiltered = 
    eegFFT
    |> Array.mapi (fun i c - 
        let f = freqs.[i]
        if abs (f - 60.0) < 5.0 then Complex(0.0, 0.0)
        else c)

let eegClean = FFT.ifft eegFFTFiltered |> unwrap |> Array.map (fun c - c.Real)
```

## Next Steps

- [Tutorial 6: Optimization](Tutorial6_Optimization.md)
- [User Guide](../USER_GUIDE.md#signal-processing)

---

*Estimated time: 45 minutes*