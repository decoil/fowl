# Chapter 8: Signal Processing

## 8.1 Introduction

Signal processing is essential for analyzing time-series data, audio, images, and more. Fowl provides comprehensive tools for spectral analysis, filtering, and feature extraction.

## 8.2 Fourier Transform

### Discrete Fourier Transform (DFT)

```fsharp
open Fowl.FFT
open System.Numerics

// Generate a signal with two frequencies
let sampleRate = 1000.0
let duration = 1.0
let t = [| 0.0 .. 1.0/sampleRate .. duration - 1.0/sampleRate |]

// Signal: 50 Hz + 120 Hz sine waves
let signal = 
    t 
    |> Array.map (fun ti -
        sin(2.0 * PI * 50.0 * ti) + 
        0.5 * sin(2.0 * PI * 120.0 * ti))

// Convert to complex
let complexSignal = signal |> Array.map (fun x -> Complex(x, 0.0))

// Compute FFT
let! spectrum = FFT.fft complexSignal

// Compute power spectral density
let psd = FFT.psd spectrum

// Find dominant frequencies
let freqs = FFT.fftfreq t.Length (1.0/sampleRate)
```

### Inverse FFT

```fsharp
// Reconstruct signal from spectrum
let! reconstructed = FFT.ifft spectrum

// Extract real part
let realSignal = reconstructed |> Array.map (fun c -> c.Real)
```

### Real FFT (RFFT)

```fsharp
// For real signals, use RFFT (returns only positive frequencies)
let! rfftResult = FFT.rfft signal

// Inverse
let! reconstructedReal = FFT.irfft rfftResult signal.Length
```

### 2D FFT (Images)

```fsharp
// For image processing
let image = Array2D.init 256 256 (fun i j -
    sin(2.0 * PI * float i / 32.0) * sin(2.0 * PI * float j / 32.0))

let! imageFFT = FFT.fft2 image
let! filtered = FFT.ifft2 imageFFT
```

## 8.3 Windowing

Window functions reduce spectral leakage:

```fsharp
// Hanning window
let hanning = FFT.hanning signal.Length
let windowedSignal = FFT.applyWindow hanning signal

// Hamming window
let hamming = FFT.hamming signal.Length

// Blackman window
let blackman = FFT.blackman signal.Length

// Apply before FFT
let! spectrum = FFT.fft (windowedSignal |> Array.map (fun x -> Complex(x, 0.0)))
```

## 8.4 Spectral Analysis

### Power Spectral Density

```fsharp
open Fowl.FFT.SignalProcessing

// Welch's method (more accurate PSD)
let! psdWelch = welch signal 256 128

// Spectrogram (time-varying spectrum)
let! spec = spectrogram signal 256 64
```

### Frequency Analysis

```fsharp
// Find peaks in spectrum
let findPeaks (spectrum: float[]) (threshold: float) =
    spectrum
    |> Array.mapi (fun i x -> (i, x))
    |> Array.filter (fun (i, x) ->
        i > 0 && i < spectrum.Length - 1 &&
        x > threshold &&
        x > spectrum.[i-1] && x > spectrum.[i+1])
    |> Array.map fst
```

## 8.5 Filtering

### Convolution

```fsharp
// Simple moving average filter
let kernel = Array.create 5 (1.0/5.0)

// Time domain convolution
let! smoothed = SignalProcessing.convolve signal kernel

// FFT-based convolution (faster for large kernels)
let! smoothedFFT = SignalProcessing.fftConvolve signal kernel
```

### FIR Filters

```fsharp
// Design low-pass filter
let designLowPass (cutoff: float) (fs: float) (order: int) =
    let fc = cutoff / fs
    let h = Array.zeroCreate order
    let center = order / 2
    
    for i = 0 to order - 1 do
        let n = i - center
        if n = 0 then
            h.[i] <- 2.0 * fc
        else
            h.[i] <- sin(2.0 * PI * fc * float n) / (PI * float n)
        // Apply window
        h.[i] <- h.[i] * (FFT.hanning order).[i]
    
    h

// Apply filter
let lpFilter = designLowPass 100.0 1000.0 51
let! filtered = SignalProcessing.convolve signal lpFilter
```

### IIR Filters (Butterworth)

```fsharp
// Simple first-order low-pass
let alpha = 0.1
let rec lowPassIIR (input: float[]) (alpha: float) =
    let output = Array.zeroCreate input.Length
    let mutable y = 0.0
    
    for i = 0 to input.Length - 1 do
        y <- alpha * input.[i] + (1.0 - alpha) * y
        output.[i] <- y
    
    output
```

### Band-Pass Filter

```fsharp
let designBandPass (fLow: float) (fHigh: float) (fs: float) (order: int) =
    // Design as difference of two low-pass filters
    let hLow1 = designLowPass fLow fs order
    let hLow2 = designLowPass fHigh fs order
    Array.map2 (-) hLow2 hLow1
```

## 8.6 Discrete Cosine Transform (DCT)

```fsharp
open Fowl.FFT.DCT

// DCT (used in JPEG compression)
let signal = [|1.0 .. 8.0|]
let dctResult = dct2 signal

// Inverse DCT
let reconstructed = idct2 dctResult

// 2D DCT
let image = Array2D.init 8 8 (fun i j -> float (i * 8 + j))
let dctImage = dct2_2d image
```

## 8.7 Feature Extraction

### Spectral Features

```fsharp
// Extract features from signal
let extractSpectralFeatures (signal: float[]) (fs: float) =
    // FFT
    let! spectrum = FFT.fft (signal |> Array.map (fun x -> Complex(x, 0.0)))
    let magnitude = spectrum |> Array.map (fun c -> c.Magnitude)
    let psd = magnitude |> Array.map (fun x -> x * x)
    
    // Spectral centroid ("brightness")
    let freqs = FFT.fftfreq signal.Length (1.0/fs)
    let centroid = 
        Array.map2 (*) freqs psd |> Array.sum 
        / Array.sum psd
    
    // Spectral rolloff
    let totalEnergy = Array.sum psd
    let rolloffIndex = 
        psd
        |> Array.scan (+) 0.0
        |> Array.findIndex (fun x -> x >= 0.85 * totalEnergy)
    let rolloff = freqs.[rolloffIndex]
    
    // Zero crossing rate
    let zcr = 
        signal.[1..]
        |> Array.mapi (fun i x -
            if (x >= 0.0) <> (signal.[i] >= 0.0) then 1 else 0)
        |> Array.sum
        |> float
    
    {| Centroid = centroid; Rolloff = rolloff; ZCR = zcr |}
```

### MFCC (Mel-Frequency Cepstral Coefficients)

```fsharp
// Simplified MFCC extraction
let computeMFCC (signal: float[]) (sampleRate: float) (numCoefficients: int) =
    // Pre-emphasis
    let preEmphasis = 0.97
    let emphasized = 
        signal.[1..]
        |> Array.mapi (fun i x -> x - preEmphasis * signal.[i])
    
    // Framing
    let frameSize = 2048
    let hopSize = 512
    let numFrames = (emphasized.Length - frameSize) / hopSize
    
    // For each frame
    [|
        for i in 0..numFrames-1 do
            let frame = emphasized.[i*hopSize .. i*hopSize+frameSize-1]
            
            // Window
            let windowed = FFT.applyWindow (FFT.hanning frameSize) frame
            
            // FFT
            let! spectrum = FFT.fft (windowed |> Array.map (fun x -> Complex(x, 0.0)))
            let power = spectrum |> Array.map (fun c -> c.Magnitude ** 2.0)
            
            // Mel filterbank (simplified)
            // ...
            
            // Log
            let logMel = power |> Array.map (fun x -> log (x + 1e-10))
            
            // DCT
            let mfcc = dct2 logMel
            
            yield mfcc.[..numCoefficients-1]
    |]
```

## 8.8 Applications

### Audio Denoising

```fsharp
let denoiseAudio (signal: float[]) (noiseProfile: float[]) =
    // Spectral subtraction
    let! spectrum = FFT.fft (signal |> Array.map (fun x -> Complex(x, 0.0)))
    let! noiseSpectrum = FFT.fft (noiseProfile |> Array.map (fun x -> Complex(x, 0.0)))
    
    let cleaned = 
        Array.map2 (fun s n -
            let magnitude = max 0.0 (s.Magnitude - n.Magnitude)
            Complex(magnitude * cos(s.Phase), magnitude * sin(s.Phase)))
            spectrum noiseSpectrum
    
    let! reconstructed = FFT.ifft cleaned
    reconstructed |> Array.map (fun c -> c.Real)
```

### Pitch Detection

```fsharp
let detectPitch (signal: float[]) (sampleRate: float) =
    // Autocorrelation method
    let autocorr = 
        [|
            for lag in 0..signal.Length/2 do
                let sum = 
                    Array.init (signal.Length - lag) (fun i -
                        signal.[i] * signal.[i + lag])
                    |> Array.sum
                yield sum
        |]
    
    // Find first peak
    let mutable peakIdx = 0
    let mutable peakVal = 0.0
    for i = 20..autocorr.Length-1 do
        if autocorr.[i] > peakVal &&
           autocorr.[i] > autocorr.[i-1] &&
           autocorr.[i] > autocorr.[i+1] then
            peakVal <- autocorr.[i]
            peakIdx <- i
    
    sampleRate / float peakIdx  // Frequency in Hz
```

### Vibration Analysis

```fsharp
// Detect bearing faults in machinery
let analyzeVibration (signal: float[]) (fs: float) =
    // Compute envelope
    let analytic = SignalProcessing.hilbert signal
    let envelope = analytic |> Array.map (fun c -> c.Magnitude)
    
    // FFT of envelope
    let! envFFT = FFT.fft (envelope |> Array.map (fun x -> Complex(x, 0.0)))
    let envPSD = FFT.psd envFFT
    
    // Find bearing fault frequencies
    let bpfo = 3.585  // Ball pass frequency outer
    // ... check for peaks at multiples of fault frequencies
    envPSD
```

## 8.9 Exercises

### Exercise 8.1: Filter Design

```fsharp
// Design a notch filter to remove 60 Hz hum
let designNotchFilter (fNotch: float) (fs: float) (bandwidth: float) =
    let fc = fNotch / fs
    let bw = bandwidth / fs
    // Implement IIR notch filter
    // ...
    [||]
```

### Exercise 8.2: STFT

```fsharp
// Short-Time Fourier Transform
let stft (signal: float[]) (windowSize: int) (hopSize: int) =
    let numFrames = (signal.Length - windowSize) / hopSize
    let window = FFT.hanning windowSize
    
    [|
        for i in 0..numFrames-1 do
            let frame = signal.[i*hopSize .. i*hopSize+windowSize-1]
            let windowed = FFT.applyWindow window frame
            let! spectrum = FFT.fft (windowed |> Array.map (fun x -> Complex(x, 0.0)))
            yield spectrum |> Array.map (fun c -> c.Magnitude)
    |]
```

### Exercise 8.3: Cross-Correlation

```fsharp
// Find time delay between signals
let crossCorrelation (x: float[]) (y: float[]) =
    let n = x.Length
    let corr = Array.zeroCreate (2 * n - 1)
    
    for lag in -(n-1)..(n-1) do
        let mutable sum = 0.0
        for i in max 0 -lag .. min (n-1) (n-1-lag) do
            sum <- sum + x.[i] * y.[i + lag]
        corr.[lag + n - 1] <- sum
    
    corr
```

## 8.10 Summary

Key concepts:
- FFT converts time domain to frequency domain
- Windowing reduces spectral leakage
- Filters modify signal characteristics
- DCT is useful for compression
- Spectral features describe signal properties

---

*Next: [Chapter 9: Advanced Signal Processing](chapter09.md)*
