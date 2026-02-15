# Chapter 9: Advanced Signal Processing

## 9.1 Time-Frequency Analysis

### Short-Time Fourier Transform (STFT)

```fsharp
open Fowl.FFT

// STFT: Analyze how frequency content changes over time
let stft (signal: float[]) (windowSize: int) (hopSize: int) =
    let numFrames = (signal.Length - windowSize) / hopSize + 1
    let window = FFT.hanning windowSize
    
    [|
        for frame = 0 to numFrames - 1 do
            let start = frame * hopSize
            let segment = signal.[start .. start + windowSize - 1]
            let windowed = FFT.applyWindow window segment
            
            let! spectrum = FFT.fft (windowed |> Array.map (fun x -> Complex(x, 0.0)))
            yield spectrum |> Array.map (fun c -> c.Magnitude)
    |]

// Usage
let audioSignal = [||]  // Load audio
let spectrogram = stft audioSignal 2048 512

// Visualize: Time on x-axis, Frequency on y-axis, Magnitude as color
```

### Continuous Wavelet Transform (CWT)

```fsharp
// Wavelet analysis for varying frequency resolution
let morletWavelet (t: float) (scale: float) : Complex =
    let sigma = 1.0
    let norm = 1.0 / sqrt(scale)
    let exponent = -t*t / (2.0 * sigma*sigma * scale*scale)
    let oscillation = cos(2.0 * PI * t / scale)
    Complex(norm * exp(exponent) * oscillation, 0.0)

let cwt (signal: float[]) (scales: float[]) : Complex[][] =
    [|
        for scale in scales do
            let wavelet = Array.init signal.Length (fun i -
                morletWavelet (float i) scale)
            
            // Convolution with wavelet
            let! result = SignalProcessing.fftConvolve signal 
                (wavelet |> Array.map (fun c -> c.Real))
            yield result |> Array.map (fun x -> Complex(x, 0.0))
    |]
```

## 9.2 Digital Filter Design

### FIR Filter Design

```fsharp
// Design low-pass FIR filter using window method
let designLowPassFIR (cutoff: float) (fs: float) (order: int) : float[] =
    let fc = cutoff / fs
    let h = Array.zeroCreate order
    let center = order / 2
    
    // Ideal sinc filter
    for i = 0 to order - 1 do
        let n = i - center
        h.[i] <-
            if n = 0 then
                2.0 * fc
            else
                sin(2.0 * PI * fc * float n) / (PI * float n)
    
    // Apply window
    let window = FFT.hamming order
    Array.map2 (*) h window

// Design high-pass, band-pass, band-stop
let designHighPassFIR (cutoff: float) (fs: float) (order: int) : float[] =
    let lp = designLowPassFIR (fs/2.0 - cutoff) fs order
    // Spectral inversion
    lp |> Array.mapi (fun i h -> if i = order/2 then 1.0 - h else -h)

let designBandPassFIR (fLow: float) (fHigh: float) (fs: float) (order: int) : float[] =
    let lp1 = designLowPassFIR fHigh fs order
    let lp2 = designLowPassFIR fLow fs order
    Array.map2 (-) lp1 lp2
```

### IIR Filter Design (Butterworth)

```fsharp
// Butterworth filter design
let butterworthCoefficients (order: int) (fc: float) (fs: float) =
    let wc = 2.0 * PI * fc / fs
    
    // Compute poles of analog prototype
    let poles = [|
        for k = 0 to order - 1 do
            let angle = PI * float (2*k + order - 1) / float (2*order)
            yield Complex(-sin(angle), cos(angle))
    |]
    
    // Bilinear transform to digital
    // ... transformation code ...
    
    // Return coefficients (b, a)
    [||], [||]

// Apply IIR filter
let applyIIR (b: float[]) (a: float[]) (input: float[]) : float[] =
    let n = input.Length
    let output = Array.zeroCreate n
    let na = a.Length
    let nb = b.Length
    
    for i = 0 to n - 1 do
        let mutable sum = 0.0
        
        // Feedforward
        for j = 0 to min (i+1) (nb-1) do
            sum <- sum + b.[j] * input.[i-j]
        
        // Feedback
        for j = 1 to min i (na-1) do
            sum <- sum - a.[j] * output.[i-j]
        
        output.[i] <- sum / a.[0]
    
    output
```

## 9.3 Adaptive Filtering

### Least Mean Squares (LMS)

```fsharp
// Adaptive noise cancellation
let lmsFilter (input: float[]) (desired: float[]) 
              (filterOrder: int) (mu: float) =
    let n = input.Length
    let weights = Array.zeroCreate filterOrder
    let output = Array.zeroCreate n
    let error = Array.zeroCreate n
    
    for i = filterOrder to n - 1 do
        // Extract window
        let x = Array.init filterOrder (fun j -
            input.[i-j])
        
        // Filter output
        let y = Array.map2 (*) weights x |> Array.sum
        output.[i] <- y
        
        // Error
        let e = desired.[i] - y
        error.[i] <- e
        
        // Weight update
        for j = 0 to filterOrder - 1 do
            weights.[j] <- weights.[j] + mu * e * x.[j]
    
    output, error, weights

// Usage: Remove 60 Hz hum from audio
let notchAdaptive (signal: float[]) (humFreq: float) (fs: float) =
    // Generate reference 60 Hz signal
    let reference = 
        Array.init signal.Length (fun i -
            sin(2.0 * PI * humFreq * float i / fs))
    
    let filtered, error, _ = lmsFilter reference signal 32 0.01
    error  // Cleaned signal
```

### Recursive Least Squares (RLS)

```fsharp
// RLS: Faster convergence than LMS
let rlsFilter (input: float[]) (desired: float[]) 
              (filterOrder: int) (lambda: float) (delta: float) =
    let n = input.Length
    let weights = Array.zeroCreate filterOrder
    let P = Array2D.init filterOrder filterOrder 
        (fun i j -> if i = j then delta else 0.0)
    
    let output = Array.zeroCreate n
    
    for i = filterOrder to n - 1 do
        let x = Array.init filterOrder (fun j -
            input.[i-j])
        
        // Gain vector
        let Px = Array.init filterOrder (fun i -
            Array.init filterOrder (fun j -
                P.[i,j] * x.[j]) |> Array.sum)
        
        let g = Array.map (fun p -> p / (lambda + Array.map2 (*) x Px |> Array.sum)) Px
        
        // Output and error
        let y = Array.map2 (*) weights x |> Array.sum
        output.[i] <- y
        let e = desired.[i] - y
        
        // Weight update
        for j = 0 to filterOrder - 1 do
            weights.[j] <- weights.[j] + g.[j] * e
        
        // Update P matrix
        // P = (P - g * x' * P) / lambda
        // ...
    
    output, weights
```

## 9.4 Spectral Estimation

### Periodogram

```fsharp
// Basic power spectral density estimate
let periodogram (signal: float[]) : float[] =
    let n = signal.Length
    let window = FFT.hanning n
    let windowed = FFT.applyWindow window signal
    
    let! spectrum = FFT.fft (windowed |> Array.map (fun x -> Complex(x, 0.0)))
    
    spectrum
    |> Array.map (fun c -
        let mag = c.Magnitude
        mag * mag / float n)
```

### Welch's Method

```fsharp
// Averaged periodogram for smoother PSD
let welchPSD (signal: float[]) (windowSize: int) (overlap: int) : float[] =
    let hopSize = windowSize - overlap
    let numFrames = (signal.Length - windowSize) / hopSize + 1
    let window = FFT.hanning windowSize
    let windowPower = window |> Array.sumBy (fun w -> w * w)
    
    let psdSum = Array.zeroCreate (windowSize / 2 + 1)
    
    for frame = 0 to numFrames - 1 do
        let start = frame * hopSize
        let segment = signal.[start .. start + windowSize - 1]
        let windowed = FFT.applyWindow window segment
        
        let! spectrum = FFT.rfft windowed
        let psd = spectrum |> Array.map (fun c -
            let mag = c.Magnitude
            mag * mag / windowPower)
        
        for i = 0 to psd.Length - 1 do
            psdSum.[i] <- psdSum.[i] + psd.[i]
    
    psdSum |> Array.map (fun x -> x / float numFrames)
```

### Multi-taper Method

```fsharp
// Thomson's multi-taper method for better spectral estimates
let multiTaperPSD (signal: float[]) (numTapers: int) : float[] =
    // Generate Slepian tapers
    let tapers = generateSlepianTapers signal.Length numTapers
    
    let psdEstimates =
        tapers
        |> Array.map (fun taper -
            let tapered = Array.map2 (*) signal taper
            let! spectrum = FFT.rfft tapered
            spectrum |> Array.map (fun c -> c.Magnitude ** 2.0))
    
    // Average across tapers
    let nFreqs = psdEstimates.[0].Length
    [|
        for i = 0 to nFreqs - 1 do
            yield psdEstimates |> Array.averageBy (fun psd -> psd.[i])
    |]
```

## 9.5 Audio Processing

### Pitch Detection

```fsharp
// YIN algorithm for pitch detection
let yinPitch (signal: float[]) (sampleRate: float) : float option =
    let tauMax = int (sampleRate / 50.0)  // Min freq 50 Hz
    
    // Step 1: Difference function
    let difference = Array.zeroCreate tauMax
    for tau = 1 to tauMax - 1 do
        let mutable sum = 0.0
        for j = 0 to signal.Length - tau - 1 do
            let diff = signal.[j] - signal.[j + tau]
            sum <- sum + diff * diff
        difference.[tau] <- sum
    
    // Step 2: Cumulative mean normalized difference
    let cmnd = Array.zeroCreate tauMax
    cmnd.[0] <- 1.0
    let mutable runningSum = 0.0
    for tau = 1 to tauMax - 1 do
        runningSum <- runningSum + difference.[tau]
        cmnd.[tau] <- difference.[tau] * float tau / runningSum
    
    // Step 3: Find minimum below threshold
    let threshold = 0.1
    let mutable pitchPeriod = None
    
    for tau = 2 to tauMax - 1 do
        if cmnd.[tau] < threshold &&
           cmnd.[tau] < cmnd.[tau-1] &&
           cmnd.[tau] < cmnd.[tau+1] then
            pitchPeriod <- Some tau
            break
    
    match pitchPeriod with
    | Some tau -> Some (sampleRate / float tau)
    | None -> None
```

### Audio Effects

```fsharp
// Reverb using comb and all-pass filters
let reverb (signal: float[]) (roomSize: float) (damping: float) =
    // Comb filters
    let comb1 = combFilter signal 1556 damping
    let comb2 = combFilter signal 1617 damping
    let comb3 = combFilter signal 1491 damping
    let comb4 = combFilter signal 1422 damping
    
    let combSum = [comb1; comb2; comb3; comb4] |> Array.map2 (+) 
                  |> Array.map (fun x -> x / 4.0)
    
    // All-pass filters
    let ap1 = allPassFilter combSum 225 0.5
    let ap2 = allPassFilter ap1 341 0.5
    let ap3 = allPassFilter ap2 441 0.5
    
    // Mix dry and wet
    Array.map2 (fun dry wet -> dry * 0.7 + wet * 0.3) signal ap3

// Chorus effect
let chorus (signal: float[]) (rate: float) (depth: float) (sampleRate: float) =
    let delayMs = 15.0  // Base delay in ms
    let delaySamples = int (delayMs * sampleRate / 1000.0)
    
    signal |> Array.mapi (fun i x -
        let lfo = depth * sin(2.0 * PI * rate * float i / sampleRate)
        let delay = delaySamples + int lfo
        
        if i > delay then
            let delayed = signal.[i - delay]
            x + delayed * 0.5
        else
            x)
```

## 9.6 Biomedical Signal Processing

### ECG Analysis

```fsharp
// R-peak detection in ECG
let detectRPeaks (ecg: float[]) (sampleRate: float) : int[] =
    // Band-pass filter: 5-15 Hz
    let filtered = bandPassFilter ecg 5.0 15.0 sampleRate
    
    // Differentiate
    let differentiated =
        filtered.[1..]
        |> Array.mapi (fun i x -> x - filtered.[i])
    
    // Square
    let squared = differentiated |> Array.map (fun x -> x * x)
    
    // Moving average integration
    let windowSize = int (0.15 * sampleRate)  // 150 ms window
    let integrated = movingAverage squared windowSize
    
    // Threshold detection
    let threshold = Array.average integrated * 2.0
    
    integrated
    |> Array.mapi (fun i x -> (i, x))
    |> Array.filter (fun (i, x) -> x > threshold)
    |> Array.map fst

// Heart rate variability
let computeHRV (peakTimes: int[]) (sampleRate: float) : float[] =
    peakTimes.[1..]
    |> Array.mapi (fun i t -
        float (t - peakTimes.[i]) / sampleRate * 1000.0)  // RR intervals in ms
```

## 9.7 Exercises

### Exercise 9.1: Real-Time Filter

```fsharp
// Implement real-time streaming filter
let createStreamingFilter (coeffs: float[]) =
    let buffer = Array.zeroCreate coeffs.Length
    let mutable idx = 0
    
    fun (sample: float) ->
        buffer.[idx] <- sample
        idx <- (idx + 1) % coeffs.Length
        
        Array.map2 (*) coeffs buffer |> Array.sum
```

### Exercise 9.2: Voice Activity Detection

```fsharp
// Detect speech vs silence
let voiceActivityDetection (signal: float[]) (frameSize: int) : bool[] =
    let energyThreshold = 0.01
    let zcrThreshold = 0.1
    
    signal
    |> Array.windowed frameSize
    |> Array.map (fun frame -
        let energy = frame |> Array.averageBy (fun x -> x * x)
        let zcr = frame.[1..] |> Array.mapi (fun i x -
            if (x >= 0.0) <> (frame.[i] >= 0.0) then 1 else 0) |> Array.sum |> float
        
        energy > energyThreshold && zcr < zcrThreshold * float frameSize)
```

### Exercise 9.3: Spectrogram Visualization

```fsharp
// Generate data for spectrogram plot
let computeSpectrogram (signal: float[]) (fs: float) =
    let windowSize = 256
    let hopSize = 128
    
    let stftResult = stft signal windowSize hopSize
    
    // Convert to dB scale
    stftResult
    |> Array.map (fun frame -
        frame |> Array.map (fun mag -
            20.0 * log10 (mag + 1e-10)))
```

## 9.8 Summary

Key concepts:
- STFT provides time-frequency resolution trade-off
- Filter design shapes frequency content
- Adaptive filters track changing signals
- Spectral estimation reveals frequency content
- Biomedical signals require specialized processing

---

*Next: [Chapter 10: Neural Networks](chapter10.md)*
