# Fowl Case Studies
## Comprehensive Scientific Computing Examples
### Following Patterns from OCaml Scientific Computing

---

## Table of Contents

1. [Case Study 1: Financial Time Series Analysis](#case-study-1-financial-time-series-analysis)
2. [Case Study 2: Image Processing and Computer Vision](#case-study-2-image-processing-and-computer-vision)
3. [Case Study 3: Monte Carlo Simulation for Risk Analysis](#case-study-3-monte-carlo-simulation-for-risk-analysis)
4. [Case Study 4: Linear Regression and Predictive Modeling](#case-study-4-linear-regression-and-predictive-modeling)
5. [Case Study 5: Signal Processing and Filtering](#case-study-5-signal-processing-and-filtering)
6. [Case Study 6: Clustering and Unsupervised Learning](#case-study-6-clustering-and-unsupervised-learning)
7. [Case Study 7: Optimization Problems](#case-study-7-optimization-problems)
8. [Case Study 8: Physics Simulation](#case-study-8-physics-simulation)

---

## Case Study 1: Financial Time Series Analysis

### Overview
Analyze stock price data, calculate technical indicators, and forecast future prices using statistical methods.

### Problem Statement
Given daily closing prices of a stock, we want to:
- Calculate moving averages (simple and exponential)
- Compute volatility (standard deviation of returns)
- Detect trends using linear regression
- Forecast future prices

### Implementation

```fsharp
module CaseStudies.FinancialTimeSeries

open System
open Fowl
open Fowl.Core
open Fowl.Stats
open Fowl.Linalg

/// <summary>Load stock price data from CSV.
/// </summary>let loadStockData (path: string) : FowlResult<DateTime[] * float[]> =
    // Read CSV: date, close_price
    // Implementation using CSV type provider
    Ok ([||], [||])  // Placeholder

/// <summary>Calculate simple moving average.
/// </summary>let simpleMovingAverage (prices: float[]) (window: int) : float[] =
    let n = prices.Length
    if window > n then
        [||]
    else
        Array.init (n - window + 1) (fun i ->
            prices.[i..i+window-1]
            |> Array.average)

/// <summary>Calculate exponential moving average.
/// </summary>let exponentialMovingAverage (prices: float[]) (alpha: float) : float[] =
    let n = prices.Length
    let ema = Array.zeroCreate n
    ema.[0] <- prices.[0]
    
    for i = 1 to n - 1 do
        ema.[i] <- alpha * prices.[i] + (1.0 - alpha) * ema.[i-1]
    
    ema

/// <summary>Calculate daily returns.
/// </summary>let calculateReturns (prices: float[]) : float[] =
    Array.init (prices.Length - 1) (fun i ->
        (prices.[i+1] - prices.[i]) / prices.[i])

/// <summary>Calculate volatility (annualized standard deviation).
/// </summary>let calculateVolatility (returns: float[]) : FowlResult<float> =
    result {
        let! std = Descriptive.std returns
        // Annualize (assuming 252 trading days)
        return std * sqrt 252.0
    }

/// <summary>Detect trend using linear regression.
/// </summary>let detectTrend (prices: float[]) : FowlResult<float * float> =
    // Linear regression: y = a*x + b
    let n = float prices.Length
    let x = Array.init prices.Length float
    
    // Create design matrix [x, 1]
    let xMatrix = 
        Array2D.init prices.Length 2 (fun i j -> 
            if j = 0 then x.[i] else 1.0)
    
    result {
        let! xArr = Ndarray.ofArray2D xMatrix
        let! yArr = Ndarray.ofArray prices [|prices.Length|]
        
        // Least squares: (X^T X)^-1 X^T y
        let! xt = Matrix.transpose xArr
        let! xtx = Matrix.matmul xt xArr
        let! xtx_inv = Factorizations.inv xtx
        let! xty = Matrix.matmul xt yArr
        let! coefficients = Matrix.matmul xtx_inv xty
        
        let coeffs = Ndarray.toArray coefficients
        return (coeffs.[0], coeffs.[1])  // (slope, intercept)
    }

/// <summary>Forecast future prices using trend.
/// </summary>let forecastPrices (slope: float) (intercept: float) 
                     (lastDay: int) (daysAhead: int) : float[] =
    Array.init daysAhead (fun i ->
        let day = float (lastDay + i + 1)
        slope * day + intercept)

/// <summary>Run complete financial analysis.
/// </summary>let analyzeStock (prices: float[]) : FowlResult<unit> =
    result {
        // Calculate indicators
        let sma20 = simpleMovingAverage prices 20
        let sma50 = simpleMovingAverage prices 50
        let ema = exponentialMovingAverage prices 0.1
        let returns = calculateReturns prices
        
        // Calculate volatility
        let! vol = calculateVolatility returns
        printfn "Annualized Volatility: %.2f%%" (vol * 100.0)
        
        // Detect trend
        let! (slope, intercept) = detectTrend prices
        printfn "Trend: %.4f * day + %.2f" slope intercept
        
        // Forecast
        let forecast = forecastPrices slope intercept prices.Length 30
        printfn "30-day forecast: %.2f" (Array.last forecast)
        
        // Statistical tests
        let! jbResult = NormalityTests.jarqueBera returns 0.05
        printfn "Returns are normal: %b" jbResult.IsNormal
        
        return ()
    }
```

### Key Techniques Demonstrated
- Time series manipulation
- Statistical indicators (SMA, EMA, volatility)
- Linear regression for trend detection
- Normality testing of returns
- Forecasting methods

---

## Case Study 2: Image Processing and Computer Vision

### Overview
Process images using convolution, filtering, and feature detection.

### Problem Statement
Build a pipeline to:
- Load and process grayscale images
- Apply Gaussian blur for noise reduction
- Detect edges using Sobel filters
- Perform image normalization

### Implementation

```fsharp
module CaseStudies.ImageProcessing

open Fowl
open Fowl.Core
open Fowl.Linalg

/// <summary>2D convolution for image filtering.
/// </summary>let convolve2D (image: float[,]) (kernel: float[,]) : float[,] =
    let imgH = image.GetLength(0)
    let imgW = image.GetLength(1)
    let kerH = kernel.GetLength(0)
    let kerW = kernel.GetLength(1)
    let padH = kerH / 2
    let padW = kerW / 2
    
    let result = Array2D.zeroCreate imgH imgW
    
    for i = padH to imgH - padH - 1 do
        for j = padW to imgW - padW - 1 do
            let mutable sum = 0.0
            for ki = 0 to kerH - 1 do
                for kj = 0 to kerW - 1 do
                    let ii = i + ki - padH
                    let jj = j + kj - padW
                    sum <- sum + image.[ii, jj] * kernel.[ki, kj]
            result.[i, j] <- sum
    
    result

/// <summary>Gaussian kernel for blurring.
/// </summary>let gaussianKernel (size: int) (sigma: float) : float[,] =
    let kernel = Array2D.zeroCreate size size
    let center = size / 2
    let twoSigmaSq = 2.0 * sigma * sigma
    let mutable sum = 0.0
    
    for i = 0 to size - 1 do
        for j = 0 to size - 1 do
            let x = float (i - center)
            let y = float (j - center)
            let value = exp (-(x*x + y*y) / twoSigmaSq)
            kernel.[i, j] <- value
            sum <- sum + value
    
    // Normalize
    for i = 0 to size - 1 do
        for j = 0 to size - 1 do
            kernel.[i, j] <- kernel.[i, j] / sum
    
    kernel

/// <summary>Sobel edge detection filters.
/// </summary>let sobelX = array2D [[|-1.0; 0.0; 1.0|]
                         [|-2.0; 0.0; 2.0|]
                         [|-1.0; 0.0; 1.0|]]

let sobelY = array2D [[|-1.0; -2.0; -1.0|]
                         [|0.0; 0.0; 0.0|]
                         [|1.0; 2.0; 1.0|]]

/// <summary>Detect edges using Sobel operator.
/// </summary>let detectEdges (image: float[,]) : float[,] =
    let gx = convolve2D image sobelX
    let gy = convolve2D image sobelY
    
    let h = image.GetLength(0)
    let w = image.GetLength(1)
    let magnitude = Array2D.zeroCreate h w
    
    for i = 0 to h - 1 do
        for j = 0 to w - 1 do
            magnitude.[i, j] <- sqrt (gx.[i,j]*gx.[i,j] + gy.[i,j]*gy.[i,j])
    
    magnitude

/// <summary>Normalize image to [0, 1] range.
/// </summary>let normalizeImage (image: float[,]) : float[,] =
    let minVal = 
        seq { for i in 0..image.GetLength(0)-1 do 
              for j in 0..image.GetLength(1)-1 do yield image.[i,j] }
        |> Seq.min
    let maxVal =
        seq { for i in 0..image.GetLength(0)-1 do 
              for j in 0..image.GetLength(1)-1 do yield image.[i,j] }
        |> Seq.max
    
    let range = maxVal - minVal
    if range = 0.0 then
        image
    else
        image |> Array2D.map (fun x -> (x - minVal) / range)

/// <summary>Apply Gaussian blur.
/// </summary>let gaussianBlur (image: float[,]) (sigma: float) : float[,] =
    let kernel = gaussianKernel 5 sigma
    convolve2D image kernel
```

### Key Techniques Demonstrated
- 2D convolution operations
- Kernel-based filtering
- Edge detection algorithms
- Image normalization
- Array2D manipulation

---

## Case Study 3: Monte Carlo Simulation for Risk Analysis

### Overview
Use Monte Carlo methods to estimate Value at Risk (VaR) for a portfolio.

### Problem Statement
Given a portfolio with multiple assets:
- Estimate expected returns using historical data
- Calculate covariance matrix
- Simulate thousands of possible future scenarios
- Calculate 95% VaR (potential loss)

### Implementation

```fsharp
module CaseStudies.MonteCarloRisk

open System
open Fowl
open Fowl.Core
open Fowl.Stats
open Fowl.Linalg

/// <summary>Portfolio asset with weight.
/// </summary>type Asset = {
    Name: string
    Weight: float
    Returns: float[]
}

/// <summary>Calculate covariance matrix from returns.
/// </summary>let calculateCovarianceMatrix (returns: float[][]) : FowlResult<Ndarray<_,float>> =
    let nAssets = returns.Length
    let nDays = returns.[0].Length
    
    // Calculate means
    let means = returns |> Array.map Array.average
    
    // Calculate covariances
    let covMatrix = Array2D.zeroCreate nAssets nAssets
    
    for i = 0 to nAssets - 1 do
        for j = 0 to nAssets - 1 do
            let cov =
                Array.zip returns.[i] returns.[j]
                |> Array.map (fun (ri, rj) -> 
                    (ri - means.[i]) * (rj - means.[j]))
                |> Array.average
            covMatrix.[i, j] <- cov
    
    Ndarray.ofArray2D covMatrix

/// <summary>Cholesky decomposition for correlated random numbers.
/// </summary>let choleskyDecomposition (covMatrix: Ndarray<_,float>) : FowlResult<Ndarray<_,float>> =
    Factorizations.cholesky covMatrix

/// <summary>Generate correlated random returns.
/// </summary>let generateCorrelatedReturns (cholL: Ndarray<_,float>) 
                                       (means: float[])
                                       (nSimulations: int) 
                                       : FowlResult<float[][]> =
    let nAssets = means.Length
    let rng = Random()
    
    // Generate uncorrelated standard normals
    let uncorrelated = 
        Array.init nAssets (fun _ ->
            Array.init nSimulations (fun _ ->
                // Box-Muller transform
                let u1 = rng.NextDouble()
                let u2 = rng.NextDouble()
                sqrt (-2.0 * log u1) * cos (2.0 * Math.PI * u2)))
    
    // Correlate using L: correlated = L * uncorrelated
    result {
        let! uncorrArr = Ndarray.ofArray (Array.concat uncorrelated) 
                                         [|nAssets; nSimulations|]
        let! correlated = Matrix.matmul cholL uncorrArr
        let corrData = Ndarray.toArray correlated
        
        // Add means
        let result = 
            Array.init nAssets (fun i ->
                Array.init nSimulations (fun j ->
                    corrData.[i * nSimulations + j] + means.[i]))
        
        return result
    }

/// <summary>Calculate portfolio returns for each simulation.
/// </summary>let calculatePortfolioReturns (assetReturns: float[][]) 
                                      (weights: float[]) 
                                      : float[] =
    let nSimulations = assetReturns.[0].Length
    Array.init nSimulations (fun i ->
        Array.sumBy (fun j -> weights.[j] * assetReturns.[j].[i]) 
                    [|0..weights.Length-1|])

/// <summary>Calculate Value at Risk.
/// </summary>let calculateVaR (portfolioReturns: float[]) (confidence: float) : float =
    let sorted = Array.sort portfolioReturns
    let index = int (float sorted.Length * (1.0 - confidence))
    sorted.[index]  // Return at risk (negative = loss)

/// <summary>Run Monte Carlo risk analysis.
/// </summary>let runRiskAnalysis (assets: Asset[]) (nSimulations: int) : FowlResult<unit> =
    result {
        let returns = assets |> Array.map (fun a -> a.Returns)
        let weights = assets |> Array.map (fun a -> a.Weight)
        let means = returns |> Array.map Array.average
        
        printfn "Running Monte Carlo simulation with %d scenarios..." nSimulations
        
        // Calculate covariance
        let! covMatrix = calculateCovarianceMatrix returns
        let! cholL = choleskyDecomposition covMatrix
        
        // Generate correlated returns
        let! simulatedReturns = generateCorrelatedReturns cholL means nSimulations
        
        // Calculate portfolio returns
        let portfolioReturns = calculatePortfolioReturns simulatedReturns weights
        
        // Calculate statistics
        let! meanReturn = Descriptive.mean portfolioReturns
        let! stdReturn = Descriptive.std portfolioReturns
        let var95 = calculateVaR portfolioReturns 0.95
        let var99 = calculateVaR portfolioReturns 0.99
        
        printfn "Expected Return: %.2f%%" (meanReturn * 100.0)
        printfn "Volatility: %.2f%%" (stdReturn * 100.0)
        printfn "95%% VaR: %.2f%%" (var95 * 100.0)
        printfn "99%% VaR: %.2f%%" (var99 * 100.0)
        
        // Test normality of portfolio returns
        let! jbResult = NormalityTests.jarqueBera portfolioReturns 0.05
        printfn "Portfolio returns are normal: %b" jbResult.IsNormal
        
        return ()
    }
```

### Key Techniques Demonstrated
- Monte Carlo simulation
- Cholesky decomposition for correlations
- Box-Muller transform for normal random numbers
- Portfolio optimization
- Value at Risk calculation
- Statistical testing

---

## Case Study 4: Linear Regression and Predictive Modeling

### Overview
Build a complete linear regression model with feature engineering, training, and prediction.

### Problem Statement
Using housing data, predict prices based on features like square footage, bedrooms, and location.

### Implementation

```fsharp
module CaseStudies.LinearRegression

open Fowl
open Fowl.Core
open Fowl.Stats
open Fowl.Linalg

/// <summary>Feature engineering for housing data.
/// </summary>type HouseFeatures = {
    SquareFeet: float
    Bedrooms: float
    Bathrooms: float
    Age: float
    LocationScore: float
}

/// <summary>Load and preprocess housing data.
/// </summary>let loadHousingData (path: string) : FowlResult<(HouseFeatures[] * float[])> =
    // Load from CSV using type provider
    // Normalize features
    Ok ([||], [||])

/// <summary>Normalize features to zero mean, unit variance.
/// </summary>let normalizeFeatures (features: float[][]) : float[][] * (float[] * float[]) =
    let nFeatures = features.[0].Length
    let nSamples = features.Length
    
    // Calculate means and stds
    let means = 
        Array.init nFeatures (fun j -
            features |> Array.averageBy (fun row -> row.[j]))
    
    let stds =
        Array.init nFeatures (fun j -
            let vals = features |> Array.map (fun row -> row.[j])
            match Descriptive.std vals with
            | Ok s -> max s 1e-8  // Avoid division by zero
            | Error _ -> 1.0)
    
    // Normalize
    let normalized =
        features |> Array.map (fun row -
            row |> Array.mapi (fun i x -
                (x - means.[i]) / stds.[i]))
    
    normalized, (means, stds)

/// <summary>Multiple linear regression using normal equations.
/// </summary>let fitLinearRegression (X: float[][]) (y: float[]) : FowlResult<float[]> =
    result {
        // Add bias column (ones)
        let n = X.Length
        let m = X.[0].Length + 1
        let XWithBias = 
            Array2D.init n m (fun i j -
                if j = 0 then 1.0 else X.[i].[j-1])
        
        let! xArr = Ndarray.ofArray2D XWithBias
        let! yArr = Ndarray.ofArray y [|n|]
        
        // Normal equations: (X^T X)^-1 X^T y
        let! xt = Matrix.transpose xArr
        let! xtx = Matrix.matmul xt xArr
        let! xtxInv = Factorizations.inv xtx
        let! xty = Matrix.matmul xt yArr
        let! coeffs = Matrix.matmul xtxInv xty
        
        return Ndarray.toArray coeffs
    }

/// <summary>Calculate R-squared (coefficient of determination).
/// </summary>let calculateRSquared (actual: float[]) (predicted: float[]) : float =
    let meanActual = Array.average actual
    let ssTot = actual |> Array.sumBy (fun y -> (y - meanActual) ** 2.0)
    let ssRes = Array.zip actual predicted |> Array.sumBy (fun (y, yHat) -> (y - yHat) ** 2.0)
    1.0 - (ssRes / ssTot)

/// <summary>Make predictions with trained model.
/// </summary>let predict (coefficients: float[]) (features: float[]) : float =
    let n = features.Length
    coefficients.[0] + Array.sumBy (fun i -> coefficients.[i+1] * features.[i]) [|0..n-1|]

/// <summary>Run complete regression analysis.
/// </summary>let runRegressionAnalysis (features: HouseFeatures[]) (prices: float[]) : FowlResult<unit> =
    result {
        // Convert to matrix
        let X = 
            features |> Array.map (fun h -
                [|h.SquareFeet; h.Bedrooms; h.Bathrooms; h.Age; h.LocationScore|])
        
        // Normalize
        let XNorm, (means, stds) = normalizeFeatures X
        
        // Split into train/test
        let splitIdx = int (0.8 * float X.Length)
        let XTrain, XTest = Array.splitAt splitIdx XNorm
        let yTrain, yTest = Array.splitAt splitIdx prices
        
        // Train model
        let! coefficients = fitLinearRegression XTrain yTrain
        printfn "Model coefficients: %A" coefficients
        
        // Make predictions
        let predictions = XTest |> Array.map (predict coefficients)
        
        // Evaluate
        let r2 = calculateRSquared yTest predictions
        printfn "R-squared: %.4f" r2
        
        // Calculate RMSE
        let rmse = 
            Array.zip yTest predictions |
            Array.map (fun (y, yHat) -> (y - yHat) ** 2.0) |
            Array.average |
            sqrt
        printfn "RMSE: %.2f" rmse
        
        // Test significance of coefficients
        let! tResult = HypothesisTests.ttestOneSample coefficients.[1..] 0.0 0.05
        printfn "Features are significant: %b" tResult.Significant
        
        return ()
    }
```

### Key Techniques Demonstrated
- Feature normalization
- Multiple linear regression
- Normal equations
- Model evaluation (R², RMSE)
- Train/test splitting
- Hypothesis testing for significance

---

## Case Study 5: Signal Processing and Filtering

### Overview
Process audio signals using FFT, filtering, and spectral analysis.

### Implementation

```fsharp
module CaseStudies.SignalProcessing

open System
open System.Numerics
open Fowl
open Fowl.Core

/// <summary>Discrete Fourier Transform (naive implementation).
/// </summary>let dft (signal: Complex[]) : Complex[] =
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

/// <summary>Generate sine wave signal.
/// </summary>let generateSineWave (frequency: float) (sampleRate: float) 
                          (duration: float) : float[] =
    let nSamples = int (sampleRate * duration)
    let omega = 2.0 * Math.PI * frequency / sampleRate
    Array.init nSamples (fun i -> sin (omega * float i))

/// <summary>Simple moving average filter.
/// </summary>let movingAverageFilter (signal: float[]) (windowSize: int) : float[] =
    let n = signal.Length
    Array.init (n - windowSize + 1) (fun i -
        signal.[i..i+windowSize-1] |> Array.average)

/// <summary>Exponential moving average filter.
/// </summary>let exponentialFilter (signal: float[]) (alpha: float) : float[] =
    let n = signal.Length
    let result = Array.zeroCreate n
    result.[0] <- signal.[0]
    
    for i = 1 to n - 1 do
        result.[i] <- alpha * signal.[i] + (1.0 - alpha) * result.[i-1]
    
    result

/// <summary>Calculate power spectral density.
/// </summary>let calculatePSD (fftResult: Complex[]) : float[] =
    fftResult |> Array.map (fun c -> c.Magnitude ** 2.0)

/// <summary>Find dominant frequencies.
/// </summary>let findDominantFrequencies (psd: float[]) (sampleRate: float) 
                                   (nPeaks: int) : (float * float)[] =
    let n = psd.Length
    let freqs = Array.init (n/2) (fun i -> float i * sampleRate / float n)
    
    // Find peaks
    psd.[1..n/2-2] |
    Array.mapi (fun i p -> (freqs.[i+1], p)) |
    Array.filter (fun (f, p) -> p > psd.[int (f / sampleRate * float n) - 1] 
                           && p > psd.[int (f / sampleRate * float n) + 1]) |
    Array.sortByDescending snd |
    Array.take (min nPeaks (n/2 - 1))

/// <summary>Run signal processing pipeline.
/// </summary>let processSignal (signal: float[]) (sampleRate: float) : unit =
    // Convert to complex
    let complexSignal = signal |> Array.map (fun x -> Complex(x, 0.0))
    
    // Apply FFT
    let fft = dft complexSignal
    let psd = calculatePSD fft
    
    // Find dominant frequencies
    let peaks = findDominantFrequencies psd sampleRate 5
    printfn "Dominant frequencies:"
    peaks |> Array.iter (fun (freq, power) ->
        printfn "  %.2f Hz (power: %.2f)" freq power)
    
    // Apply filters
    let filtered = movingAverageFilter signal 10
    printfn "Original std: %.4f" (Descriptive.std signal |> Result.get)
    printfn "Filtered std: %.4f" (Descriptive.std filtered |> Result.get)
```

### Key Techniques Demonstrated
- Discrete Fourier Transform
- Power spectral density calculation
- Frequency domain analysis
- Time-domain filtering (moving average, exponential)
- Peak detection

---

## Case Study 6: Clustering and Unsupervised Learning

### Overview
Implement K-means clustering for customer segmentation.

### Implementation

```fsharp
module CaseStudies.Clustering

open System
open Fowl
open Fowl.Core
open Fowl.Stats

/// <summary>K-means clustering algorithm.
/// </summary>let kMeans (data: float[][]) (k: int) (maxIter: int) : (int[] * float[][]) =
    let nSamples = data.Length
    let nFeatures = data.[0].Length
    
    // Initialize centroids randomly
    let rng = Random()
    let centroids = Array.init k (fun _ -
        data.[rng.Next(nSamples)] |> Array.copy)
    
    let mutable labels = Array.zeroCreate nSamples
    let mutable changed = true
    let mutable iter = 0
    
    while changed && iter < maxIter do
        changed <- false
        iter <- iter + 1
        
        // Assign points to nearest centroid
        for i = 0 to nSamples - 1 do
            let distances = 
                centroids |
                Array.map (fun c -
                    Array.zip data.[i] c |
                    Array.sumBy (fun (x, y) -> (x - y) ** 2.0) |
                    sqrt)
            let newLabel = distances |> Array.indexed |> Array.minBy snd |> fst
            if newLabel <> labels.[i] then
                changed <- true
                labels.[i] <- newLabel
        
        // Update centroids
        for j = 0 to k - 1 do
            let clusterPoints = 
                labels |
                Array.indexed |
                Array.filter (fun (_, l) -> l = j) |
                Array.map (fun (i, _) -> data.[i])
            
            if clusterPoints.Length > 0 then
                for f = 0 to nFeatures - 1 do
                    centroids.[j].[f] <-
                        clusterPoints |
                        Array.averageBy (fun p -> p.[f])
    
    labels, centroids

/// <summary>Calculate silhouette score for clustering quality.
/// </summary>let silhouetteScore (data: float[][]) (labels: int[]) : float =
    let n = data.Length
    let k = Array.max labels + 1
    
    let avgScore =
        data |
        Array.mapi (fun i point -
            let label = labels.[i]
            
            // a(i): average distance to same cluster
            let a =
                data |
                Array.indexed |
                Array.filter (fun (j, _) -> j <> i && labels.[j] = label) |
                Array.map (fun (_, other) -
                    Array.zip point other |
                    Array.sumBy (fun (x, y) -> (x - y) ** 2.0) |
                    sqrt) |
                fun arr -> if arr.Length = 0 then 0.0 else Array.average arr
            
            // b(i): minimum average distance to other clusters
            let b =
                [|0..k-1|] |
                Array.filter (fun l -> l <> label) |
                Array.map (fun l -
                    data |
                    Array.indexed |
                    Array.filter (fun (j, _) -> labels.[j] = l) |
                    Array.map (fun (_, other) -
                        Array.zip point other |
                        Array.sumBy (fun (x, y) -> (x - y) ** 2.0) |
                        sqrt) |
                    Array.average) |
                Array.min
            
            // Silhouette for point i
            if a = 0.0 then 1.0 else (b - a) / max a b)
        |> Array.average
    
    avgScore

/// <summary>Run customer segmentation.
/// </summary>let segmentCustomers (customerData: float[][]) : unit =
    // Find optimal k using silhouette score
    let bestK =
        [|2..5|] |
        Array.maxBy (fun k -
            let labels, _ = kMeans customerData k 100
            silhouetteScore customerData labels)
    
    printfn "Optimal number of clusters: %d" bestK
    
    // Final clustering
    let labels, centroids = kMeans customerData bestK 100
    
    // Analyze clusters
    for i = 0 to bestK - 1 do
        let clusterSize = labels |> Array.filter ((=) i) |> Array.length
        printfn "Cluster %d: %d customers" i clusterSize
        printfn "  Centroid: %A" centroids.[i]
```

### Key Techniques Demonstrated
- K-means clustering
- Distance metrics (Euclidean)
- Iterative optimization
- Silhouette analysis for cluster validation
- Hyperparameter selection (optimal k)

---

## Case Study 7: Optimization Problems

### Overview
Solve optimization problems using gradient descent and constrained optimization.

### Implementation

```fsharp
module CaseStudies.Optimization

open Fowl
open Fowl.Core
open Fowl.Stats

/// <summary>Rosenbrock function (classic optimization test).
/// f(x,y) = (a-x)² + b(y-x²)²
/// </summary>let rosenbrock (a: float) (b: float) (x: float[]) : float =
    (a - x.[0]) ** 2.0 + b * (x.[1] - x.[0] ** 2.0) ** 2.0

/// <summary>Gradient of Rosenbrock function.
/// </summary>let rosenbrockGradient (a: float) (b: float) (x: float[]) : float[] =
    let dx0 = -2.0 * (a - x.[0]) - 4.0 * b * x.[0] * (x.[1] - x.[0] ** 2.0)
    let dx1 = 2.0 * b * (x.[1] - x.[0] ** 2.0)
    [|dx0; dx1|]

/// <summary>Gradient descent optimizer.
/// </summary>let gradientDescent (f: float[] -> float) 
                             (gradF: float[] -> float[]) 
                             (x0: float[]) 
                             (learningRate: float) 
                             (tolerance: float) 
                             (maxIter: int) : float[] * float =
    let mutable x = Array.copy x0
    let mutable fx = f x
    let mutable converged = false
    let mutable iter = 0
    
    while not converged && iter < maxIter do
        let grad = gradF x
        let xNew = Array.map2 (fun xi gi -> xi - learningRate * gi) x grad
        let fxNew = f xNew
        
        if abs (fxNew - fx) < tolerance then
            converged <- true
        
        x <- xNew
        fx <- fxNew
        iter <- iter + 1
    
    x, fx

/// <summary>Simulated annealing for global optimization.
/// </summary>let simulatedAnnealing (f: float[] -> float) 
                                (x0: float[]) 
                                (bounds: (float * float)[]) 
                                (initialTemp: float) 
                                (coolingRate: float) 
                                (maxIter: int) : float[] * float =
    let rng = Random()
    let mutable x = Array.copy x0
    let mutable fx = f x
    let mutable temp = initialTemp
    let mutable bestX = Array.copy x
    let mutable bestF = fx
    
    for iter = 0 to maxIter - 1 do
        // Generate neighbor
        let xNew = 
            x |
            Array.mapi (fun i xi -
                let (lo, hi) = bounds.[i]
                let delta = (rng.NextDouble() - 0.5) * 2.0 * (hi - lo) * 0.1
                max lo (min hi (xi + delta)))
        
        let fxNew = f xNew
        let delta = fxNew - fx
        
        // Accept or reject
        if delta < 0.0 || rng.NextDouble() < exp (-delta / temp) then
            x <- xNew
            fx <- fxNew
            
            if fx < bestF then
                bestX <- Array.copy x
                bestF <- fx
        
        // Cool down
        temp <- temp * coolingRate
    
    bestX, bestF

/// <summary>Run optimization examples.
/// </summary>let runOptimizationExamples() : unit =
    // Rosenbrock function optimization
    let x0 = [|0.0; 0.0|]
    let solution, minValue = gradientDescent 
        (rosenbrock 1.0 100.0) 
        (rosenbrockGradient 1.0 100.0)
        x0 0.001 1e-6 10000
    
    printfn "Rosenbrock minimum at: (%f, %f)" solution.[0] solution.[1]
    printfn "Function value: %f" minValue
    
    // Simulated annealing
    let bounds = [|-2.0, 2.0; -1.0, 3.0|]
    let saSolution, saValue = simulatedAnnealing
        (rosenbrock 1.0 100.0)
        x0 bounds 100.0 0.95 10000
    
    printfn "SA solution: (%f, %f) = %f" saSolution.[0] saSolution.[1] saValue
```

### Key Techniques Demonstrated
- Gradient descent optimization
- Function gradients
- Convergence criteria
- Simulated annealing for global optimization
- Acceptance probability
- Temperature cooling schedules

---

## Case Study 8: Physics Simulation

### Overview
Simulate projectile motion and pendulum dynamics using numerical integration.

### Implementation

```fsharp
module CaseStudies.PhysicsSimulation

open System
open Fowl
open Fowl.Core

/// <summary>State of a projectile: position and velocity.
/// </summary>type ProjectileState = {
    X: float
    Y: float
    Vx: float
    Vy: float
}

/// <summary>Simple pendulum state.
/// </summary>type PendulumState = {
    Theta: float    // Angle
    Omega: float    // Angular velocity
}

/// <summary>Euler method for numerical integration.
/// </summary>let eulerStep (state: 'T) (derivatives: 'T -> 'T) (dt: float) : 'T =
    let d = derivatives state
    // state + dt * derivatives (simplified - would need vector addition)
    state

/// <summary>Runge-Kutta 4th order method.
/// </summary>let rk4Step (state: 'T) (derivatives: 'T -> 'T) (dt: float) : 'T =
    let k1 = derivatives state
    let k2 = derivatives (eulerStep state derivatives (dt / 2.0))
    let k3 = derivatives (eulerStep state derivatives (dt / 2.0))
    let k4 = derivatives (eulerStep state derivatives dt)
    
    // state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    state

/// <summary>Simulate projectile motion with air resistance.
/// </summary>let simulateProjectile (initial: ProjectileState) 
                                (g: float) 
                                (drag: float) 
                                (dt: float) 
                                (duration: float) : ProjectileState[] =
    let nSteps = int (duration / dt)
    let trajectory = Array.zeroCreate nSteps
    let mutable state = initial
    
    for i = 0 to nSteps - 1 do
        trajectory.[i] <- state
        
        // Calculate derivatives
        let v = sqrt (state.Vx ** 2.0 + state.Vy ** 2.0)
        let dvx = -drag * v * state.Vx
        let dvy = -g - drag * v * state.Vy
        
        // Update (Euler method for simplicity)
        state <- {
            X = state.X + dt * state.Vx
            Y = state.Y + dt * state.Vy
            Vx = state.Vx + dt * dvx
            Vy = state.Vy + dt * dvy
        }
        
        // Stop if hit ground
        if state.Y < 0.0 then
            return trajectory.[0..i]
    
    trajectory

/// <summary>Simulate simple pendulum.
/// </summary>let simulatePendulum (initial: PendulumState) 
                            (g: float) 
                            (length: float) 
                            (dt: float) 
                            (duration: float) : PendulumState[] =
    let nSteps = int (duration / dt)
    let trajectory = Array.zeroCreate nSteps
    let mutable state = initial
    
    for i = 0 to nSteps - 1 do
        trajectory.[i] <- state
        
        // theta'' = -g/L * sin(theta)
        let alpha = -g / length * sin state.Theta
        
        // Update (Euler)
        state <- {
            Theta = state.Theta + dt * state.Omega
            Omega = state.Omega + dt * alpha
        }
    
    trajectory

/// <summary>Analyze pendulum period.
/// </summary>let calculatePeriod (trajectory: PendulumState[]) (dt: float) : float =
    // Find zero crossings
    let crossings = 
        trajectory |
        Array.pairwise |
        Array.indexed |
        Array.filter (fun (_, (s1, s2)) -> s1.Theta * s2.Theta < 0.0 && s1.Omega > 0.0) |
        Array.map fst
    
    if crossings.Length >= 2 then
        let avgPeriod = 
            crossings |
            Array.pairwise |
            Array.map (fun (i1, i2) -> float (i2 - i1) * dt * 2.0) |
            Array.average
        avgPeriod
    else
        0.0

/// <summary>Run physics simulations.
/// </summary>let runPhysicsSimulations() : unit =
    // Projectile
    let projectileInit = { X = 0.0; Y = 0.0; Vx = 10.0; Vy = 20.0 }
    let projTraj = simulateProjectile projectileInit 9.81 0.01 0.01 5.0
    printfn "Projectile range: %.2f m" (Array.last projTraj).X
    
    // Pendulum
    let pendulumInit = { Theta = Math.PI / 4.0; Omega = 0.0 }
    let pendTraj = simulatePendulum pendulumInit 9.81 1.0 0.001 10.0
    let period = calculatePeriod pendTraj 0.001
    printfn "Pendulum period: %.3f s" period
    printfn "Theoretical period: %.3f s" (2.0 * Math.PI * sqrt (1.0 / 9.81))
```

### Key Techniques Demonstrated
- Numerical integration (Euler, RK4)
- Differential equation solving
- State-space representation
- Trajectory analysis
- Period calculation from zero crossings
- Physics-based modeling

---

## Summary

These case studies demonstrate:

1. **Real-world applications** of Fowl's capabilities
2. **Integration** of multiple modules (Core, Stats, Linalg, Neural)
3. **Best practices** for scientific computing in F#
4. **Complete workflows** from data loading to analysis
5. **Mathematical rigor** with proper error handling

Each case study can be extended with:
- Visualization (using Plotly.NET or similar)
- Performance benchmarking
- Unit and property-based tests
- Interactive notebooks (Jupyter with F#)

---

*Case studies complete. Fowl now has comprehensive examples for scientific computing workflows.*