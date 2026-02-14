/// Financial Time Series Analysis Example
/// Based on OCaml Scientific Computing use cases

module FinancialAnalysisExample

open System
open Fowl
open Fowl.Core
open Fowl.Stats
open Fowl.Linalg

/// Sample stock price data (daily closing prices)
let samplePrices = [|
    100.0; 102.5; 101.0; 103.5; 105.0; 104.0; 106.5; 108.0; 107.0; 109.5;
    111.0; 110.0; 112.5; 114.0; 113.0; 115.5; 117.0; 116.0; 118.5; 120.0;
    119.0; 121.5; 123.0; 122.0; 124.5; 126.0; 125.0; 127.5; 129.0; 128.0
|]

/// Calculate simple moving average
let simpleMovingAverage (prices: float[]) (window: int) : float[] =
    let n = prices.Length
    if window > n then
        [||]
    else
        Array.init (n - window + 1) (fun i ->
            prices.[i..i+window-1]
            |> Array.average)

/// Calculate exponential moving average
let exponentialMovingAverage (prices: float[]) (alpha: float) : float[] =
    let n = prices.Length
    let ema = Array.zeroCreate n
    ema.[0] <- prices.[0]
    
    for i = 1 to n - 1 do
        ema.[i] <- alpha * prices.[i] + (1.0 - alpha) * ema.[i-1]
    
    ema

/// Calculate daily returns
let calculateReturns (prices: float[]) : float[] =
    Array.init (prices.Length - 1) (fun i ->
        (prices.[i+1] - prices.[i]) / prices.[i])

/// Calculate annualized volatility
let calculateVolatility (returns: float[]) : FowlResult<float> =
    result {
        let! std = Descriptive.std returns
        // Annualize (assuming 252 trading days)
        return std * sqrt 252.0
    }

/// Linear regression for trend detection
let detectTrend (prices: float[]) : FowlResult<(float * float)> =
    let n = prices.Length
    let x = Array.init n float
    
    // Create design matrix [x, 1]
    let xMatrix = 
        Array2D.init n 2 (fun i j -> 
            if j = 0 then x.[i] else 1.0)
    
    result {
        let! xArr = Ndarray.ofArray2D xMatrix
        let! yArr = Ndarray.ofArray prices [|n|]
        
        // Least squares: (X^T X)^-1 X^T y
        let! xt = Matrix.transpose xArr
        let! xtx = Matrix.matmul xt xArr
        let! xtx_inv = Factorizations.inv xtx
        let! xty = Matrix.matmul xt yArr
        let! coefficients = Matrix.matmul xtx_inv xty
        
        let coeffs = Ndarray.toArray coefficients
        return (coeffs.[0], coeffs.[1])  // (slope, intercept)
    }

/// Run complete financial analysis
let runAnalysis() : FowlResult<unit> =
    result {
        printfn "=== Financial Time Series Analysis ==="
        printfn "Sample size: %d days" samplePrices.Length
        printfn ""
        
        // Moving averages
        let sma20 = simpleMovingAverage samplePrices 20
        let ema = exponentialMovingAverage samplePrices 0.1
        
        printfn "Simple Moving Average (20-day): %.2f" (Array.last sma20)
        printfn "Exponential Moving Average: %.2f" (Array.last ema)
        printfn ""
        
        // Returns and volatility
        let returns = calculateReturns samplePrices
        let! vol = calculateVolatility returns
        printfn "Daily Returns: %.2f%% average" ((Array.average returns) * 100.0)
        printfn "Annualized Volatility: %.2f%%" (vol * 100.0)
        printfn ""
        
        // Trend detection
        let! (slope, intercept) = detectTrend samplePrices
        printfn "Trend Analysis:"
        printfn "  Slope: %.4f (%.2f%% per day)" slope (slope * 100.0)
        printfn "  Intercept: %.2f" intercept
        
        if slope > 0.001 then
            printfn "  → UPWARD TREND detected"
        elif slope < -0.001 then
            printfn "  → DOWNWARD TREND detected"
        else
            printfn "  → NEUTRAL TREND"
        printfn ""
        
        // Statistical tests
        let! jbResult = NormalityTests.jarqueBera returns 0.05
        printfn "Jarque-Bera Normality Test:"
        printfn "  Statistic: %.4f" jbResult.Statistic
        printfn "  P-value: %.4f" jbResult.PValue
        printfn "  Returns are normal: %b" jbResult.IsNormal
        printfn ""
        
        // Value at Risk (95%)
        let sortedReturns = Array.sort returns
        let var95 = sortedReturns.[int (float sortedReturns.Length * 0.05)]
        printfn "Value at Risk (95%%): %.2f%%" (var95 * 100.0)
        
        printfn ""
        printfn "=== Analysis Complete ==="
        
        return ()
    }

/// Entry point
[<EntryPoint>]
let main argv =
    match runAnalysis() with
    | Ok () -> 
        printfn "Success!"
        0
    | Error e -> 
        printfn "Error: %A" e
        1