/// Monte Carlo Simulation Example
/// Portfolio risk analysis and Value at Risk (VaR)

module MonteCarloExample

open System
open Fowl
open Fowl.Core
open Fowl.Stats
open Fowl.Linalg

/// Portfolio assets with historical returns
let assets = [|
    "AAPL", [|0.02; 0.015; -0.01; 0.025; 0.01; -0.005; 0.03; 0.02; -0.015; 0.025|]
    "GOOGL", [|0.015; 0.01; 0.005; 0.02; -0.01; 0.015; 0.01; 0.025; 0.005; 0.015|]
    "MSFT", [|0.01; 0.02; 0.015; 0.005; 0.02; 0.01; 0.015; 0.01; 0.02; 0.01|]
|]

/// Portfolio weights (must sum to 1.0)
let weights = [|0.4; 0.35; 0.25|]

/// Calculate covariance matrix
let calculateCovariance (returns: float[][]) : FowlResult<Ndarray<_,float>> =
    let nAssets = returns.Length
    let means = returns |> Array.map Array.average
    
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

/// Generate correlated random returns using Cholesky
let simulateReturns (cholL: Ndarray<_,float>) 
                    (means: float[])
                    (nSimulations: int) 
                    : FowlResult<float[][]> =
    let nAssets = means.Length
    let rng = Random()
    
    // Generate uncorrelated standard normals
    let uncorrelated = 
        Array.init nAssets (fun _ ->
            Array.init nSimulations (fun _ -
                // Box-Muller
                let u1 = rng.NextDouble()
                let u2 = rng.NextDouble()
                sqrt (-2.0 * log u1) * cos (2.0 * Math.PI * u2)))
    
    result {
        let! uncorrArr = Ndarray.ofArray (Array.concat uncorrelated) 
                                         [|nAssets; nSimulations|]
        let! correlated = Matrix.matmul cholL uncorrArr
        let corrData = Ndarray.toArray correlated
        
        // Add means
        let result = 
            Array.init nAssets (fun i -
                Array.init nSimulations (fun j -
                    corrData.[i * nSimulations + j] + means.[i]))
        
        return result
    }

/// Calculate portfolio returns
let calculatePortfolioReturns (assetReturns: float[][]) 
                              (weights: float[]) 
                              : float[] =
    let nSimulations = assetReturns.[0].Length
    Array.init nSimulations (fun i -
        Array.sumBy (fun j -> weights.[j] * assetReturns.[j].[i]) 
                    [|0..weights.Length-1|])

/// Calculate Value at Risk
let calculateVaR (returns: float[]) (confidence: float) : float =
    let sorted = Array.sort returns
    let index = max 0 (min (sorted.Length - 1) (int (float sorted.Length * (1.0 - confidence))))
    sorted.[index]

/// Run Monte Carlo simulation
let runMonteCarlo() : FowlResult<unit> =
    result {
        printfn "=== Monte Carlo Risk Analysis ==="
        printfn "Portfolio: AAPL (40%%), GOOGL (35%%), MSFT (25%%)"
        printfn ""
        
        let returns = assets |> Array.map snd
        let means = returns |> Array.map Array.average
        
        printfn "Historical Returns:"
        assets |> Array.iteri (fun i (name, _) -
            printfn "  %s: %.2f%% avg daily" name (means.[i] * 100.0))
        printfn ""
        
        // Calculate covariance
        let! covMatrix = calculateCovariance returns
        let! cholL = Factorizations.cholesky covMatrix
        
        printfn "Running 10,000 simulations..."
        let nSimulations = 10000
        
        // Generate correlated returns
        let! simulatedReturns = simulateReturns cholL means nSimulations
        
        // Calculate portfolio returns
        let portfolioReturns = calculatePortfolioReturns simulatedReturns weights
        
        // Calculate statistics
        let! meanReturn = Descriptive.mean portfolioReturns
        let! stdReturn = Descriptive.std portfolioReturns
        let var95 = calculateVaR portfolioReturns 0.95
        let var99 = calculateVaR portfolioReturns 0.99
        
        printfn ""
        printfn "Simulation Results:"
        printfn "  Expected Daily Return: %.4f%%" (meanReturn * 100.0)
        printfn "  Daily Volatility: %.4f%%" (stdReturn * 100.0)
        printfn "  Annualized Return: %.2f%%" (meanReturn * 252.0 * 100.0)
        printfn "  Annualized Volatility: %.2f%%" (stdReturn * sqrt 252.0 * 100.0)
        printfn ""
        printfn "Value at Risk:"
        printfn "  95%% VaR: %.4f%% (%.2f%% annualized)" 
                (var95 * 100.0) (var95 * sqrt 252.0 * 100.0)
        printfn "  99%% VaR: %.4f%% (%.2f%% annualized)"
                (var99 * 100.0) (var99 * sqrt 252.0 * 100.0)
        printfn ""
        printfn "Interpretation:"
        printfn "  With 95%% confidence, daily loss will not exceed %.2f%%"
                (abs var95 * 100.0)
        printfn "  With 99%% confidence, daily loss will not exceed %.2f%%"
                (abs var99 * 100.0)
        printfn ""
        
        // Normality test
        let! jbResult = NormalityTests.jarqueBera portfolioReturns 0.05
        printfn "Jarque-Bera Test:"
        printfn "  Portfolio returns normal: %b" jbResult.IsNormal
        printfn "  (p-value: %.4f)" jbResult.PValue
        
        if not jbResult.IsNormal then
            printfn "  ⚠️  Returns are not normal - VaR may underestimate risk"
        printfn ""
        
        printfn "=== Simulation Complete ==="
        
        return ()
    }

[<EntryPoint>]
let main argv =
    match runMonteCarlo() with
    | Ok () -> 0
    | Error e -> 
        printfn "Error: %A" e
        1