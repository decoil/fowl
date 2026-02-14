/// Linear Regression Example
/// Predicting housing prices from features

module LinearRegressionExample

open Fowl
open Fowl.Core
open Fowl.Stats
open Fowl.Linalg

/// Sample housing data: [SqFt, Bedrooms, Bathrooms] -> Price
let trainingData = [|
    ([|1500.0; 3.0; 2.0|], 250000.0)
    ([|2000.0; 4.0; 2.5|], 320000.0)
    ([|1200.0; 2.0; 1.5|], 180000.0)
    ([|1800.0; 3.0; 2.0|], 280000.0)
    ([|2200.0; 4.0; 3.0|], 350000.0)
    ([|1600.0; 3.0; 2.0|], 260000.0)
    ([|1400.0; 2.0; 1.5|], 200000.0)
    ([|1900.0; 3.0; 2.5|], 300000.0)
    ([|2100.0; 4.0; 2.5|], 330000.0)
    ([|1700.0; 3.0; 2.0|], 270000.0)
|]

/// Normalize features to zero mean, unit variance
let normalizeFeatures (features: float[][]) : float[][] * (float[] * float[]) =
    let nFeatures = features.[0].Length
    
    let means = 
        Array.init nFeatures (fun j -
            features |> Array.averageBy (fun row -> row.[j]))
    
    let stds =
        Array.init nFeatures (fun j -
            let vals = features |> Array.map (fun row -> row.[j])
            match Descriptive.std vals with
            | Ok s -> max s 1e-8
            | Error _ -> 1.0)
    
    let normalized =
        features |> Array.map (fun row -
            row |> Array.mapi (fun i x -
                (x - means.[i]) / stds.[i]))
    
    normalized, (means, stds)

/// Fit linear regression using normal equations
let fitLinearRegression (X: float[][]) (y: float[]) : FowlResult<float[]> =
    result {
        // Add bias column
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

/// Calculate R-squared
let calculateRSquared (actual: float[]) (predicted: float[]) : float =
    let meanActual = Array.average actual
    let ssTot = actual |> Array.sumBy (fun y -> (y - meanActual) ** 2.0)
    let ssRes = Array.zip actual predicted |> Array.sumBy (fun (y, yHat) -> (y - yHat) ** 2.0)
    1.0 - (ssRes / ssTot)

/// Make prediction
let predict (coeffs: float[]) (features: float[]) : float =
    coeffs.[0] + Array.sumBy (fun i -> coeffs.[i+1] * features.[i]) [|0..features.Length-1|]

/// Run regression example
let runRegression() : FowlResult<unit> =
    result {
        printfn "=== Linear Regression Example ==="
        printfn "Predicting house prices from features"
        printfn ""
        
        // Extract features and targets
        let features = trainingData |> Array.map fst
        let prices = trainingData |> Array.map snd
        
        printfn "Training samples: %d" trainingData.Length
        printfn "Features: Square footage, Bedrooms, Bathrooms"
        printfn ""
        
        // Normalize features
        let normalizedFeatures, (means, stds) = normalizeFeatures features
        
        // Train model
        let! coefficients = fitLinearRegression normalizedFeatures prices
        
        printfn "Model Coefficients:"
        printfn "  Bias: %.2f" coefficients.[0]
        printfn "  SqFt: %.2f" coefficients.[1]
        printfn "  Bedrooms: %.2f" coefficients.[2]
        printfn "  Bathrooms: %.2f" coefficients.[3]
        printfn ""
        
        // Make predictions
        let predictions = normalizedFeatures |> Array.map (predict coefficients)
        
        // Evaluate
        let r2 = calculateRSquared prices predictions
        let rmse = 
            Array.zip prices predictions
            |> Array.map (fun (y, yHat) -> (y - yHat) ** 2.0)
            |> Array.average
            |> sqrt
        
        printfn "Model Performance:"
        printfn "  R-squared: %.4f" r2
        printfn "  RMSE: $%.2f" rmse
        printfn ""
        
        // Test prediction
        let testHouse = [|1750.0; 3.0; 2.0|]
        let normalizedTest = testHouse |> Array.mapi (fun i x -> (x - means.[i]) / stds.[i])
        let predictedPrice = predict coefficients normalizedTest
        
        printfn "Prediction for 1750 sqft, 3 bed, 2 bath:"
        printfn "  Predicted price: $%.2f" predictedPrice
        printfn ""
        
        // Statistical significance test
        let! tResult = HypothesisTests.ttestOneSample coefficients.[1..] 0.0 0.05
        printfn "Coefficient Significance Test:"
        printfn "  Features are significant: %b" tResult.Significant
        printfn "  (p-value: %.4f)" tResult.PValue
        printfn ""
        
        printfn "=== Regression Complete ==="
        
        return ()
    }

[<EntryPoint>]
let main argv =
    match runRegression() with
    | Ok () -> 0
    | Error e -> 
        printfn "Error: %A" e
        1