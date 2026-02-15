// TimeSeriesForecasting.fsx
// LSTM-based time series prediction for stock prices

#r "nuget: Fowl"
#r "nuget: FSharp.Data"

open System
open Fowl
open Fowl.Core.Types
open Fowl.Neural
open Fowl.Neural.Graph
open Fowl.Neural.Layers
open Fowl.Neural.Forward
open Fowl.Neural.Backward
open Fowl.Neural.Loss
open Fowl.Neural.Optimizer
open Fowl.Stats

/// Generate synthetic stock price data (random walk with trend)
let generateStockData (n: int) (seed: int) =
    let rng = Random(seed)
    let prices = Array.zeroCreate n
    prices.[0] <- 100.0
    
    for i = 1 to n - 1 do
        let drift = 0.001
        let volatility = 0.02
        let shock = rng.NextDouble() * 2.0 - 1.0
        prices.[i] <- prices.[i-1] * (1.0 + drift + volatility * shock)
    
    prices

/// Normalize data to [0, 1] range
let normalize (data: float[]) =
    let minVal = Array.min data
    let maxVal = Array.max data
    let range = maxVal - minVal
    
    data
    |> Array.map (fun x -> (x - minVal) / range),
    (minVal, maxVal)

/// Denormalize back to original scale
let denormalize (data: float[]) (minVal: float, maxVal: float) =
    data |> Array.map (fun x -> x * (maxVal - minVal) + minVal)

/// Create sequences for LSTM: (input_window, target)
let createSequences (data: float[]) (windowSize: int) =
    let n = data.Length - windowSize
    [|
        for i = 0 to n - 1 do
            let input = data.[i .. i + windowSize - 1]
            let target = data.[i + windowSize]
            yield (input, target)
    |]

/// Build LSTM model for time series
let buildLSTMModel (windowSize: int) =
    result {
        // LSTM layer: input -> hidden
        let! lstm = RecurrentLayers.LSTM.create windowSize 50 (Some 42)
        
        // Output layer: hidden -> 1 (predict next value)
        let! output = Layers.dense 50 1 None (Some 43)
        
        let forward (input: Node) =
            result {
                // LSTM expects [batch; time; features]
                let reshaped = input  // Shape: [windowSize]
                let! lstmOut = RecurrentLayers.LSTM.forward lstm reshaped
                // Take last output
                let lastOutput = lstmOut  // Simplified
                let! prediction = Layers.forwardDense output lastOutput
                return prediction
            }
        
        return (lstm, output, forward)
    }

/// Train the model
let trainLSTM 
    (model) 
    (trainData: (float[] * float)[]) 
    (epochs: int) 
    (learningRate: float) =
    
    result {
        let (lstm, output, forward) = model
        
        // Create graph
        let input = input "sequence" [||]
        let target = input "target" [||]
        
        let! prediction = forward input
        let! loss = Loss.mse prediction target
        
        let optimizer = Optimizer.adam learningRate 0.9 0.999 1e-8
        
        // Training loop
        for epoch = 1 to epochs do
            let mutable totalLoss = 0.0
            
            for (seq, tgt) in trainData do
                // Reset gradients
                let params = 
                    Layers.getParameters lstm @ Layers.getParameters output
                params |> List.iter (fun (p, _) -> p.Grad.Value <- None)
                
                // Forward
                let inputs = Map ["sequence", seq; "target", [|tgt|]]
                do! Forward.runWithInputs loss inputs
                
                // Backward
                do! Backward.run [loss]
                
                // Update
                Optimizer.updateAdam optimizer params
                
                // Track loss
                match loss.Value.Value with
                | Some l -> totalLoss <- totalLoss + l
                | None -> ()
            
            if epoch % 10 = 0 then
                printfn "Epoch %d: Loss = %.6f" epoch (totalLoss / float trainData.Length)
        
        return model
    }

/// Make predictions
let predict (model) (data: float[]) =
    result {
        let (_, _, forward) = model
        
        let input = input "sequence" [||]
        let! prediction = forward input
        
        let inputs = Map ["sequence", data]
        do! Forward.runWithInputs prediction inputs
        
        match prediction.Value.Value with
        | Some p -> return p.[0]
        | None -> return! Error.invalidState "No prediction"
    }

/// Walk-forward validation
let walkForwardValidation (model) (data: float[]) (windowSize: int) =
    result {
        let predictions = ResizeArray<float>()
        let actuals = ResizeArray<float>()
        
        for i = windowSize to data.Length - 1 do
            let window = data.[i - windowSize .. i - 1]
            let! pred = predict model window
            predictions.Add(pred)
            actuals.Add(data.[i])
        
        // Calculate metrics
        let predArray = predictions.ToArray()
        let actualArray = actuals.ToArray()
        
        let mse = 
            Array.map2 (fun p a -> (p - a) ** 2.0) predArray actualArray
            |> Array.average
        
        let mae =
            Array.map2 (fun p a -> abs (p - a)) predArray actualArray
            |> Array.average
        
        return (mse, mae, predArray, actualArray)
    }

/// Main execution
let runExample () =
    result {
        printfn "=== Time Series Forecasting with LSTM ==="
        
        // Generate data
        let data = generateStockData 1000 42
        printfn "Generated %d data points" data.Length
        
        // Normalize
        let normalizedData, normParams = normalize data
        printfn "Data normalized"
        
        // Split train/test
        let splitPoint = int (0.8 * float data.Length)
        let trainDataRaw = normalizedData.[.. splitPoint - 1]
        let testDataRaw = normalizedData.[splitPoint ..]
        
        // Create sequences
        let windowSize = 20
        let trainSequences = createSequences trainDataRaw windowSize
        printfn "Created %d training sequences" trainSequences.Length
        
        // Build model
        let! model = buildLSTMModel windowSize
        printfn "Model built"
        
        // Train
        printfn "\nTraining..."
        let! trainedModel = trainLSTM model trainSequences 100 0.001
        
        // Evaluate
        printfn "\nEvaluating..."
        let! (mse, mae, predictions, actuals) = 
            walkForwardValidation trainedModel testDataRaw windowSize
        
        printfn "\nTest Results:"
        printfn "  MSE: %.6f" mse
        printfn "  MAE: %.6f" mae
        printfn "  RMSE: %.6f" (sqrt mse)
        
        // Denormalize first few predictions
        let denormPreds = denormalize predictions.[..4] normParams
        let denormActuals = denormalize actuals.[..4] normParams
        
        printfn "\nSample predictions vs actual:"
        for i = 0 to 4 do
            printfn "  Pred: %.2f, Actual: %.2f" denormPreds.[i] denormActuals.[i]
        
        return trainedModel
    }

// Run the example
match runExample () with
| Ok _ -> printfn "\nExample completed successfully!"
| Error e -> printfn "\nError: %A" e
