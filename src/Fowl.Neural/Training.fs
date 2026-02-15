namespace Fowl.Neural

open System
open Fowl
open Fowl.Core.Types

/// <summary>Training utilities for neural networks.
/// </summary>
module Training =
    
    /// <summary>Training metrics for monitoring progress.
    /// </summary>
type TrainingMetrics = {
        Epoch: int
        Step: int
        Loss: float
        Accuracy: float option
    }
    
    /// <summary>Training configuration.
    /// </summary>
type TrainingConfig = {
        Epochs: int
        BatchSize: int
        LearningRate: float
        Momentum: float
        Verbose: bool
    }
    
    /// <summary>Default training configuration.
    /// </summary>
let defaultConfig = {
        Epochs = 10
        BatchSize = 32
        LearningRate = 0.01
        Momentum = 0.9
        Verbose = true
    }
    
    /// <summary>Simple training loop for a single layer model.
    /// </summary>
    /// <param name="model">Dense layer model.</param>
    /// <param name="lossFn">Function to compute loss node.</param>
    /// <param name="optimizer">Optimizer state.</param>
    /// <param name="xTrain">Training inputs [n_samples, n_features].</param>
    /// <param name="yTrain">Training targets [n_samples, n_outputs].</param>
    /// <param name="config">Training configuration.</param>
    /// <returns>Training metrics history.</returns>
let train (model: DenseLayer)
              (lossFn: Node -> Node -> Node)
              (optimizer: Optimizer.SGD)
              (xTrain: float[,])
              (yTrain: float[,])
              (config: TrainingConfig) : TrainingMetrics list =
        
        let nSamples = xTrain.GetLength(0)
        let nFeatures = xTrain.GetLength(1)
        let nOutputs = yTrain.GetLength(1)
        
        let inputNode = Graph.input "x" [|nFeatures|]
        let targetNode = Graph.input "y" [|nOutputs|]
        
        // Build computation graph
        match Layers.forwardDense model inputNode with
        | Error e -> 
            printfn "Error building model: %A" e
            []
        | Ok outputNode ->
            let lossNode = lossFn outputNode targetNode
            let parameters = Layers.getParameters model
            
            let mutable metrics = []
            
            for epoch = 1 to config.Epochs do
                let mutable epochLoss = 0.0
                let nBatches = (nSamples + config.BatchSize - 1) / config.BatchSize
                
                for batchIdx = 0 to nBatches - 1 do
                    let startIdx = batchIdx * config.BatchSize
                    let endIdx = min (startIdx + config.BatchSize) nSamples
                    let batchSize = endIdx - startIdx
                    
                    if batchSize > 0 then
                        // Extract batch data
                        let batchX = Array.init (batchSize * nFeatures) (fun i ->
                            let row = startIdx + i / nFeatures
                            let col = i % nFeatures
                            if row < nSamples then xTrain.[row, col] else 0.0)
                        
                        let batchY = Array.init (batchSize * nOutputs) (fun i ->
                            let row = startIdx + i / nOutputs
                            let col = i % nOutputs
                            if row < nSamples then yTrain.[row, col] else 0.0)
                        
                        // Set input values
                        inputNode.Value <- Some batchX
                        targetNode.Value <- Some batchY
                        
                        // Forward pass
                        match Forward.run [lossNode] with
                        | Error e -> 
                            printfn "Forward pass error: %A" e
                        | Ok () ->
                            match lossNode.Value with
                            | Some lossVal ->
                                let loss = lossVal.[0] / float batchSize
                                epochLoss <- epochLoss + loss
                                
                                // Backward pass
                                match Backward.run [lossNode] with
                                | Error e -> printfn "Backward pass error: %A" e
                                | Ok () ->
                                    // Update parameters
                                    Optimizer.updateSGD optimizer parameters
                            | None -> ()
                
                let avgLoss = epochLoss / float nBatches
                if config.Verbose then
                    printfn "Epoch %d/%d - Loss: %.4f" epoch config.Epochs avgLoss
                
                metrics <- { Epoch = epoch; Step = 0; Loss = avgLoss; Accuracy = None } :: metrics
            
            List.rev metrics
    
    /// <summary>Evaluate model on test data.
    /// </summary>
let evaluate (model: DenseLayer) (xTest: float[,]) (yTest: float[,]) : float =
        let nSamples = xTest.GetLength(0)
        let nFeatures = xTest.GetLength(1)
        
        let inputNode = Graph.input "x" [|nFeatures|]
        
        match Layers.forwardDense model inputNode with
        | Error _ -> infinity
        | Ok outputNode ->
            let mutable totalLoss = 0.0
            
            for i = 0 to nSamples - 1 do
                let x = Array.init nFeatures (fun j -> xTest.[i, j])
                inputNode.Value <- Some x
                
                match Forward.run [outputNode] with
                | Ok () ->
                    match outputNode.Value with
                    | Some pred ->
                        let target = Array.init (yTest.GetLength(1)) (fun j -> yTest.[i, j])
                        // MSE
                        let loss = 
                            Array.map2 (fun p t -> (p - t) ** 2.0) pred target
                            |> Array.average
                        totalLoss <- totalLoss + loss
                    | None -> ()
                | Error _ -> ()
            
            totalLoss / float nSamples
    
    /// <summary>Simple linear regression test.
    /// Verifies the neural network infrastructure works.
    /// </summary>
let testLinearRegression() : unit =
        printfn "Testing linear regression..."
        
        // Generate synthetic data: y = 2x + 1 + noise
        let nSamples = 100
        let rng = Random(42)
        let xData = Array2D.init nSamples 1 (fun _ _ -> rng.NextDouble() * 10.0)
        let yData = Array2D.init nSamples 1 (fun i _ -> 2.0 * xData.[i, 0] + 1.0 + (rng.NextDouble() - 0.5) * 0.5)
        
        // Create model: 1 input -> 1 output
        match Layers.dense 1 1 None (Some 42) with
        | Error e -> printfn "Error creating model: %A" e
        | Ok model ->
            let optimizer = Optimizer.sgd 0.01 0.0
            let config = { defaultConfig with Epochs = 100; Verbose = false }
            
            printfn "Training..."
            let metrics = train model Loss.mse optimizer xData yData config
            
            // Print final loss
            match List.tryLast metrics with
            | Some m -> printfn "Final loss: %.4f" m.Loss
            | None -> ()
            
            // Check learned parameters
            let weights = model.Weights.Value |> Option.defaultValue [||]
            let bias = model.Bias.Value |> Option.defaultValue [||]
            
            printfn "Learned weight: %.4f (expected ~2.0)" weights.[0]
            printfn "Learned bias: %.4f (expected ~1.0)" bias.[0]
            
            if abs (weights.[0] - 2.0) < 0.2 && abs (bias.[0] - 1.0) < 0.2 then
                printfn "✓ Linear regression test PASSED"
            else
                printfn "✗ Linear regression test FAILED"

/// <summary>Model serialization utilities.
/// </summary>
module Serialization =
    open System.IO
    open System.Text.Json
    
    /// <summary>Save a dense layer to a file.
    /// </summary>
    /// <param name="path">File path to save to.</param>
    /// <param name="model">Dense layer model.</param>
let saveDense (path: string) (model: DenseLayer) : FowlResult<unit> =
        try
            let weights = model.Weights.Value |> Option.defaultValue [||]
            let bias = model.Bias.Value |> Option.defaultValue [||]
            
            let data = {
                InputDim = model.InputDim
                OutputDim = model.OutputDim
                Weights = weights
                Bias = bias
                Activation = model.Activation.ToString()
            }
            
            let json = JsonSerializer.Serialize(data)
            File.WriteAllText(path, json)
            Ok ()
        with
        | ex -> Error.ioError (sprintf "Failed to save model: %s" ex.Message)
    
    /// <summary>Load a dense layer from a file.
    /// </summary>
    /// <param name="path">File path to load from.</param>
    /// <returns>Loaded dense layer.</returns>
let loadDense (path: string) : FowlResult<DenseLayer> =
        try
            let json = File.ReadAllText(path)
            let data = JsonSerializer.Deserialize<{| InputDim: int; OutputDim: int; Weights: float[]; Bias: float[]; Activation: string |}>(json)
            
            let activation = 
                match data.Activation with
                | "ReLU" -> Some ReLU
                | "Sigmoid" -> Some Sigmoid
                | "Tanh" -> Some Tanh
                | "Softmax" -> Some Softmax
                | _ -> None
            
            let model = {
                Weights = Graph.parameter "W" [|data.InputDim; data.OutputDim|] data.Weights
                Bias = Graph.parameter "b" [|data.OutputDim|] data.Bias
                Activation = activation
                InputDim = data.InputDim
                OutputDim = data.OutputDim
            }
            
            Ok model
        with
        | ex -> Error.ioError (sprintf "Failed to load model: %s" ex.Message)