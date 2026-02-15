// MNISTClassifier.fsx
// Complete MNIST handwritten digit classifier

#r "nuget: Fowl"

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

printfn "=== MNIST Digit Classifier ==="

/// Simplified MNIST-like data generator (replace with real MNIST loading)
let generateFakeMNIST (numSamples: int) =
    let rng = Random(42)
    [|
        for i in 1..numSamples do
            // Random 28x28 "image" and random label 0-9
            let image = Array.init 784 (fun _ -> rng.NextDouble())
            let label = rng.Next(10)
            let oneHot = Array.init 10 (fun j -> if j = label then 1.0 else 0.0)
            (image, oneHot, label)
    |]

/// Build CNN model for MNIST
let buildModel () =
    result {
        // Conv block 1: 1 -> 32 channels
        let! conv1 = Conv2D.create 1 32 3 Padding.Same (Some 42)
        let pool1 = Pool2D.create MaxPool 2 2 Padding.Valid
        
        // Conv block 2: 32 -> 64 channels
        let! conv2 = Conv2D.create 32 64 3 Padding.Same (Some 43)
        let pool2 = Pool2D.create MaxPool 2 2 Padding.Valid
        
        // Flatten and dense layers
        // After 2 poolings: 28 -> 14 -> 7
        // 7 * 7 * 64 = 3136
        let! fc1 = Layers.dense 3136 128 (Some ReLU) (Some 44)
        let! output = Layers.dense 128 10 (Some Softmax) (Some 45)
        
        let forward (input: Node) =
            result {
                // Conv block 1
                let! h1 = Conv2D.forward conv1 input
                let h1_act = Graph.activate ReLU h1
                let! h2 = Pool2D.forward pool1 h1_act
                
                // Conv block 2
                let! h3 = Conv2D.forward conv2 h2
                let h3_act = Graph.activate ReLU h3
                let! h4 = Pool2D.forward pool2 h3_act
                
                // Flatten
                let flat = Graph.reshape [|1; 3136|] h4
                
                // Dense layers
                let! h5 = Layers.forwardDense fc1 flat
                let! out = Layers.forwardDense output h5
                
                return out
            }
        
        return (conv1, conv2, fc1, output, forward)
    }

/// Training function
let train (model) (trainData: (float[] * float[] * int)[]) (epochs: int) (batchSize: int) =
    result {
        let (conv1, conv2, fc1, output, forward) = model
        
        // Create graph
        let input = input "image" [|1; 28; 28; 1|]
        let target = input "label" [|10|]
        
        let! prediction = forward input
        let! loss = Loss.crossEntropy prediction target
        
        // Get all parameters
        let params = 
            Layers.getParameters conv1 @
            Layers.getParameters conv2 @
            Layers.getParameters fc1 @
            Layers.getParameters output
        
        let optimizer = Optimizer.adam 0.001 0.9 0.999 1e-8
        
        // Training loop
        for epoch = 1 to epochs do
            let mutable totalLoss = 0.0
            let mutable correct = 0
            let mutable total = 0
            
            // Shuffle data each epoch
            let rng = Random(epoch)
            let shuffled = trainData |> Array.sortBy (fun _ -> rng.Next())
            
            // Process in batches
            for batchStart in 0..batchSize..(shuffled.Length - batchSize) do
                let batch = shuffled.[batchStart .. min (batchStart + batchSize - 1) (shuffled.Length - 1)]
                
                // Accumulate gradients
                params |> List.iter (fun (p, _) -> p.Grad.Value <- None)
                
                let mutable batchLoss = 0.0
                
                for (image, label, _) in batch do
                    let inputs = Map ["image", image; "label", label]
                    
                    // Forward
                    do! Forward.runWithInputs loss inputs
                    batchLoss <- batchLoss + (loss.Value.Value |> Option.defaultValue 0.0)
                    
                    // Check accuracy
                    let predIdx =
                        match prediction.Value.Value with
                        | Some probs -> probs |> Array.mapi (fun i p -> (i, p)) |> Array.maxBy snd |> fst
                        | None -> -1
                    let trueIdx = label |> Array.mapi (fun i v -> (i, v)) |> Array.maxBy snd |> fst
                    
                    if predIdx = trueIdx then
                        correct <- correct + 1
                    total <- total + 1
                
                // Backward (average gradients)
                do! Backward.run [loss]
                
                // Scale gradients by batch size
                for (p, _) in params do
                    match p.Grad.Value with
                    | Some grad ->
                        for i = 0 to grad.Length - 1 do
                            grad.[i] <- grad.[i] / float batch.Length
                    | None -> ()
                
                // Update parameters
                Optimizer.updateAdam optimizer params
                
                totalLoss <- totalLoss + batchLoss
            
            let avgLoss = totalLoss / float (shuffled.Length / batchSize)
            let accuracy = float correct / float total * 100.0
            
            if epoch % 5 = 0 then
                printfn "Epoch %d/%d: Loss = %.4f, Accuracy = %.2f%%" epoch epochs avgLoss accuracy
        
        return model
    }

/// Evaluation function
let evaluate (model) (testData: (float[] * float[] * int)[]) =
    result {
        let (_, _, _, _, forward) = model
        
        let input = input "image" [|1; 28; 28; 1|]
        let! prediction = forward input
        
        let mutable correct = 0
        
        for (image, _, trueLabel) in testData do
            let inputs = Map ["image", image]
            do! Forward.runWithInputs prediction inputs
            
            let predLabel =
                match prediction.Value.Value with
                | Some probs -> probs |> Array.mapi (fun i p -> (i, p)) |> Array.maxBy snd |> fst
                | None -> -1
            
            if predLabel = trueLabel then
                correct <- correct + 1
        
        let accuracy = float correct / float testData.Length * 100.0
        printfn "Test Accuracy: %.2f%% (%d/%d)" accuracy correct testData.Length
        
        return accuracy
    }

/// Main execution
let runExample () =
    result {
        printfn "\nGenerating synthetic MNIST data..."
        let trainData = generateFakeMNIST 1000
        let testData = generateFakeMNIST 200
        printfn "Training: %d samples, Test: %d samples" trainData.Length testData.Length
        
        printfn "\nBuilding model..."
        let! model = buildModel ()
        printfn "Model built successfully"
        
        printfn "\nTraining..."
        let! trainedModel = train model trainData 20 32
        printfn "Training complete"
        
        printfn "\nEvaluating..."
        let! accuracy = evaluate trainedModel testData
        
        return trainedModel
    }

// Run the example
match runExample () with
| Ok _ -> printfn "\n=== MNIST Example Complete ==="
| Error e -> printfn "\nError: %A" e
