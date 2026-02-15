# Chapter 11: Neural Networks

## 11.1 Introduction

Neural networks are function approximators composed of layers of interconnected nodes. Fowl provides a flexible computation graph framework for building and training neural networks.

## 11.2 Building Blocks

### The Computation Graph

```fsharp
open Fowl.Neural
open Fowl.Neural.Graph

// Create a simple computation: y = 2x + 1
let x = input "x" [||]        // Scalar input
let w = parameter "w" [||] [|2.0|]  // Weight = 2.0
let b = parameter "b" [||] [|1.0|]  // Bias = 1.0

let h = Graph.mul x w
let y = Graph.add h b  // y = 2x + 1
```

### Forward Pass

```fsharp
open Fowl.Neural.Forward

// Execute with input value
let inputs = Map ["x", [|3.0|]]
match Forward.runWithInputs y inputs with
| Ok () ->
    match y.Value.Value with
    | Some result -> printfn "2*3 + 1 = %f" result  // 7.0
    | None -> printfn "No value computed"
| Error e -> printfn "Error: %A" e
```

## 11.3 Layers

### Dense Layer

```fsharp
open Fowl.Neural.Layers

// Create a dense layer: 784 inputs -> 256 outputs with ReLU
let layer1 = Layers.dense 784 256 (Some ReLU) (Some 42) |> unwrap

// Another layer: 256 -> 10 with softmax
let outputLayer = Layers.dense 256 10 (Some Softmax) (Some 42) |> unwrap

// Build network
let input = input "image" [|784|]
let! hidden = Layers.forwardDense layer1 input
let! output = Layers.forwardDense outputLayer hidden
```

### Activation Functions

```fsharp
// ReLU: max(0, x)
let relu = Graph.activate ReLU x

// Sigmoid: 1 / (1 + exp(-x))
let sigmoid = Graph.activate Sigmoid x

// Tanh
let tanh = Graph.activate Tanh x

// Softmax (for multi-class classification)
let softmax = Graph.activate Softmax x

// LeakyReLU: max(alpha*x, x)
let leaky = Graph.activate (LeakyReLU 0.1) x
```

## 11.4 Loss Functions

```fsharp
open Fowl.Neural.Loss

// Mean Squared Error (for regression)
let! mseLoss = Loss.mse predictions targets

// Binary Cross Entropy (for binary classification)
let! bceLoss = Loss.binaryCrossEntropy predictions targets

// Cross Entropy (for multi-class classification)
let! ceLoss = Loss.crossEntropy predictions targets
```

## 11.5 Training

### Backward Pass

```fsharp
open Fowl.Neural.Backward

// After forward pass, compute gradients
match Backward.run [loss] with
| Ok () ->
    // Gradients are now stored in parameter nodes
    match w.Grad.Value with
    | Some grad -> printfn "Gradient of w: %A" grad
    | None -> printfn "No gradient computed"
| Error e -> printfn "Error: %A" e
```

### Optimizers

```fsharp
open Fowl.Neural.Optimizer

// SGD
let sgd = Optimizer.sgd 0.01 0.9  // lr=0.01, momentum=0.9

// Adam
let adam = Optimizer.adam 0.001 0.9 0.999 1e-8

// Update parameters
let parameters = Layers.getParameters model
Optimizer.updateSGD sgd parameters
```

### Complete Training Loop

```fsharp
open Fowl.Neural.Training

// Training configuration
let config = {
    Epochs = 100
    BatchSize = 32
    LearningRate = 0.01
    Verbose = true
}

// Train
match Training.train model loss trainingData validationData config with
| Ok history ->
    printfn "Training complete!"
    printfn "Final training loss: %f" (List.last history.TrainLoss)
    printfn "Final validation loss: %f" (List.last history.ValLoss)
| Error e -> printfn "Training failed: %A" e
```

## 11.6 Advanced Architectures

### Multi-Layer Network

```fsharp
let buildMLP (inputSize: int) (hiddenSize: int) (outputSize: int) =
    result {
        let! layer1 = Layers.dense inputSize hiddenSize (Some ReLU) (Some 42)
        let! layer2 = Layers.dense hiddenSize hiddenSize (Some ReLU) (Some 42)
        let! output = Layers.dense hiddenSize outputSize (Some Softmax) (Some 42)
        
        let forward (input: Node) =
            result {
                let! h1 = Layers.forwardDense layer1 input
                let! h2 = Layers.forwardDense layer2 h1
                let! out = Layers.forwardDense output h2
                return out
            }
        
        return (forward, [layer1; layer2; output])
    }
```

### Convolutional Layer (CNN)

```fsharp
open Fowl.Neural.ConvLayers

// Conv2D: 3 input channels, 16 output channels, 3x3 kernel
let conv1 = Conv2D.create 3 16 3 (Padding.Same) (Some 42) |> unwrap

// Max pooling
let pool = Pool2D.create MaxPool 2 2 Padding.Valid

// Forward pass for image
let image = input "image" [|28; 28; 3|]  // 28x28 RGB image
let! convOut = Conv2D.forward conv1 image
let! pooled = Pool2D.forward pool convOut
```

### Recurrent Layer (LSTM)

```fsharp
open Fowl.Neural.RecurrentLayers

// LSTM: input size 100, hidden size 256
let lstm = LSTM.create 100 256 (Some 42) |> unwrap

// Process sequence
let sequence = input "sequence" [|10; 100|]  // 10 timesteps, 100 features
let! outputs = LSTM.forward lstm sequence
```

## 11.7 Complete Example: MNIST Classifier

```fsharp
module MNISTClassifier

open Fowl
open Fowl.Core.Types
open Fowl.Neural
open Fowl.Neural.Graph
open Fowl.Neural.Layers
open Fowl.Neural.Forward
open Fowl.Neural.Backward
open Fowl.Neural.Loss
open Fowl.Neural.Optimizer
open Fowl.Neural.Training

/// Build a simple neural network for MNIST
let buildModel () =
    result {
        // Layer 1: 784 -> 256 with ReLU
        let! hidden1 = Layers.dense 784 256 (Some ReLU) (Some 42)
        
        // Layer 2: 256 -> 128 with ReLU
        let! hidden2 = Layers.dense 256 128 (Some ReLU) (Some 43)
        
        // Output: 128 -> 10 with Softmax
        let! output = Layers.dense 128 10 (Some Softmax) (Some 44)
        
        return (hidden1, hidden2, output)
    }

/// Forward pass through the network
let forward (hidden1, hidden2, output) (input: Node) =
    result {
        let! h1 = Layers.forwardDense hidden1 input
        let! h2 = Layers.forwardDense hidden2 h1
        let! out = Layers.forwardDense output h2
        return out
    }

/// Training function
let trainModel (model) (trainData: (float[] * float[])[]) (epochs: int) =
    result {
        // Create graph nodes
        let input = input "image" [|784|]
        let target = input "label" [|10|]
        
        // Forward pass
        let! predictions = forward model input
        
        // Loss
        let! loss = Loss.crossEntropy predictions target
        
        // Optimizer
        let optimizer = Optimizer.adam 0.001 0.9 0.999 1e-8
        
        // Training loop
        for epoch = 1 to epochs do
            let mutable totalLoss = 0.0
            let mutable count = 0
            
            for (image, label) in trainData do
                // Reset gradients
                let params = 
                    let (h1, h2, out) = model
                    Layers.getParameters h1 @ Layers.getParameters h2 @ Layers.getParameters out
                params |> List.iter (fun (p, _) -> p.Grad.Value <- None)
                
                // Forward
                let inputs = Map ["image", image; "label", label]
                do! Forward.runWithInputs loss inputs
                
                // Backward
                do! Backward.run [loss]
                
                // Update
                Optimizer.updateAdam optimizer params
                
                // Track loss
                match loss.Value.Value with
                | Some l -> totalLoss <- totalLoss + l
                | None -> ()
                count <- count + 1
            
            if epoch % 10 = 0 then
                printfn "Epoch %d: Loss = %.4f" epoch (totalLoss / float count)
        
        return model
    }

/// Inference
let predict (model) (image: float[]) =
    result {
        let input = input "image" [|784|]
        let! predictions = forward model input
        
        let inputs = Map ["image", image]
        do! Forward.runWithInputs predictions inputs
        
        match predictions.Value.Value with
        | Some output -> return output
        | None -> return! Error.invalidState "No output"
    }

// Example usage
let example () =
    result {
        // Build model
        let! model = buildModel ()
        
        // Create dummy training data (in practice, load real MNIST)
        let trainData = [|
            for _ in 1..100 ->
                let image = Array.init 784 (fun _ -> Random.Shared.NextDouble())
                let label = Array.init 10 (fun _ -> if Random.Shared.Next(10) = 0 then 1.0 else 0.0)
                (image, label)
        |]
        
        // Train
        let! trained = trainModel model trainData 50
        
        // Predict
        let testImage = Array.init 784 (fun _ -> Random.Shared.NextDouble())
        let! prediction = predict trained testImage
        
        printfn "Prediction: %A" prediction
        
        return trained
    }
```

## 11.8 Regularization

### Dropout

```fsharp
// During training
let! dropped = Dropout.forward 0.5 hidden  // 50% dropout

// During inference (no dropout)
let! inference = Dropout.forward 0.0 hidden
```

### Batch Normalization

```fsharp
// Batch norm layer
let bn = BatchNorm.create 256  // for 256 features

// Forward pass
let! normalized = BatchNorm.forward bn hidden (isTraining=true)
```

## 11.9 Saving and Loading Models

```fsharp
open System.IO
open System.Text.Json

/// Save model parameters
let saveModel (model: 'a) (path: string) =
    let parameters = Layers.getParameters model
    let weights = 
        parameters
        |> List.map (fun (p, _) -
            match p.Value.Value with
            | Some v -> v
            | None -> 0.0)
    
    let json = JsonSerializer.Serialize(weights)
    File.WriteAllText(path, json)

/// Load model parameters
let loadModel (model: 'a) (path: string) =
    let json = File.ReadAllText(path)
    let weights = JsonSerializer.Deserialize<float[] list>(json)
    
    let parameters = Layers.getParameters model
    List.iter2 (fun (p, _) w -
        p.Value.Value <- Some w) parameters weights
```

## 11.10 Exercises

### Exercise 11.1: XOR Problem

Implement a neural network that learns XOR:

```fsharp
let xorNetwork () =
    result {
        let! hidden = Layers.dense 2 4 (Some Sigmoid) (Some 42)
        let! output = Layers.dense 4 1 (Some Sigmoid) (Some 43)
        
        // XOR training data
        let data = [|
            ([|0.0; 0.0|], [|0.0|])
            ([|0.0; 1.0|], [|1.0|])
            ([|1.0; 0.0|], [|1.0|])
            ([|1.0; 1.0|], [|0.0|])
        |]
        
        // Train until convergence
        // ... training code ...
        
        return (hidden, output)
    }
```

### Exercise 11.2: Autoencoder

Build an autoencoder for dimensionality reduction:

```fsharp
let autoencoder (inputSize: int) (encodingSize: int) =
    result {
        // Encoder
        let! enc1 = Layers.dense inputSize 128 (Some ReLU) (Some 42)
        let! encoder = Layers.dense 128 encodingSize (Some Sigmoid) (Some 43)
        
        // Decoder
        let! dec1 = Layers.dense encodingSize 128 (Some ReLU) (Some 44)
        let! decoder = Layers.dense 128 inputSize (Some Sigmoid) (Some 45)
        
        let forward (input: Node) =
            result {
                let! h1 = Layers.forwardDense enc1 input
                let! encoded = Layers.forwardDense encoder h1
                let! h2 = Layers.forwardDense dec1 encoded
                let! decoded = Layers.forwardDense decoder h2
                return (encoded, decoded)
            }
        
        return forward
    }
```

### Exercise 11.3: Custom Layer

Implement a custom residual block:

```fsharp
let residualBlock (inputSize: int) =
    result {
        let! layer1 = Layers.dense inputSize inputSize (Some ReLU) (Some 42)
        let! layer2 = Layers.dense inputSize inputSize None (Some 43)
        
        let forward (input: Node) =
            result {
                let! h1 = Layers.forwardDense layer1 input
                let! h2 = Layers.forwardDense layer2 h1
                // Residual connection: output = h2 + input
                return! Graph.add h2 input
            }
        
        return forward
    }
```

## 11.11 Best Practices

1. **Always seed random initializers** for reproducibility
2. **Use appropriate activation functions**:
   - ReLU for hidden layers
   - Sigmoid for binary output
   - Softmax for multi-class
   - Linear for regression
3. **Normalize inputs** (zero mean, unit variance)
4. **Monitor validation loss** to detect overfitting
5. **Use learning rate scheduling** for better convergence

## 11.12 Summary

Key concepts:
- Computation graphs define the forward pass
- Backward pass computes gradients via automatic differentiation
- Layers encapsulate weights and forward/backward logic
- Optimizers update parameters based on gradients
- Training loops iterate over data, computing loss and updating weights

---

*Next: [Chapter 12: Deep Learning](chapter12.md)*
