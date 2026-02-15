# Tutorial 4: Neural Networks

Build and train neural networks with Fowl's computation graph and automatic differentiation.

## Overview

Fowl provides a flexible neural network framework based on computation graphs and automatic differentiation. This tutorial covers building, training, and evaluating neural networks.

## Learning Objectives

- Understand computation graphs
- Build neural network architectures
- Train with backpropagation
- Use optimizers and loss functions
- Implement common architectures

## Setup

```fsharp
open Fowl
open Fowl.Core.Types
open Fowl.Neural
open Fowl.Neural.Graph
open Fowl.Neural.Layers
open Fowl.Neural.Optimizer
open Fowl.Neural.Loss

let unwrap = function Ok x -> x | Error e -> failwith e.Message
```

## Computation Graphs

### Building a Graph

```fsharp
// Create input nodes
let x = input "x" [|784|]  // 28x28 flattened image
let y = input "y" [|10|]   // One-hot label

// Create parameters
let W1 = parameter "W1" [|784; 256|] (Array.init 200704 (fun _ - 
    0.01 * (Random.Shared.NextDouble() * 2.0 - 1.0)))
let b1 = constantArray (Array.zeroCreate 256) [|256|]

let W2 = parameter "W2" [|256; 10|] (Array.init 2560 (fun _ - 
    0.01 * (Random.Shared.NextDouble() * 2.0 - 1.0)))
let b2 = constantArray (Array.zeroCreate 10) [|10|]

// Build forward pass
let h1 = activate ReLU (add (matmul x W1) b1)
let logits = add (matmul h1 W2) b2
let output = activate Softmax logits

// Define loss
let loss = Loss.crossEntropy output y
```

### Forward Pass

```fsharp
// Prepare input data
let inputs = Map.empty
let inputsWithData = Map.add "x" (Array.init 784 (fun _ - 0.5)) inputs
let inputData = Map.add "y" (Array.init 10 (fun i - if i = 5 then 1.0 else 0.0)) inputData

// Forward pass
Forward.run [loss] |> ignore

// Get output
let outputValue = output.Value
printfn "Predictions: %A" outputValue
```

### Backward Pass

```fsharp
// Compute gradients
Backward.run [loss] |> ignore

// Access gradients
let dW1 = W1.Grad
let db1 = b1.Grad
let dW2 = W2.Grad
let db2 = b2.Grad

printfn "Gradient norms: W1=%.4f, W2=%.4f" 
    (Array.map abs dW1 |> Array.average) 
    (Array.map abs dW2 |> Array.average)
```

## Building Layers

### Dense (Fully Connected) Layers

```fsharp
// Create a dense layer
let layer = dense 784 256 (Some ReLU) (Some 42) |> unwrap

// Forward pass through layer
let input = Array.init 784 (fun _ - 0.1)
let output = forwardDense layer (constantArray input [|784|]) |> unwrap
let outputArray = output.Value.Value
```

### Multi-Layer Perceptron

```fsharp
// Build a 3-layer MLP
let inputSize = 784
let hidden1Size = 256
let hidden2Size = 128
let outputSize = 10

let layer1 = dense inputSize hidden1Size (Some ReLU) (Some 42) |> unwrap
let layer2 = dense hidden1Size hidden2Size (Some ReLU) (Some 42) |> unwrap
let layer3 = dense hidden2Size outputSize (Some Softmax) (Some 42) |> unwrap

let x = input "x" [|inputSize|]
let y = input "y" [|outputSize|]

let h1 = forwardDense layer1 x |> unwrap
let h2 = forwardDense layer2 h1 |> unwrap
let output = forwardDense layer3 h2 |> unwrap

let loss = Loss.crossEntropy output y
```

## Optimizers

### SGD (Stochastic Gradient Descent)

```fsharp
let parameters = [|W1; b1; W2; b2|]
let optimizer = Optimizer.sgd 0.01 0.0  // learning rate, momentum

for epoch = 1 to 10 do
    // Forward and backward
    Forward.run [loss] |> ignore
    Backward.run [loss] |> ignore
    
    // Update parameters
    Optimizer.updateSGD optimizer parameters
    
    printfn "Epoch %d: Loss = %.4f" epoch loss.Value.Value
```

### Adam Optimizer

```fsharp
let parameters = [|W1; b1; W2; b2|]
let optimizer = Optimizer.adam 0.001 0.9 0.999 1e-8  // lr, beta1, beta2, epsilon

for epoch = 1 to 10 do
    Forward.run [loss] |> ignore
    Backward.run [loss] |> ignore
    
    Optimizer.updateAdam optimizer parameters
    
    printfn "Epoch %d: Loss = %.4f" epoch loss.Value.Value
```

### RMSprop

```fsharp
let optimizer = Optimizer.rmsprop 0.001 0.9 1e-8  // lr, alpha, epsilon

// Update loop same as above
Optimizer.updateRMSprop optimizer parameters
```

## Loss Functions

### Cross-Entropy Loss

```fsharp
// For classification with one-hot labels
let loss = Loss.crossEntropy logits y
```

### Mean Squared Error

```fsharp
// For regression tasks
let predictions = input "preds" [|1|]
let targets = input "targets" [|1|]
let mseLoss = Loss.mse predictions targets
```

### Binary Cross-Entropy

```fsharp
// For binary classification
let probs = activate Sigmoid logits
let binaryLoss = Loss.binaryCrossEntropy probs y
```

## Complete Training Loop

```fsharp
// Build model
let input = input "x" [|784|]
let target = input "y" [|10|]

let W1 = parameter "W1" [|784; 256|] (Xavier.initializer [|784; 256|] |> unwrap)
let b1 = constantArray (Array.zeroCreate 256) [|256|]
let W2 = parameter "W2" [|256; 10|] (Xavier.initializer [|256; 10|] |> unwrap)
let b2 = constantArray (Array.zeroCreate 10) [|10|]

let h1 = activate ReLU (add (matmul input W1) b1)
let logits = add (matmul h1 W2) b2
let output = activate Softmax logits
let loss = Loss.crossEntropy output target

// Setup optimizer
let parameters = [|W1; b1; W2; b2|]
let optimizer = Optimizer.adam 0.001 0.9 0.999 1e-8

// Training configuration
let epochs = 10
let batchSize = 32

for epoch = 1 to epochs do
    let mutable totalLoss = 0.0
    let mutable numBatches = 0
    
    for batchStart = 0 to 1000 - batchSize step batchSize do
        // Prepare batch data (simplified)
        let batchX = Array.init batchSize (fun _ - 
            Array.init 784 (fun _ - Random.Shared.NextDouble()))
        let batchY = Array.init batchSize (fun _ - 
            let label = Random.Shared.Next(10)
            Array.init 10 (fun i - if i = label then 1.0 else 0.0))
        
        // Train on batch
        for i = 0 to batchSize - 1 do
            Forward.runWithInputs loss 
                (Map.empty |> Map.add "x" batchX.[i] |> Map.add "y" batchY.[i]) 
            |> ignore
            Backward.run [loss] |> ignore
        
        // Update
        Optimizer.updateAdam optimizer parameters
        
        totalLoss <- totalLoss + loss.Value.Value
        numBatches <- numBatches + 1
    
    printfn "Epoch %d: Average Loss = %.4f" epoch (totalLoss / float numBatches)
```

## Recurrent Neural Networks

### LSTM

```fsharp
open Fowl.Neural.RecurrentLayers

// Create LSTM layer
let lstm = createLSTM inputSize=10 hiddenSize=20 numLayers=2 dropout=0.1 seed=42 |> unwrap

// Forward pass through sequence
let sequence = Array.init 50 (fun _ - 
    Array.init 32 (fun _ - 
        Array.init 10 (fun _ - Random.Shared.NextDouble())))

let outputs = lstmForward lstm sequence

printfn "Output shape: %A" outputs.[49].Length  // Last time step
```

### GRU

```fsharp
// GRU (simpler alternative to LSTM)
let gru = createGRU inputSize=10 hiddenSize=20 numLayers=1 dropout=0.1 seed=42 |> unwrap

let outputs = gruForward gru sequence
```

### Many-to-Many Architecture

```fsharp
// Process sequences with LSTM and classify each time step
let input = input "seq" [|32; 10|]  // batch_size x time_steps x features

let lstmLayer = createLSTM 10 20 numLayers=1 |> unwrap
let hidden = gruForward lstmLayer (Array.init 32 (fun _ - [|...|]))

let denseLayer = dense 20 5 (Some Softmax) (Some 42) |> unwrap
let outputs = Array.map (fun h - forwardDense denseLayer (constantArray h [|20|]) |> unwrap) hidden
```

## Practical Example: MNIST Digit Classification

```fsharp
// Simplified MNIST training

// Network architecture: 784 -> 256 -> 10
let input = input "x" [|784|]
let target = input "y" [|10|]

let W1 = parameter "W1" [|784; 256|] (Xavier.initializer [|784; 256|] |> unwrap)
let b1 = constantArray (Array.zeroCreate 256) [|256|]
let W2 = parameter "W2" [|256; 10|] (Xavier.initializer [|256; 10|] |> unwrap)
let b2 = constantArray (Array.zeroCreate 10) [|10|]

let h1 = activate ReLU (add (matmul input W1) b1)
let logits = add (matmul h1 W2) b2
let output = activate Softmax logits
let loss = Loss.crossEntropy output target

// Training
let optimizer = Optimizer.adam 0.001 0.9 0.999 1e-8
let parameters = [|W1; b1; W2; b2|]

for epoch = 1 to 5 do
    let mutable correct = 0
    let mutable total = 0
    
    for batch = 0 to 60000 / 100 - 1 do
        // Load batch (simplified)
        let batchX = [|for _ in 0..99 - do Array.init 784 (fun _ - Random.Shared.NextDouble())|]
        let batchY = [|for _ in 0..99 - do 
            let label = Random.Shared.Next(10)
            Array.init 10 (fun i - if i = label then 1.0 else 0.0)|]
        
        // Train
        for i = 0 to 99 - 1 do
            Forward.runWithInputs loss 
                (Map.empty |> Map.add "x" batchX.[i] |> Map.add "y" batchY.[i]) 
            |> ignore
            Backward.run [loss] |> ignore
        
        Optimizer.updateAdam optimizer parameters
        
        // Accuracy
        for i = 0 to 99 - 1 do
            Forward.runWithInputs output (Map.add "x" batchX.[i]) |> ignore
            let pred = Array.findIndex (fun x - x = Array.max output.Value.Value) output.Value.Value
            let actual = Array.findIndex (fun x - x = 1.0) batchY.[i]
            if pred = actual then correct <- correct + 1
            total <- total + 1
    
    printfn "Epoch %d: Accuracy = %.2f%%" epoch (float correct / float total * 100.0)
```

## Evaluation and Metrics

```fsharp
// Compute accuracy
let computeAccuracy model (samples: (float[] * float[])[]) =
    let mutable correct = 0
    for input, target in samples do
        Forward.runWithInputs model (Map.add "x" input) |> ignore
        let pred = Array.findIndex (fun x - x = Array.max model.Value.Value) model.Value.Value
        let actual = Array.findIndex (fun x - x = 1.0) target
        if pred = actual then correct <- correct + 1
    float correct / float samples.Length

// Confusion matrix
let confusionMatrix predictions targets numClasses =
    let confusion = Array2D.create numClasses numClasses 0
    for i = 0 to predictions.Length - 1 do
        let pred = predictions.[i]
        let actual = targets.[i]
        confusion.[pred, actual] <- confusion.[pred, actual] + 1
    confusion
```

## Common Architectures

### Simple CNN (Convolutional)

```fsharp
// Note: Conv2D implementation is in progress
// Placeholder for future API

let convLayer = conv2D inputChannels=1 outputChannels=32 kernelSize=3 stride=1 padding="same" |> unwrap
let poolLayer = maxPool2D poolSize=2 stride=2 |> unwrap
```

### ResNet Block

```fsharp
let resNetBlock input channels =
    let conv1 = conv2D inputChannels=channels outputChannels=channels kernelSize=3 stride=1 padding="same" |> unwrap
    let conv2 = conv2D inputChannels=channels outputChannels=channels kernelSize=3 stride=1 padding="same" |> unwrap
    
    let h1 = activate ReLU (conv1 input)
    let h2 = conv2 h1
    let output = activate ReLU (add h2 input)  // Residual connection
    output
```

## Best Practices

### 1. Weight Initialization

```fsharp
// Xavier initialization for tanh/sigmoid
let weights = Xavier.initializer [|inputSize; outputSize|] |> unwrap

// He initialization for ReLU
let weights = He.initializer [|inputSize; outputSize|] |> unwrap
```

### 2. Regularization

```fsharp
// Dropout (randomly zero activations)
let h = dropout ReLU h1 rate=0.5

// L2 regularization (add to loss)
let l2Loss = L2.regularization [|W1; W2|] lambda=0.001
let totalLoss = add loss l2Loss
```

### 3. Batch Normalization

```fsharp
// Normalize layer inputs
let bn = batchNorm h1
let h2 = activate ReLU bn
```

### 4. Learning Rate Scheduling

```fsharp
// Decay learning rate
let mutable lr = 0.001
for epoch = 1 to epochs do
    if epoch % 10 = 0 then
        lr <- lr * 0.1
    optimizer <- Optimizer.adam lr 0.9 0.999 1e-8
    // ... training loop
```

## Exercises

1. Build a 4-layer MLP for regression
2. Implement early stopping based on validation loss
3. Add L2 regularization to prevent overfitting
4. Build a character-level RNN for text generation
5. Implement a simple CNN for image classification

## Solutions

```fsharp
// Exercise 1: 4-layer MLP for regression
let input = input "x" [|10|]
let target = input "y" [|1|]

let layers = [
    dense 10 64 (Some ReLU) (Some 42)
    dense 64 32 (Some ReLU) (Some 42)
    dense 32 16 (Some ReLU) (Some 42)
    dense 16 1 None (Some 42)
] |> List.map (fun l - l |> unwrap)

let mutable current = input
for layer in layers do
    current <- forwardDense layer current |> unwrap

let loss = Loss.mse current target

// Exercise 2: Early stopping
let mutable bestLoss = System.Double.MaxValue
let mutable patience = 0
let maxPatience = 5

for epoch = 1 to epochs do
    // Train...
    let valLoss = computeValidationLoss model validationSet
    
    if valLoss < bestLoss then
        bestLoss <- valLoss
        patience <- 0
    else
        patience <- patience + 1
        if patience >= maxPatience then
            printfn "Early stopping at epoch %d" epoch
            break

// Exercise 3: L2 regularization
let lambda = 0.001
let l2Loss = 
    [|W1; W2; W3|]
    |> Array.map (fun w - Array.sumBy (fun x - x * x) w.Data)
    |> Array.sum
    |> (*) lambda
let totalLoss = add loss (constantArray [|l2Loss|] [||])
```

## Next Steps

- [Tutorial 5: Signal Processing](Tutorial5_SignalProcessing.md)
- [User Guide](../USER_GUIDE.md#neural-networks)

---

*Estimated time: 60 minutes*