# Chapter 10: Advanced Neural Networks

## 10.1 Modern Architectures

### Residual Networks (ResNet)

```fsharp
// Residual block: output = F(x) + x
open Fowl.Neural

let residualBlock (inputSize: int) =
    result {
        let! conv1 = Conv2D.create inputSize inputSize 3 Padding.Same (Some 42)
        let! bn1 = BatchNorm2D.create inputSize
        let! conv2 = Conv2D.create inputSize inputSize 3 Padding.Same (Some 43)
        let! bn2 = BatchNorm2D.create inputSize
        
        let forward (input: Node) =
            result {
                // First conv block
                let! h1 = Conv2D.forward conv1 input
                let! h1_bn = BatchNorm2D.forward bn1 h1 (isTraining=true)
                let h1_act = Graph.activate ReLU h1_bn
                
                // Second conv block
                let! h2 = Conv2D.forward conv2 h1_act
                let! h2_bn = BatchNorm2D.forward bn2 h2 (isTraining=true)
                
                // Residual connection
                return! Graph.add h2_bn input
            }
        
        return forward
    }

// Skip connections solve vanishing gradients in deep networks
```

### Batch Normalization Deep Dive

```fsharp
// Why batch norm works:
// 1. Reduces internal covariate shift
// 2. Acts as regularization
// 3. Allows higher learning rates

let batchNormForward (bn: BatchNorm2DLayer) (x: Node) (isTraining: bool) =
    result {
        if isTraining then
            // Compute batch statistics
            let! mean = Ndarray.meanAxis x 0
            let! var = Ndarray.varAxis x 0
            
            // Update running statistics
            bn.RunningMean <- Array.map2 (fun run curr -
                bn.Momentum * run + (1.0 - bn.Momentum) * curr)
                bn.RunningMean mean
            
            // Normalize: (x - mean) / sqrt(var + epsilon)
            let! normalized = 
                result {
                    let! centered = Ndarray.sub x mean
                    let! scaled = Ndarray.div centered 
                        (Ndarray.sqrt (Ndarray.add var bn.Epsilon) |> unwrap)
                    return scaled
                }
            
            // Scale and shift: gamma * normalized + beta
            let! scaled = Ndarray.mul normalized (Ndarray.reshape [||] bn.Gamma |> unwrap)
            return! Ndarray.add scaled (Ndarray.reshape [||] bn.Beta |> unwrap)
        else
            // Use running statistics
            // ...
            return x
    }
```

### Dropout and Regularization

```fsharp
// Dropout: randomly zero out neurons during training
let dropout (x: Node) (rate: float) (isTraining: bool) (rng: Random) : Node =
    if not isTraining then
        // At test time: scale by (1 - rate)
        Graph.mul x (Graph.constant (1.0 - rate))
    else
        // At training time: randomly mask
        // This would need custom implementation in graph
        x

// L2 Regularization (Weight Decay)
let l2Loss (parameters: Node list) (lambda: float) : Node =
    parameters
    |> List.map (fun p -> Graph.pow p (Graph.constant 2.0))
    |> List.reduce Graph.add
    |> fun sum -> Graph.mul sum (Graph.constant (lambda / 2.0))

// Total loss = data loss + regularization
let totalLoss = 
    Graph.add dataLoss (l2Loss parameters 0.0001)
```

## 10.2 Recurrent Neural Networks

### LSTM Cell

```fsharp
// LSTM: Long Short-Term Memory
// Solves vanishing gradients in RNNs

let lstmCell (inputSize: int) (hiddenSize: int) =
    result {
        // Gates: input, forget, output
        let! W_i = Layers.dense inputSize hiddenSize (Some Sigmoid) (Some 42)
        let! W_f = Layers.dense inputSize hiddenSize (Some Sigmoid) (Some 43)
        let! W_o = Layers.dense inputSize hiddenSize (Some Sigmoid) (Some 44)
        let! W_c = Layers.dense inputSize hiddenSize (Some Tanh) (Some 45)
        
        let forward (input: Node) (prevHidden: Node) (prevCell: Node) =
            result {
                // Concatenate input and previous hidden
                let! combined = Graph.add input prevHidden
                
                // Gates
                let! i = Layers.forwardDense W_i combined  // Input gate
                let! f = Layers.forwardDense W_f combined  // Forget gate
                let! o = Layers.forwardDense W_o combined  // Output gate
                let! c_tilde = Layers.forwardDense W_c combined  // Candidate
                
                // Cell state: c = f * prev_c + i * c_tilde
                let! f_c = Graph.mul f prevCell
                let! i_c = Graph.mul i c_tilde
                let! cell = Graph.add f_c i_c
                
                // Hidden state: h = o * tanh(c)
                let! hidden = 
                    result {
                        let! tanh_c = Graph.activate Tanh cell
                        return! Graph.mul o tanh_c
                    }
                
                return (hidden, cell)
            }
        
        return forward
    }

// Process sequence
let processSequence (lstm: LSTMCell) (sequence: Node[]) =
    result {
        let mutable hidden = Graph.constantArray [||] [||]  // Zero init
        let mutable cell = Graph.constantArray [||] [||]
        let outputs = ResizeArray()
        
        for input in sequence do
            let! h, c = lstm input hidden cell
            outputs.Add(h)
            hidden <- h
            cell <- c
        
        return outputs.ToArray()
    }
```

### GRU (Gated Recurrent Unit)

```fsharp
// GRU: Simpler than LSTM, often works as well

let gruCell (inputSize: int) (hiddenSize: int) =
    result {
        let! W_z = Layers.dense inputSize hiddenSize (Some Sigmoid) (Some 42)
        let! W_r = Layers.dense inputSize hiddenSize (Some Sigmoid) (Some 43)
        let! W_h = Layers.dense inputSize hiddenSize (Some Tanh) (Some 44)
        
        let forward (input: Node) (prevHidden: Node) =
            result {
                // Update gate
                let! combined = Graph.add input prevHidden
                let! z = Layers.forwardDense W_z combined
                
                // Reset gate
                let! r = Layers.forwardDense W_r combined
                
                // Candidate activation
                let! r_h = Graph.mul r prevHidden
                let! combined2 = Graph.add input r_h
                let! h_tilde = Layers.forwardDense W_h combined2
                
                // Final hidden: (1-z) * prev + z * h_tilde
                let! one_minus_z = Graph.sub (Graph.constant 1.0) z
                let! term1 = Graph.mul one_minus_z prevHidden
                let! term2 = Graph.mul z h_tilde
                
                return! Graph.add term1 term2
            }
        
        return forward
    }
```

## 10.3 Attention Mechanisms

### Self-Attention

```fsharp
// Attention: Query, Key, Value
let selfAttention (embedDim: int) (numHeads: int) =
    result {
        let headDim = embedDim / numHeads
        
        let! W_q = Layers.dense embedDim embedDim None (Some 42)
        let! W_k = Layers.dense embedDim embedDim None (Some 43)
        let! W_v = Layers.dense embedDim embedDim None (Some 44)
        let! W_o = Layers.dense embedDim embedDim None (Some 45)
        
        let forward (input: Node) =
            result {
                // Q, K, V projections
                let! Q = Layers.forwardDense W_q input
                let! K = Layers.forwardDense W_k input
                let! V = Layers.forwardDense W_v input
                
                // Scaled dot-product attention
                // Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
                
                let! Kt = Matrix.transpose K
                let! scores = Matrix.matmul Q Kt
                
                // Scale
                let scale = 1.0 / sqrt (float headDim)
                let! scaled = Ndarray.mulScalar scores scale
                
                // Softmax (along last axis)
                let! attnWeights = softmax scaled
                
                // Apply to values
                let! output = Matrix.matmul attnWeights V
                
                // Output projection
                return! Layers.forwardDense W_o output
            }
        
        return forward
    }
```

### Multi-Head Attention

```fsharp
// Multiple attention heads in parallel
let multiHeadAttention (embedDim: int) (numHeads: int) =
    result {
        let headDim = embedDim / numHeads
        
        // Create attention heads
        let! heads = 
            [1..numHeads]
            |> List.map (fun _ -> selfAttention headDim 1)
            |> Result.sequence
        
        let forward (input: Node) =
            result {
                // Apply each head
                let! headOutputs =
                    heads
                    |> List.map (fun head -> head input)
                    |> Result.sequence
                
                // Concatenate
                // ... concatenation code ...
                
                return input  // Placeholder
            }
        
        return forward
    }
```

## 10.4 Transfer Learning

### Loading Pre-trained Models

```fsharp
// Use pre-trained weights for faster training
open System.IO
open System.Text.Json

let loadPretrainedWeights (path: string) : Map<string, float[]> =
    let json = File.ReadAllText(path)
    JsonSerializer.Deserialize<Map<string, float[]>>(json)

let applyPretrained (model: 'a) (weights: Map<string, float[]>) =
    let parameters = Layers.getParameters model
    
    for (param, name) in parameters do
        match Map.tryFind name weights with
        | Some w -> param.Value.Value <- Some w
        | None -> ()  // Keep random initialization

// Fine-tuning strategy
let fineTune (model: 'a) (trainableLayers: string[]) =
    // Freeze early layers, train only final layers
    let parameters = Layers.getParameters model
    
    for (param, name) in parameters do
        if not (Array.contains name trainableLayers) then
            // Mark as non-trainable (skip in gradient update)
            ()
```

## 10.5 Generative Models

### Variational Autoencoder (VAE)

```fsharp
// VAE: Learn latent representations

let vaeEncoder (inputSize: int) (latentSize: int) =
    result {
        let! fc1 = Layers.dense inputSize 256 (Some ReLU) (Some 42)
        let! fc_mu = Layers.dense 256 latentSize None (Some 43)
        let! fc_logvar = Layers.dense 256 latentSize None (Some 44)
        
        let forward (input: Node) =
            result {
                let! h = Layers.forwardDense fc1 input
                let! mu = Layers.forwardDense fc_mu h
                let! logvar = Layers.forwardDense fc_logvar h
                return (mu, logvar)
            }
        
        return forward
    }

let vaeDecoder (latentSize: int) (outputSize: int) =
    result {
        let! fc1 = Layers.dense latentSize 256 (Some ReLU) (Some 45)
        let! fc2 = Layers.dense 256 outputSize (Some Sigmoid) (Some 46)
        
        let forward (z: Node) =
            result {
                let! h = Layers.forwardDense fc1 z
                return! Layers.forwardDense fc2 h
            }
        
        return forward
    }

// Reparameterization trick
let reparameterize (mu: Node) (logvar: Node) (rng: Random) =
    result {
        let std = Graph.exp (Graph.mul logvar (Graph.constant 0.5))
        let eps = Graph.constantArray 
            (Array.init (Ndarray.shape mu |> Array.reduce (*)) (fun _ -> 
                sqrt(-2.0 * log(rng.NextDouble())) * cos(2.0 * PI * rng.NextDouble())))
            (Ndarray.shape mu)
        
        return! Graph.add mu (Graph.mul std eps)
    }

// VAE Loss: Reconstruction + KL Divergence
let vaeLoss (recon: Node) (input: Node) (mu: Node) (logvar: Node) =
    result {
        // Reconstruction loss (binary cross entropy)
        let! bce = Loss.binaryCrossEntropy recon input
        
        // KL Divergence: -0.5 * sum(1 + log(var) - mu^2 - var)
        let kl = 
            Graph.mul
                (Graph.sub
                    (Graph.add (Graph.constant 1.0) logvar)
                    (Graph.add (Graph.pow mu (Graph.constant 2.0))
                               (Graph.exp logvar)))
                (Graph.constant -0.5)
        
        return! Graph.add bce kl
    }
```

### Generative Adversarial Networks (GAN)

```fsharp
// GAN: Generator and Discriminator

let generator (latentSize: int) (imageSize: int) =
    result {
        let! fc1 = Layers.dense latentSize 256 (Some ReLU) (Some 42)
        let! fc2 = Layers.dense 256 512 (Some ReLU) (Some 43)
        let! fc3 = Layers.dense 512 imageSize (Some Tanh) (Some 44)
        
        let forward (z: Node) =
            result {
                let! h1 = Layers.forwardDense fc1 z
                let! h2 = Layers.forwardDense fc2 h1
                return! Layers.forwardDense fc3 h2
            }
        
        return forward
    }

let discriminator (imageSize: int) =
    result {
        let! fc1 = Layers.dense imageSize 512 (Some LeakyReLU 0.2) (Some 45)
        let! fc2 = Layers.dense 512 256 (Some LeakyReLU 0.2) (Some 46)
        let! fc3 = Layers.dense 256 1 (Some Sigmoid) (Some 47)
        
        let forward (x: Node) =
            result {
                let! h1 = Layers.forwardDense fc1 x
                let! h2 = Layers.forwardDense fc2 h1
                return! Layers.forwardDense fc3 h2
            }
        
        return forward
    }

// GAN Training loop
let trainGAN (generator) (discriminator) (realData: float[][]) =
    // Alternate between training D and G
    // ...
    ()
```

## 10.6 Neural Architecture Search

### Random Search

```fsharp
// Simple NAS: Try random architectures
let randomArchitecture (searchSpace: Map<string, obj[]>) (rng: Random) =
    searchSpace
    |> Map.map (fun _ choices -
        choices.[rng.Next(choices.Length)])

let architectureSearch (trainData: obj) (valData: obj) 
                       (searchSpace: Map<string, obj[]>)
                       (numTrials: int) =
    let rng = Random(42)
    let results = ResizeArray()
    
    for trial = 1 to numTrials do
        let arch = randomArchitecture searchSpace rng
        let accuracy = trainAndEvaluate arch trainData valData
        results.Add((arch, accuracy))
    
    results
    |> Seq.maxBy snd
    |> fst
```

## 10.7 Deployment

### Model Serialization

```fsharp
// Save model for production
let saveModel (model: 'a) (path: string) =
    let parameters = Layers.getParameters model
    let state =
        parameters
        |> List.map (fun (param, name) -
            let value = param.Value.Value |> Option.defaultValue [||]
            (name, value))
        |> Map.ofList
    
    let json = JsonSerializer.Serialize(state)
    File.WriteAllText(path, json)

let loadModel (model: 'a) (path: string) =
    let json = File.ReadAllText(path)
    let state = JsonSerializer.Deserialize<Map<string, float[]>>(json)
    
    applyPretrained model state
```

### ONNX Export

```fsharp
// Export to ONNX for interoperability
// (Would need ONNX runtime integration)
let exportToONNX (model: 'a) (path: string) =
    // Convert computation graph to ONNX format
    // ...
    ()
```

## 10.8 Exercises

### Exercise 10.1: Implement ResNet-18

```fsharp
// Build ResNet-18 architecture
let resnet18 (numClasses: int) =
    result {
        // Initial conv
        let! conv1 = Conv2D.create 3 64 7 Padding.Same (Some 42)
        let pool1 = Pool2D.create MaxPool 3 2 Padding.Valid
        
        // Residual groups
        let! group1 = residualGroup 64 64 2  // 2 blocks
        let! group2 = residualGroup 64 128 2 // 2 blocks
        let! group3 = residualGroup 128 256 2 // 2 blocks
        let! group4 = residualGroup 256 512 2 // 2 blocks
        
        // Final layers
        let! fc = Layers.dense 512 numClasses (Some Softmax) (Some 43)
        
        // ... forward pass ...
        return ()
    }
```

### Exercise 10.2: Implement Transformer Block

```fsharp
// Transformer encoder block
let transformerBlock (embedDim: int) (numHeads: int) (ffDim: int) =
    result {
        let! attn = multiHeadAttention embedDim numHeads
        let! ff = feedForward embedDim ffDim
        
        let forward (input: Node) =
            result {
                // Self-attention with residual
                let! attnOut = attn input
                let! normed1 = layerNorm (Graph.add input attnOut)
                
                // Feed-forward with residual
                let! ffOut = ff normed1
                let! output = layerNorm (Graph.add normed1 ffOut)
                
                return output
            }
        
        return forward
    }
```

## 10.9 Best Practices

### Training Tips

1. **Learning Rate Scheduling**
```fsharp
let cosineAnnealing (initialLR: float) (minLR: float) 
                    (currentEpoch: int) (maxEpochs: int) =
    minLR + 0.5 * (initialLR - minLR) * 
    (1.0 + cos(PI * float currentEpoch / float maxEpochs))
```

2. **Early Stopping**
```fsharp
let earlyStopping (patience: int) =
    let mutable bestLoss = Double.MaxValue
    let mutable wait = 0
    
    fun (valLoss: float) ->
        if valLoss < bestLoss then
            bestLoss <- valLoss
            wait <- 0
            false  // Continue training
        else
            wait <- wait + 1
            wait >= patience  // Stop if patience exceeded
```

3. **Gradient Clipping**
```fsharp
let clipGradients (gradients: float[][]) (maxNorm: float) =
    let globalNorm = 
        gradients
        |> Array.collect id
        |> Array.sumBy (fun g -> g * g)
        |> sqrt
    
    if globalNorm > maxNorm then
        let scale = maxNorm / globalNorm
        gradients |> Array.map (Array.map ((*) scale))
    else
        gradients
```

## 10.10 Summary

Key concepts:
- Modern architectures use skip connections, normalization, attention
- RNNs/LSTMs/GRUs process sequences
- Transfer learning accelerates training
- Generative models create new data
- Attention mechanisms enable long-range dependencies
- Deployment requires careful optimization

---

*Next: [Chapter 12: Case Studies](chapter12.md)*
