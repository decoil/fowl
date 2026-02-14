# Neural Network Foundation Plan

## Research Sources
- Owl Tutorial: Neural Network chapter (to study)
- Architecture Book: Chapter 4 (Neural Network Module)
- Owl source: src/owl/neural (graph module, layers, optimizers)
- PyTorch/TensorFlow patterns for comparison

## Design Principles
1. **Computation Graph**: Lazy evaluation with automatic differentiation
2. **Modular Layers**: Composable, type-safe layer definitions
3. **F# Idioms**: Computation expressions for model definition
4. **Performance**: Leverage existing AD module
5. **Owl Compatibility**: Similar API for easy migration

## Architecture Overview

### Core Components

#### 1. Computation Graph (Fowl.Neural.Graph)
```fsharp
type Node<'T> = {
    Id: int
    Shape: Shape
    Value: 'T option ref
    Grad: 'T option ref
    Op: Operation<'T>
    Parents: Node<'T> list
}

type Operation<'T> =
    | Input of string
    | Const of 'T
    | Add
    | Mul
    | MatMul
    | Activation of ActivationFn
    | Loss of LossFn
    | Custom of (Node<'T> list -> 'T) * (Node<'T> list -> 'T list)
```

#### 2. Layer Types (Fowl.Neural.Layers)
```fsharp
type Layer = {
    Name: string
    Parameters: Node<float> list
    Forward: Node<float> -> Node<float>
    ShapeIn: Shape
    ShapeOut: Shape
}

// Layers to implement
- Dense (fully connected)
- Conv2D (convolutional)
- MaxPool2D / AvgPool2D
- Dropout
- BatchNorm
- Flatten
- Activation layers
```

#### 3. Activation Functions (Fowl.Neural.Activations)
```fsharp
type ActivationFn =
    | ReLU
    | Sigmoid
    | Tanh
    | Softmax
    | LeakyReLU of float
    | ELU of float
    | Swish
    | Gelu
    | Custom of (float -> float) * (float -> float)  // fn * derivative
```

#### 4. Loss Functions (Fowl.Neural.Losses)
```fsharp
type LossFn =
    | MSE           // Mean squared error
    | CrossEntropy  // Cross-entropy (log loss)
    | BCE           // Binary cross-entropy
    | NLL           // Negative log-likelihood
    | Huber of float // Huber loss (delta)
    | Custom of (float[] -> float[] -> float) * (float[] -> float[] -> float[])
```

#### 5. Optimizers (Fowl.Neural.Optimizers)
```fsharp
type Optimizer = {
    LearningRate: float
    Update: Node<float> list -> unit
}

// Optimizers to implement
- SGD (with momentum)
- Adam (adaptive moment estimation)
- RMSprop
- Adagrad
- Adadelta
```

### Model Definition API

Using computation expressions:
```fsharp
let model = neuralNet {
    input [|784|]
    dense 256 --> relu
    dropout 0.5
    dense 128 --> relu
    dense 10  --> softmax
}

// Or functional style
let model =
    input [|784|]
    |> dense 256
    |> relu
    |> dropout 0.5
    |> dense 128
    |> relu
    |> dense 10
    |> softmax
```

### Training Loop
```fsharp
let train (model: Model) (data: DataLoader) (epochs: int) =
    for epoch = 1 to epochs do
        for batch in data do
            // Forward pass
            let output = model.Forward batch.Inputs
            let loss = Loss.crossEntropy output batch.Targets
            
            // Backward pass (use existing AD module)
            let grads = AD.grad loss
            
            // Update parameters
            optimizer.Update model.Parameters grads
```

## Implementation Phases

### Phase 1: Core Graph (Week 1)
- [ ] Node and Graph types
- [ ] Basic operations (+, *, matmul)
- [ ] Forward pass execution
- [ ] Topological sorting for graph execution

### Phase 2: AD Integration (Week 1-2)
- [ ] Connect to existing AD module
- [ ] Reverse-mode autodiff through graph
- [ ] Gradient computation for all ops

### Phase 3: Layer Implementations (Week 2-3)
- [ ] Dense layer
- [ ] Conv2D layer
- [ ] Pooling layers
- [ ] Dropout
- [ ] Batch normalization

### Phase 4: Training Infrastructure (Week 3-4)
- [ ] Optimizers (SGD, Adam)
- [ ] Loss functions
- [ ] Training loop
- [ ] Metrics (accuracy, loss tracking)

### Phase 5: Advanced Features (Week 4-5)
- [ ] Model serialization
- [ ] Data loaders
- [ ] Callbacks (early stopping, LR scheduling)
- [ ] Pre-trained model loading

## Technical Decisions

### 1. AD Integration
Reuse existing AD module from Fowl.AD:
- Wrap AD operations in Graph nodes
- Use dual numbers for forward pass
- Use computation graph for backward pass

### 2. Memory Management
- Use ArrayPool for temporary buffers
- Lazy evaluation to avoid intermediate allocations
- Minimize GC pressure during training

### 3. Type Safety
- Phantom types for tensor shapes (when possible)
- Compile-time shape checking where feasible
- Runtime shape validation as fallback

### 4. Performance
- SIMD operations from Fowl.SIMD
- Parallel batch processing
- Cache-friendly data layouts

## References

### Owl Neural Network Module
- `owl/src/owl/neural/owl_neural_graph.ml` - Computation graph
- `owl/src/owl/neural/owl_neural_layers.ml` - Layer definitions
- `owl/src/owl/neural/owl_neural_optimisers.ml` - Optimizers

### External Resources
- PyTorch internals documentation
- TensorFlow graph execution
- "Deep Learning with PyTorch" (book)
- Fast.ai course materials

## Success Metrics
- [ ] MNIST classification working
- [ ] CIFAR-10 classification working
- [ ] Training speed within 2x of PyTorch CPU
- [ ] Model serialization/deserialization
- [ ] Pre-trained weights loading

---

_Plan created: 2026-02-14_
_Next: Begin Phase 1 - Core Graph implementation_