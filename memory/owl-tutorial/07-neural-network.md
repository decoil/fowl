# Owl Tutorial Chapter 7: Deep Neural Networks

## From Perceptron to Neural Networks

### Perceptron
```ocaml
g(x) = 1 if w·x + b > 0, else 0
```
- Simple binary classifier
- Cannot solve XOR problem (led to multi-layer networks)

### Logistic Regression → Neural Network
- Add hidden layers between input and output
- Each layer: `y = a(x·w + b)` where a is activation function

## Feed Forward Network Implementation

### Core Types
```ocaml
type layer = {
  mutable w: t;         (* weights *)
  mutable b: t;         (* bias *)
  mutable a: t -> t;    (* activation function *)
}

type network = { layers: layer array }
```

### Forward Pass
```ocaml
let run_layer x l = Maths.((x *@ l.w) + l.b) |> l.a
let run_network x nn = Array.fold_left run_layer x nn.layers
```

### Backpropagation
Just reverse-mode AD! The gradient of the loss w.r.t. each parameter.
```ocaml
let loss = loss_fun nn x y in
reverse_prop (F 1.) loss;
let w' = adjval l.w |> unpack_arr in  (* gradient for weights *)
let b' = adjval l.b |> unpack_arr in  (* gradient for bias *)
```

## Activation Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| Sigmoid | `1/(1+e^{-x})` | Binary classification |
| Tanh | `(e^x - e^{-x})/(e^x + e^{-x})` | Hidden layers |
| ReLU | `max(0, x)` | Hidden layers (avoids vanishing gradient) |
| Softmax | `e^{x_i}/Σe^{x_k}` | Multi-class output |

## Initialisation

### Problem
- Random uniform/gaussian → values explode or vanish
- Need careful scaling

### Solutions
```ocaml
Init.Standard     (* Xavier: √(1/n) *)
Init.LecunNormal  (* √(1/n) Gaussian *)
Init.GlorotUniform (* √(6/(n₀+n₁)) *)
```

## Owl Neural Network Module

### Module Architecture
```
Neural
├── Neuron (layer implementations)
├── Graph (network construction)
├── Optimise (training)
└── Algodiff (automatic differentiation)
```

### Network Construction
```ocaml
open Neural.S.Graph

let make_network input_shape =
  input input_shape
  |> fully_connected 40 ~act_typ:Activation.Tanh
  |> linear 10 ~act_typ:Activation.(Softmax 1)
  |> get_network
```

### Training Parameters
```ocaml
let params = Params.config
  ~batch:(Batch.Mini 100)
  ~learning_rate:(Learning_Rate.Adagrad 0.005)
  0.1  (* epochs *)

Graph.train ~params network x y
```

## Advanced Network Types

### Convolutional Neural Network (CNN)
```ocaml
let make_cnn input_shape =
  input input_shape
  |> lambda (fun x -> Maths.(x / F 256.))
  |> conv2d [|5;5;1;32|] [|1;1|] ~act_typ:Activation.Relu
  |> max_pool2d [|2;2|] [|2;2|]
  |> dropout 0.1
  |> fully_connected 1024 ~act_typ:Activation.Relu
  |> linear 10 ~act_typ:Activation.(Softmax 1)
  |> get_network
```

### Recurrent Neural Network (RNN)
- Processes sequential data
- Hidden state passes between time steps

### LSTM (Long Short-Term Memory)
- Three gates: forget, remember, output
- Solves vanishing gradient problem in sequences

### Generative Adversarial Network (GAN)
- Generator + Discriminator competing
- Generator learns to create realistic data

## Key Insight: AD Makes Neural Networks Simple

The entire neural network training is just:
1. Define forward pass
2. Use reverse-mode AD for gradients
3. Update parameters with gradient descent

**Owl's neural network module is only ~3500 LOC** because AD handles all the complexity!

## F# Design Notes

### Neuron Types
```fsharp
type Neuron =
    | Input of shape: int array
    | Linear of outputs: int * activation: Activation
    | Conv2D of filters: int array * stride: int array
    | MaxPool2D of kernel: int array * stride: int array
    | Dropout of rate: float
    | LSTM of units: int
```

### Network Builder
```fsharp
type NetworkBuilder() =
    member _.Yield(()) = []
    [<CustomOperation("input")>]
    member _.Input(state, shape) = Input shape :: state
    [<CustomOperation("dense")>]
    member _.Dense(state, outputs, activation) = ...

let network = NetworkBuilder()
let nn = network {
    input [|28;28;1|]
    dense 40 Activation.Tanh
    dense 10 (Activation.Softmax 1)
}
```

---
_Learned: 2026-02-13_
