# Neural Network Research Notes

## Source: Architecture of Advanced Numerical Analysis Systems (Ch 4)
## Date: 2026-02-14 (studied earlier today)

---

## 1. Owl's Neural Network Architecture

### Core Design Philosophy
- **Computation Graph**: Central abstraction for lazy evaluation
- **Automatic Differentiation**: Reverse-mode through graph traversal
- **Modular Layers**: Composable units with forward/backward passes
- **Type Safety**: OCaml's type system ensures shape compatibility

### Graph Structure

Owl uses a directed acyclic graph (DAG) where:
- **Nodes** represent tensors (ndarrays)
- **Edges** represent operations
- **Leaf nodes** are inputs/parameters
- **Root nodes** are outputs/losses

```ocaml
(* Owl's graph node (conceptual) *)
type node = {
  id: int;
  shape: int array;
  value: tensor option ref;
  grad: tensor option ref;
  op: op_type;
  parents: node list;
  children: node list;
}
```

Key insight: **Reference cells (ref)** allow lazy evaluation - values computed only when needed.

### Operation Types

Owl categorizes operations:
1. **Element-wise**: add, mul, sin, exp, etc.
2. **Reduction**: sum, mean, max along axes
3. **Shape**: reshape, transpose, slice
4. **Matrix**: matmul, conv2d, maxpool
5. **Activation**: relu, sigmoid, softmax
6. **Loss**: cross_entropy, mse

Each operation implements:
- `forward`: Compute output from inputs
- `backward`: Compute gradients w.r.t. inputs

### Layer Abstraction

```ocaml
(* Owl layer type (simplified) *)
type layer = {
  name: string;
  trainable: bool;
  parameters: tensor list;  (* weights, biases *)
  forward: tensor -> tensor;
}
```

Layers are **stateless functions** - parameters stored separately in graph nodes.

---

## 2. Automatic Differentiation Integration

### Reverse-Mode Autodiff

From Architecture book:
1. **Forward pass**: Compute all node values (topological order)
2. **Backward pass**: Compute gradients (reverse topological order)
3. **Chain rule**: Applied at each node

```ocaml
(* Simplified backward pass *)
let backward (output: node) : unit =
  output.grad := ones_like output.shape;
  let sorted = topological_sort output in
  List.iter (fun node ->
    match node.op with
    | Add -> (* grad = sum of parent grads *)
    | MatMul -> (* grad = matmul(grad, transpose(w)) *)
    (* ... *)
  ) (List.rev sorted)
```

### Gradient Accumulation

Key pattern: **Gradients accumulate** in ref cells:
```ocaml
node.grad := !(node.grad) + computed_grad
```

This handles:
- Multiple children (gradient from all paths)
- Shared parameters (weight sharing)

---

## 3. Optimizer Design

### Optimizer State

```ocaml
type optimizer = {
  learning_rate: float;
  update: tensor list -> tensor list -> unit;
}
```

### SGD with Momentum

```ocaml
let sgd ~lr ~momentum params grads =
  let velocity = List.map (fun p -> zeros_like p) params in
  fun () ->
    List.iter2 (fun v (p, g) ->
      v := momentum * !v + g;  (* update velocity *)
      p := p - lr * !v         (* update parameter *)
    ) velocity (List.combine params grads)
```

Key insight: **Stateful optimizers** maintain velocity/accumulator tensors.

### Adam Optimizer

From Architecture book:
- First moment (mean of gradients)
- Second moment (mean of squared gradients)
- Bias correction for early iterations

```ocaml
let adam ~lr ~beta1 ~beta2 params =
  let m = List.map zeros_like params in  (* first moment *)
  let v = List.map zeros_like params in  (* second moment *)
  let t = ref 0 in
  fun grads ->
    t := !t + 1;
    List.iter2 (fun (m, v) (p, g) ->
      m := beta1 * !m + (1. - beta1) * g;
      v := beta2 * !v + (1. - beta2) * g * g;
      let m_hat = !m / (1. - beta1 ** !t) in
      let v_hat = !v / (1. - beta2 ** !t) in
      p := p - lr * m_hat / (sqrt v_hat + epsilon)
    ) (List.combine m v) (List.combine params grads)
```

---

## 4. F# Implementation Patterns

### Pattern 1: Computation Expression for Model Building

```fsharp
type ModelBuilder() =
    member _.Yield(_) = []
    member _.Run(layers) = Sequential(layers)
    
    [<CustomOperation("dense")>]
    member _.Dense(layers, units) = Dense(units) :: layers
    
    [<CustomOperation("relu")>]
    member _.Relu(layers) = Activation(ReLU) :: layers

let model = modelBuilder {
    dense 256
    relu
    dense 10
}
```

### Pattern 2: Railway-Oriented Error Handling

```fsharp
let forward (input: Ndarray) (layer: Layer) : FowlResult<Ndarray> =
    match layer with
    | Dense w b -> matmul input w |> Result.bind (add b)
    | Activation fn -> applyActivation fn input
```

### Pattern 3: Phantom Types for Shape Safety

```fsharp
type Node<'Shape, 'Dtype> = {
    Value: 'Dtype Ndarray
    Grad: 'Dtype Ndarray option ref
}

// Compile-time shape checking where possible
let matmul (a: Node<|[|N; M|], _>) (b: Node<|[|M; P|], _>) : Node<|[|N; P|], _>
```

### Pattern 4: Active Patterns for Layer Matching

```fsharp
let (|Trainable|NonTrainable|) (layer: Layer) =
    match layer with
    | Dense _ | Conv2D _ -> Trainable
    | ReLU | Dropout _ -> NonTrainable
```

---

## 5. Key Implementation Decisions

### Decision 1: Eager vs Lazy Evaluation

**Owl choice**: Lazy evaluation via computation graph
- Pros: Optimization opportunities, memory efficiency
- Cons: Debugging complexity

**Fowl approach**: Hybrid
- Graph construction: Lazy (build computation graph)
- Forward pass: Eager (compute immediately for debugging)
- Backward pass: Lazy (compute only needed gradients)

### Decision 2: Mutable vs Immutable State

**Owl choice**: Mutable (ref cells for values/grads)

**Fowl approach**: Controlled mutability
- Graph structure: Immutable (pure F#)
- Values/grads: Mutable refs (necessary for AD)
- Optimizer state: Mutable refs (velocity, moments)

### Decision 3: Integration with Existing AD Module

**Approach**: Wrap AD operations in Graph nodes
```fsharp
// Use existing forward-mode AD for local gradients
// Use graph traversal for global gradient flow
```

---

## 6. Testing Strategy

From Architecture book principles:
1. **Unit tests**: Each operation's forward/backward
2. **Gradient checks**: Finite difference verification
3. **Shape propagation**: Verify output shapes
4. **Integration tests**: End-to-end training on MNIST

---

## 7. Performance Considerations

### Memory Management
- Use **ArrayPool** for temporary buffers
- **Gradient checkpointing** for large models
- **In-place operations** where safe

### Parallelization
- **Data parallel**: Process batches in parallel
- **Model parallel**: Split large layers (future)

### SIMD
- Leverage existing **Fowl.SIMD** module
- Vectorized element-wise operations

---

## 8. References

1. **Architecture Book** Ch 4: Neural Network Module
2. **Owl Tutorial**: Neural Network chapter
3. **Owl Source**: `src/owl/neural/*.ml`
4. **PyTorch**: `torch.nn.Module` design
5. **TensorFlow**: Graph execution model

---

## Next Steps

1. ✅ Research complete (this document)
2. ⏳ Implement core Graph types
3. ⏳ Implement basic operations (+, *, matmul)
4. ⏳ Integrate with existing AD module
5. ⏳ Implement Dense layer
6. ⏳ Test with simple regression

---

_Research complete. Ready for implementation._
