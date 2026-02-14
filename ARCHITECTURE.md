# Fowl Architecture Specification v0.1

**Project:** Fowl - F# Numerical Computing Library  
**Based on:** Owl Architecture (OCaml) + F# Idioms  
**Date:** 2026-02-14

---

## 1. System Overview

### Goals
- Production-grade numerical computing in F#
- Type-safe, composable API
- Competitive performance with Owl/NumPy
- Cross-platform (.NET Core + accelerators)

### Architecture Principles (from Owl)
1. **Modular functor design** → F# interfaces + records
2. **C-backend for performance** → Native interop
3. **Computation graph** → Lazy evaluation + optimization
4. **Memory efficiency** → Pebble-game allocation
5. **Hardware abstraction** → Pluggable engines

---

## 2. Core Modules

### 2.1 Ndarray (Foundation)

**Design:**
```fsharp
type Layout = CLayout | FortranLayout

type Shape = int array

type Ndarray<'T> =
    | Dense of DenseArray<'T>
    | Sparse of SparseArray<'T>

and DenseArray<'T> = {
    data: 'T array
    shape: Shape
    strides: int array
    offset: int
}

and SparseArray<'T> = {
    indices: int array array
    values: 'T array
    shape: Shape
    format: SparseFormat
}

and SparseFormat = CSR | CSC | COO
```

**Type Safety via Phantom Types:**
```fsharp
type Float32 = class end
type Float64 = class end
type Complex32 = class end
type Complex64 = class end

type Ndarray<'Kind, 'T> = ... // phantom type for dispatch
```

**Module Structure:**
```fsharp
module Ndarray =
    // Creation
    val zeros<'K> : Shape -> Ndarray<'K, float>
    val ones<'K> : Shape -> Ndarray<'K, float>
    val empty<'K> : Shape -> Ndarray<'K, float>
    val create<'K> : Shape -> float -> Ndarray<'K, float>
    val linspace : float -> float -> int -> Ndarray<Float64, float>
    val arange : float -> float -> float -> Ndarray<Float64, float>
    
    // Indexing
    val get : Ndarray<'K, 'T> -> int array -> 'T
    val set : Ndarray<'K, 'T> -> int array -> 'T -> unit
    val slice : Ndarray<'K, 'T> -> SliceSpec array -> Ndarray<'K, 'T>
    
    // Manipulation
    val reshape : Ndarray<'K, 'T> -> Shape -> Ndarray<'K, 'T>
    val transpose : Ndarray<'K, 'T> -> int array -> Ndarray<'K, 'T>
    val concatenate : Ndarray<'K, 'T> array -> int -> Ndarray<'K, 'T>
    val tile : Ndarray<'K, 'T> -> int array -> Ndarray<'K, 'T>
    
    // Iteration
    val map : ('T -> 'U) -> Ndarray<'K, 'T> -> Ndarray<'K, 'U>
    val fold : ('State -> 'T -> 'State) -> 'State -> Ndarray<'K, 'T> -> 'State
    val scan : ('State -> 'T -> 'State) -> 'State -> Ndarray<'K, 'T> -> Ndarray<'K, 'State>
    
    // I/O
    val toArray : Ndarray<'K, 'T> -> 'T array
    val ofArray : 'T array -> Shape -> Ndarray<'K, 'T>
    val save : string -> Ndarray<'K, 'T> -> unit
    val load : string -> Ndarray<'K, 'T>
```

**Slice Specification:**
```fsharp
type SliceSpec =
    | All
    | Index of int
    | Range of (int option * int option * int option) // (start, stop, step)
    | IndexArray of int array
```

### 2.2 Core Math Module

**C-Backed Primitives:**
```fsharp
module Core =
    // Unary
    val sin : Ndarray<'K, float> -> Ndarray<'K, float>
    val cos : Ndarray<'K, float> -> Ndarray<'K, float>
    val exp : Ndarray<'K, float> -> Ndarray<'K, float>
    val log : Ndarray<'K, float> -> Ndarray<'K, float>
    val sqrt : Ndarray<'K, float> -> Ndarray<'K, float>
    val abs : Ndarray<'K, float> -> Ndarray<'K, float>
    
    // Binary
    val add : Ndarray<'K, 'T> -> Ndarray<'K, 'T> -> Ndarray<'K, 'T>
    val sub : Ndarray<'K, 'T> -> Ndarray<'K, 'T> -> Ndarray<'K, 'T>
    val mul : Ndarray<'K, 'T> -> Ndarray<'K, 'T> -> Ndarray<'K, 'T>
    val div : Ndarray<'K, 'T> -> Ndarray<'K, 'T> -> Ndarray<'K, 'T>
    val pow : Ndarray<'K, float> -> Ndarray<'K, float> -> Ndarray<'K, float>
    
    // Matrix
    val dot : Ndarray<'K, float> -> Ndarray<'K, float> -> Ndarray<'K, float>
    val matmul : Ndarray<'K, float> -> Ndarray<'K, float> -> Ndarray<'K, float>
```

### 2.3 Linear Algebra (Linalg)

```fsharp
module Linalg =
    // Factorizations
    val lu : Ndarray<'K, float> -> Ndarray<'K, float> * Ndarray<'K, float>
    val qr : Ndarray<'K, float> -> Ndarray<'K, float> * Ndarray<'K, float>
    val svd : Ndarray<'K, float> -> Ndarray<'K, float> * Ndarray<'K, float> * Ndarray<'K, float>
    val cholesky : Ndarray<'K, float> -> Ndarray<'K, float>
    
    // Solvers
    val solve : Ndarray<'K, float> -> Ndarray<'K, float> -> Ndarray<'K, float>
    val lstsq : Ndarray<'K, float> -> Ndarray<'K, float> -> Ndarray<'K, float>
    
    // Eigen
    val eig : Ndarray<'K, float> -> Ndarray<'K, complex> * Ndarray<'K, complex>
    val eigvals : Ndarray<'K, float> -> Ndarray<'K, complex>
```

### 2.4 Statistics (Stats)

```fsharp
module Stats =
    // Descriptive
    val mean : Ndarray<'K, float> -> float
    val std : Ndarray<'K, float> -> float
    val var : Ndarray<'K, float> -> float
    val median : Ndarray<'K, float> -> float
    val percentile : Ndarray<'K, float> -> float -> float
    
    // Moments
    val skewness : Ndarray<'K, float> -> float
    val kurtosis : Ndarray<'K, float> -> float
    val moment : int -> Ndarray<'K, float> -> float
    
    // Distributions (each has _pdf, _cdf, _rvs, _ppf)
    module Gaussian =
        val pdf : mu:float -> sigma:float -> float -> float
        val cdf : mu:float -> sigma:float -> float -> float
        val rvs : mu:float -> sigma:float -> Shape -> Ndarray<Float64, float>
    
    module Uniform =
        val pdf : low:float -> high:float -> float -> float
        val rvs : low:float -> high:float -> Shape -> Ndarray<Float64, float>
```

### 2.5 Algorithmic Differentiation (AD)

**Core Types:**
```fsharp
namespace Fowl.AD

type Mode = Forward | Reverse

type DualNumber<'T> =
    | Forward of primal:'T * tangent:'T * tag:int
    | Reverse of primal:'T * adjoint:'T ref * callbacks:AdjointCallbacks * tag:int

and AdjointCallbacks = {
    adjoint: 'T -> 'T ref -> (('T * 'T ref) list) -> ('T * 'T ref) list
    register: 'T ref list -> 'T ref list
    label: string * 'T ref list
}
```

**API:**
```fsharp
module AD =
    val makeForward : 'T -> 'T -> int -> DualNumber<'T>
    val makeReverse : 'T -> int -> DualNumber<'T>
    
    val primal : DualNumber<'T> -> 'T
    val tangent : DualNumber<'T> -> 'T
    val adjval : DualNumber<'T> -> 'T
    
    val diff : (DualNumber<'T> -> DualNumber<'T>) -> 'T -> 'T
    val grad : (DualNumber<'T> -> DualNumber<float>) -> 'T -> 'T
    val jacobian : (DualNumber<'T> -> DualNumber<'U>) -> 'T -> 'U
    val hessian : (DualNumber<'T> -> DualNumber<float>) -> 'T -> 'T
```

**Computation Graph:**
```fsharp
module ComputationGraph =
    type Node = {
        id: int
        name: string
        inputs: Node array
        operation: Operation
        shape: Shape option
    }
    
    and Operation =
        | Constant of Ndarray<'K, 'T>
        | Variable of string * Shape
        | Add
        | Mul
        | Sin
        | MatMul
        | Conv2d of kernel:Shape * stride:Shape * padding:Padding
        | ...
    
    and Padding = Same | Valid
    
    // Construction
    val create : Shape -> Node  // variable
    val constant : Ndarray<'K, 'T> -> Node
    val add : Node -> Node -> Node
    val mul : Node -> Node -> Node
    
    // Optimization
    val optimize : Node -> Node  // constant folding, fusing, etc.
    
    // Execution
    val eval : Node -> Ndarray<'K, 'T>
    val compile : Node -> CompiledGraph
```

### 2.6 Neural Networks

```fsharp
namespace Fowl.Neural

type Neuron = {
    forward: Ndarray<'K, float> -> Ndarray<'K, float>
    init: unit -> unit
    getParams: unit -> Ndarray<'K, float> array
    update: Ndarray<'K, float> array -> unit
}

module Neurons =
    val linear : inFeatures:int -> outFeatures:int -> Neuron
    val conv2d : inChannels:int -> outChannels:int -> kernelSize:int -> Neuron
    val relu : Neuron
    val sigmoid : Neuron
    val tanh : Neuron
    val dropout : rate:float -> Neuron
    val batchNorm : features:int -> Neuron

type Network = {
    nodes: Node array
    parameters: Ndarray<'K, float> array
    forward: Ndarray<'K, float> -> Ndarray<'K, float>
}

module Network =
    val sequential : Neuron list -> Network
    val forward : Network -> Ndarray<'K, float> -> Ndarray<'K, float>
    val train : Network -> (Ndarray<'K, float> * Ndarray<'K, float>) array -> TrainConfig -> unit
    
    val save : string -> Network -> unit
    val load : string -> Network
```

---

## 3. Implementation Strategy

### Phase 1: Foundation (Weeks 1-2)
- [ ] Core Ndarray type with Dense implementation
- [ ] Shape manipulation (reshape, transpose, slice)
- [ ] Basic math operations (map, fold, scan)
- [ ] C interop for performance-critical loops

### Phase 2: Math Primitives (Weeks 3-4)
- [ ] Interface to OpenBLAS for linear algebra
- [ ] Special functions (via MathNet or Cephes)
- [ ] Statistics module
- [ ] Basic plotting integration

### Phase 3: Advanced Features (Weeks 5-6)
- [ ] Algorithmic differentiation
- [ ] Computation graph with lazy evaluation
- [ ] Graph optimization (constant folding, fusion)

### Phase 4: Neural Networks (Weeks 7-8)
- [ ] Neuron abstractions
- [ ] Network construction DSL
- [ ] Training loop with optimizer
- [ ] Model serialization

### Phase 5: Acceleration (Weeks 9-10)
- [ ] ONNX export for cross-platform execution
- [ ] GPU support (via TorchSharp or custom CUDA)
- [ ] SIMD optimization

---

## 4. F# Idioms & Patterns

### Computation Expressions
```fsharp
// For AD tracking
let result = ad {
    let! x = variable 2.0
    let! y = variable 3.0
    return x * x + y * y
}

// For neural network construction
let net = neural {
    input [28; 28]
    conv2d 32 3
    relu
    maxPool 2
    flatten
    linear 128
    relu
    dropout 0.5
    linear 10
    softmax
}
```

### Type Providers (optional)
```fsharp
// Load data with schema inference
type Dataset = CsvProvider<"data.csv">
let data = Dataset.Load("data.csv")
```

### Active Patterns for Shape Checking
```fsharp
let (|Matrix|_|) (arr: Ndarray<'K, 'T>) =
    if arr.shape.Length = 2 then Some (arr.shape.[0], arr.shape.[1])
    else None

let (|Vector|_|) (arr: Ndarray<'K, 'T>) =
    if arr.shape.Length = 1 then Some arr.shape.[0]
    elif arr.shape.Length = 2 && arr.shape.[1] = 1 then Some arr.shape.[0]
    else None
```

### Pipeline Operators
```fsharp
let normalize (x: Ndarray<'K, float>) =
    x
    |> Ndarray.sub (Stats.mean x)
    |> Ndarray.div (Stats.std x)
```

---

## 5. Performance Considerations

### Memory Layout
- C-layout (row-major) default
- Cache-aware tiling for matrix operations
- Memory pool for temporary allocations

### Parallelism
- Parallel.For for embarrasingly parallel operations
- SIMD vectorization via System.Runtime.Intrinsics
- OpenMP-style parallelism for CPU-bound operations

### Interop Strategy
```fsharp
// P/Invoke for native libraries
module Native =
    [<DllImport("openblas")>]
    extern void cblas_sgemm(...)
    
    [<DllImport("fowl_core")>]
    extern void fowl_map_float32(...)
```

---

## 6. Dependencies

### Core
- .NET 8.0+ (Span<T>, hardware intrinsics)
- FSharp.Core

### Math
- MathNet.Numerics (fallback for managed linear algebra)
- OpenBLAS (native, primary)

### Acceleration
- TorchSharp (optional, for GPU)
- Microsoft.ML.OnnxRuntime (optional)

### Testing
- Expecto (testing framework)
- FsCheck (property-based testing)

---

## 7. Directory Structure

```
fowl/
├── src/
│   ├── Fowl.Core/           # Ndarray, core types
│   ├── Fowl.Math/           # Mathematical functions
│   ├── Fowl.Linalg/         # Linear algebra
│   ├── Fowl.Stats/          # Statistics
│   ├── Fowl.AD/             # Algorithmic differentiation
│   ├── Fowl.Graph/          # Computation graph
│   ├── Fowl.Neural/         # Neural networks
│   ├── Fowl.Native/         # C interop bindings
│   └── Fowl.Symbolic/       # ONNX integration
├── tests/
│   ├── Fowl.Core.Tests/
│   ├── Fowl.Math.Tests/
│   └── ...
├── benchmarks/
│   └── Fowl.Benchmarks/
├── docs/
│   └── ...
└── Fowl.sln
```

---

## 8. API Examples

### Basic Usage
```fsharp
open Fowl
open Fowl.Ndarray

// Create arrays
let x = Ndarray.linspace 0.0 10.0 100
let y = Ndarray.sin x

// Slicing
let z = x.[1..50]  // F# slice syntax
let w = x.[..10]   // First 10 elements

// Broadcasting
let a = Ndarray.ones [3; 4]
let b = Ndarray.ones [4]
let c = a + b  // Broadcasting works!
```

### Linear Algebra
```fsharp
open Fowl.Linalg

let A = Ndarray.random [100; 100]
let B = Ndarray.random [100; 1]

let X = Linalg.solve A B
let L, U = Linalg.lu A
let U, S, Vt = Linalg.svd A
```

### Neural Networks
```fsharp
open Fowl.Neural

let model = Network.sequential [
    Neurons.linear 784 256
    Neurons.relu
    Neurons.dropout 0.2
    Neurons.linear 256 10
    Neurons.softmax
]

let loss = Network.train model trainData {
    epochs = 10
    batchSize = 32
    learningRate = 0.001
    optimizer = Adam
}
```

### AD Example
```fsharp
open Fowl.AD

let f x = x * x + 3.0 * x + 2.0
let f' = diff f  // First derivative
let f'' = diff f' // Second derivative

printfn "f(2) = %f, f'(2) = %f, f''(2) = %f" 
    (f 2.0) (f' 2.0) (f'' 2.0)
```

---

## 9. Benchmarking Targets

| Operation | Target (vs NumPy) |
|-----------|------------------|
| Element-wise add | 1.0x |
| Matrix multiply (1000x1000) | 1.0x (via OpenBLAS) |
| FFT | 0.8x |
| Conv2D | 0.7x |
| AD forward pass | 1.2x |

---

## 10. Future Work

- [ ] Distributed computing (Actor model)
- [ ] GPU kernels (CUDA/ROCm)
- [ ] TPU support via JAX integration
- [ ] Probabilistic programming
- [ ] Automatic hyperparameter tuning
- [ ] Model serving infrastructure

---

**Status:** Architecture specification complete  
**Next:** Begin Phase 1 implementation

