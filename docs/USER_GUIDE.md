# Fowl User Guide

**The F# Numerical Computing Library**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/github-decoil/fowl-black.svg)](https://github.com/decoil/fowl)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Ndarray Operations](#ndarray-operations)
6. [Linear Algebra](#linear-algebra)
7. [Statistics](#statistics)
8. [Neural Networks](#neural-networks)
9. [Algorithmic Differentiation](#algorithmic-differentiation)
10. [Optimization](#optimization)
11. [Signal Processing](#signal-processing)
12. [Performance Guide](#performance-guide)
13. [Best Practices](#best-practices)
14. [Troubleshooting](#troubleshooting)

---

## Introduction

Fowl is a comprehensive numerical computing library for F#, inspired by [Owl](https://ocaml.xyz) (OCaml) and [NumPy](https://numpy.org) (Python). It provides:

- **Ndarray**: N-dimensional arrays with broadcasting
- **Linear Algebra**: Matrix operations, factorizations, solvers
- **Statistics**: Distributions, hypothesis tests, descriptive stats
- **Neural Networks**: Graph-based automatic differentiation
- **Signal Processing**: FFT, filtering, spectral analysis
- **Optimization**: Gradient-based and global optimization algorithms

### Why Fowl?

| Feature | Fowl | Owl (OCaml) | NumPy |
|---------|------|-------------|-------|
| Type Safety | ✅ Strong types | ✅ Strong types | ❌ Dynamic |
| Performance | ✅ Optimized | ✅ Optimized | ✅ Fast |
| Ecosystem | ✅ .NET | ⚠️ Limited | ✅ Massive |
| Functional | ✅ First-class | ✅ First-class | ❌ OOP |
| Cross-Platform | ✅ .NET Core | ✅ Native | ✅ Python |

---

## Installation

### Prerequisites

- .NET 8.0 SDK or later
- F# compiler (included with .NET SDK)

### Install from NuGet (when published)

```bash
dotnet add package Fowl
```

### Build from Source

```bash
git clone https://github.com/decoil/fowl.git
cd fowl
dotnet build
```

### Reference in Your Project

```xml
<ItemGroup>
  <ProjectReference Include="path/to/Fowl.Core.fsproj" />
  <ProjectReference Include="path/to/Fowl.Linalg.fsproj" />
  <ProjectReference Include="path/to/Fowl.Stats.fsproj" />
</ItemGroup>
```

---

## Quick Start

### Basic Ndarray Operations

```fsharp
open Fowl
open Fowl.Core.Types

// Create arrays
let a = Ndarray.ofArray [|1.0; 2.0; 3.0|] [|3|] |> Result.get
let b = Ndarray.zeros [|3; 3|] |> Result.get
let c = Ndarray.ones [|2; 2|] |> Result.get

// Mathematical operations
let sum = Ndarray.add a a |> Result.get
let product = Ndarray.mul a a |> Result.get
```

### Linear Algebra

```fsharp
open Fowl.Linalg

// Matrix creation
let A = Ndarray.ofArray2D (array2D [[1.0; 2.0]; [3.0; 4.0]]) |> Result.get

// Solve Ax = b
let b = Ndarray.ofArray [|5.0; 11.0|] [|2|] |> Result.get
let x = Factorizations.solve A b |> Result.get

// Matrix factorizations
let q, r = Factorizations.qr A |> Result.get
let u, s, vt = Factorizations.svd A |> Result.get
```

### Statistics

```fsharp
open Fowl.Stats

// Descriptive statistics
let data = [|1.0; 2.0; 3.0; 4.0; 5.0|]
let mean = Descriptive.mean data |> Result.get  // 3.0
let std = Descriptive.std data |> Result.get    // ~1.41

// Distributions
let normalPdf = GaussianDistribution.pdf 0.0 1.0 0.0 |> Result.get  // 0.399
let samples = GaussianDistribution.rvs 0.0 1.0 (Some 42) |> Result.get

// Hypothesis testing
let group1 = [|1.0; 2.0; 3.0|]
let group2 = [|4.0; 5.0; 6.0|]
let tResult = HypothesisTests.ttest_independent group1 group2 |> Result.get
```

### Neural Networks

```fsharp
open Fowl.Neural

// Build computation graph
let x = Graph.input "x" [|784|]
let W = Graph.parameter "W" [|784; 10|] (Array.init 7840 (fun _ -> 0.01))
let b = Graph.constantArray (Array.zeroCreate 10) [|10|]
let h = Graph.matmul x W |> Result.get
let y = Graph.add h b
let pred = Graph.activate Softmax y

// Forward pass
let inputs = Map ["x", Array.init 784 (fun _ -> 0.5)]
let output = Forward.runWithInputs pred inputs |> Result.get
```

---

## Core Concepts

### The Result Type

Fowl uses F#'s `Result<'T, 'Error>` type for error handling:

```fsharp
// Success case
let ok = Ok 42

// Error case  
let error = Error (FowlError.InvalidArgument "message")

// Pattern matching
match Ndarray.add a b with
| Ok result -> printfn "Success: %A" result
| Error e -> printfn "Error: %s" e.Message

// Using result computation expression
let computation = result {
    let! a = Ndarray.zeros [|3|]
    let! b = Ndarray.ones [|3|]
    return! Ndarray.add a b
}
```

### Type Safety

Fowl uses phantom types and shape information where possible:

```fsharp
// Ndarray is generic over element type
let intArray: Ndarray<int> = ...
let floatArray: Ndarray<float> = ...

// Shape information available at runtime
let shape = Ndarray.shape array  // [|3; 4; 5|]
```

### Broadcasting

Fowl supports NumPy-style broadcasting:

```fsharp
let a = Ndarray.ofArray [|1.0; 2.0; 3.0|] [|3|] |> Result.get  // Shape [|3|]
let b = Ndarray.ofArray [|10.0|] [|1|] |> Result.get            // Shape [|1|]

// Broadcasting: [|3|] + [|1|] -> [|3|]
let c = Ndarray.add a b |> Result.get  // [|11.0; 12.0; 13.0|]
```

---

## Ndarray Operations

### Creation

```fsharp
// From array
let a1 = Ndarray.ofArray [|1.0; 2.0; 3.0|] [|3|] |> Result.get

// Zeros and ones
let z = Ndarray.zeros [|3; 4|] |> Result.get
let o = Ndarray.ones [|2; 2; 2|] |> Result.get

// Identity matrix
let eye = Ndarray.eye 3 |> Result.get

// Range
let range = Ndarray.arange 0.0 10.0 1.0 |> Result.get  // [|0; 1; 2; ...; 9|]

// Linear space
let linspace = Ndarray.linspace 0.0 1.0 11 |> Result.get  // 11 points from 0 to 1
```

### Indexing and Slicing

```fsharp
let arr = Ndarray.ofArray2D (array2D [[1.0; 2.0; 3.0]; [4.0; 5.0; 6.0]]) |> Result.get

// Get element
let elem = Ndarray.get arr [|0; 1|] |> Result.get  // 2.0

// Slicing
let row = Slice.slice arr [|(0, 0); (0, 2)|] |> Result.get   // First row
let col = Slice.slice arr [|Slice.All; (1, 1)|] |> Result.get // Second column
```

### Mathematical Operations

```fsharp
let a = Ndarray.ofArray [|1.0; 2.0; 3.0|] [|3|] |> Result.get
let b = Ndarray.ofArray [|4.0; 5.0; 6.0|] [|3|] |> Result.get

// Arithmetic
let sum = Ndarray.add a b |> Result.get
let diff = Ndarray.sub a b |> Result.get
let prod = Ndarray.mul a b |> Result.get  // Element-wise
let quot = Ndarray.div a b |> Result.get

// Powers and roots
let squared = Ndarray.pow a 2.0 |> Result.get
let sqrted = Ndarray.map sqrt a |> Result.get

// Exponentials and logarithms
let exp_a = Ndarray.map exp a |> Result.get
let log_a = Ndarray.map log a |> Result.get
```

### Reductions

```fsharp
let arr = Ndarray.ofArray2D (array2D [[1.0; 2.0]; [3.0; 4.0]]) |> Result.get

// Sum all elements
let total = Ndarray.sum arr  // 10.0

// Sum along axis
let rowSums = Ndarray.sumAxis arr 0 |> Result.get  // Sum columns

// Other reductions
let maxVal = Ndarray.max arr
let minVal = Ndarray.min arr
let meanVal = Ndarray.mean arr
```

### Shape Manipulation

```fsharp
let arr = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0|] [|4|] |> Result.get

// Reshape
let matrix = Ndarray.reshape arr [|2; 2|] |> Result.get

// Transpose
let transposed = Matrix.transpose matrix |> Result.get

// Flatten
let flat = Ndarray.flatten matrix |> Result.get
```

---

## Linear Algebra

### Matrix Operations

```fsharp
open Fowl.Linalg

let A = Ndarray.ofArray2D (array2D [[1.0; 2.0]; [3.0; 4.0]]) |> Result.get
let B = Ndarray.ofArray2D (array2D [[5.0; 6.0]; [7.0; 8.0]]) |> Result.get

// Matrix multiplication
let C = Matrix.matmul A B |> Result.get

// Dot product
let dotProduct = Matrix.dot (Ndarray.flatten A |> Result.get) (Ndarray.flatten B |> Result.get) |> Result.get

// Outer product
let outer = Matrix.outer (Ndarray.flatten A |> Result.get) (Ndarray.flatten B |> Result.get) |> Result.get
```

### Factorizations

```fsharp
let A = Ndarray.ofArray2D (array2D [[4.0; 3.0]; [6.0; 3.0]]) |> Result.get

// LU decomposition
let L, U, P = Factorizations.lu A |> Result.get

// QR decomposition
let Q, R = Factorizations.qr A |> Result.get

// SVD
let U, S, Vt = Factorizations.svd A |> Result.get

// Cholesky (for positive definite matrices)
let PD = Ndarray.ofArray2D (array2D [[4.0; 2.0]; [2.0; 5.0]]) |> Result.get
let L = Factorizations.cholesky PD |> Result.get

// Eigenvalue decomposition (symmetric matrices)
let eigenvals, eigenvecs = Factorizations.eigSymmetric PD |> Result.get
```

### Solvers

```fsharp
let A = Ndarray.ofArray2D (array2D [[3.0; 1.0]; [1.0; 2.0]]) |> Result.get
let b = Ndarray.ofArray [|9.0; 8.0|] [|2|] |> Result.get

// Linear solve
let x = Factorizations.solve A b |> Result.get  // [|2.0; 3.0|]

// Least squares
let A_ls = Ndarray.ofArray2D (array2D [[1.0; 1.0]; [1.0; 2.0]; [1.0; 3.0]]) |> Result.get
let b_ls = Ndarray.ofArray [|2.0; 3.0; 5.0|] [|3|] |> Result.get
let x_ls, residual, rank, s = AdvancedOps.lstsq A_ls b_ls |> Result.get

// Pseudoinverse
let A_pinv = AdvancedOps.pinv A |> Result.get
```

### Matrix Properties

```fsharp
let A = Ndarray.ofArray2D (array2D [[1.0; 2.0]; [3.0; 4.0]]) |> Result.get

// Determinant
let det = Factorizations.det A |> Result.get

// Trace
let tr = Core.trace A |> Result.get

// Rank
let r = AdvancedOps.rank A |> Result.get

// Condition number
let cond = AdvancedOps.cond A |> Result.get

// Inverse
let inv = Factorizations.inv A |> Result.get
```

---

## Statistics

### Descriptive Statistics

```fsharp
open Fowl.Stats

let data = [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0|]

// Central tendency
let mean = Descriptive.mean data |> Result.get
let median = Descriptive.median data |> Result.get
let mode = Descriptive.mode data  // May return None

// Dispersion
let variance = Descriptive.var data |> Result.get
let stdDev = Descriptive.std data |> Result.get
let range = Descriptive.range data |> Result.get

// Shape
let skew = Descriptive.skewness data |> Result.get
let kurt = Descriptive.kurtosis data |> Result.get

// Percentiles
let p25 = Descriptive.percentile data 25.0 |> Result.get
let p75 = Descriptive.percentile data 75.0 |> Result.get
let iqr = DescriptiveExtended.iqr data |> Result.get
```

### Distributions

```fsharp
// Normal (Gaussian)
let normalPdf = GaussianDistribution.pdf 0.0 1.0 0.0 |> Result.get
let normalCdf = GaussianDistribution.cdf 0.0 1.0 1.96 |> Result.get
let normalPpf = GaussianDistribution.ppf 0.0 1.0 0.975 |> Result.get
let normalSample = GaussianDistribution.rvs 0.0 1.0 (Some 42) |> Result.get

// Other continuous distributions
let gammaSample = GammaDistribution.rvs 2.0 1.0 (Some 42) |> Result.get
let betaSample = BetaDistribution.rvs 2.0 5.0 (Some 42) |> Result.get
let studentTSample = StudentTDistribution.rvs 10.0 (Some 42) |> Result.get
let chi2Sample = ChiSquareDistribution.rvs 3.0 (Some 42) |> Result.get
let fSample = FDistribution.rvs 5.0 10.0 (Some 42) |> Result.get

// Discrete distributions
let binomialSample = BinomialDistribution.rvs 10 0.3 (Some 42) |> Result.get
let poissonSample = PoissonDistribution.rvs 2.5 (Some 42) |> Result.get
let geometricSample = GeometricDistribution.rvs 0.3 (Some 42) |> Result.get
```

### Hypothesis Testing

```fsharp
// One-sample t-test
let data = [|1.2; 1.5; 1.3; 1.7; 1.4|]
let tResult = HypothesisTests.ttest_one_sample data 1.0 |> Result.get
printfn "t-statistic: %.4f, p-value: %.4f" tResult.Statistic tResult.PValue

// Independent two-sample t-test
let group1 = [|1.0; 2.0; 3.0|]
let group2 = [|4.0; 5.0; 6.0|]
let tIndResult = HypothesisTests.ttest_independent group1 group2 |> Result.get

// Chi-square test
let observed = [|10.0; 20.0; 30.0|]
let expected = [|15.0; 15.0; 30.0|]
let chi2Result = HypothesisTests.chi2_goodness observed expected |> Result.get

// ANOVA
let groups = [|[|1.0; 2.0; 3.0|]; [|4.0; 5.0; 6.0|]; [|7.0; 8.0; 9.0|]|]
let anovaResult = Anova.oneWay groups |> Result.get
printfn "F-statistic: %.4f, p-value: %.4f" anovaResult.FStatistic anovaResult.PValue

// Non-parametric tests
let kwResult = NonParametricTests.kruskalWallis groups |> Result.get
let mannWhitneyResult = NonParametricTests.mannWhitneyU group1 group2 |> Result.get
```

### Correlation

```fsharp
let x = [|1.0; 2.0; 3.0; 4.0; 5.0|]
let y = [|2.0; 4.0; 6.0; 8.0; 10.0|]

// Pearson correlation
let pearsonR = Correlation.pearsonCorrelation x y |> Result.get

// Spearman rank correlation
let spearmanRho = RankCorrelation.spearman x y |> Result.get

// Kendall tau correlation
let kendallTau = RankCorrelation.kendall x y |> Result.get

// Correlation matrix
let data = [|[|1.0; 2.0|]; [|2.0; 3.0|]; [|3.0; 4.0|]|]
let corrMatrix = Correlation.correlationMatrix data |> Result.get
```

---

## Neural Networks

### Building a Network

```fsharp
open Fowl.Neural
open Fowl.Neural.Layers

// Create layers
let dense1 = Layers.dense 784 256 (Some ReLU) (Some 42) |> Result.get
let dense2 = Layers.dense 256 10 (Some Softmax) (Some 42) |> Result.get

// Build computation graph
let input = Graph.input "x" [|784|]
let h1 = forwardDense dense1 input |> Result.get
let output = forwardDense dense2 h1 |> Result.get
let target = Graph.input "y" [|10|]
let loss = Loss.crossEntropy output target

// Training configuration
let config = {
    Epochs = 10
    BatchSize = 32
    LearningRate = 0.001
    Verbose = true
}
```

### Training

```fsharp
// Create optimizer
let optimizer = Optimizer.adam 0.001 0.9 0.999 1e-8

// Training loop (simplified)
for epoch = 1 to config.Epochs do
    // Forward pass
    Forward.run [loss] |> ignore
    
    // Backward pass
    Backward.run [loss] |> ignore
    
    // Update parameters
    let params = getParameters [dense1; dense2]
    Optimizer.updateAdam optimizer params
    
    printfn "Epoch %d: Loss = %.4f" epoch loss.Value.Value
```

### Recurrent Networks

```fsharp
open Fowl.Neural.RecurrentLayers

// LSTM for sequence modeling
let lstm = createLSTM inputSize=10 hiddenSize=20 numLayers=2 |> Result.get

// Input: [seq_len; batch; input_size]
let sequence = Array.init 50 (fun _ -> Array.init 32 (fun _ -> Array.init 10 (fun _ -> 0.5)))
let outputs = lstmForward lstm sequence

// GRU (simpler alternative)
let gru = createGRU inputSize=10 hiddenSize=20 numLayers=1 |> Result.get
let gruOutputs = gruForward gru sequence
```

### Advanced Optimizers

```fsharp
open Fowl.Neural.AdvancedOptimizers

// Adagrad
let adagradState = createAdagrad 0.01 1e-8 numParameters
updateAdagrad adagradState parameters gradients

// RMSprop
let rmspropState = createRMSprop 0.001 0.9 1e-8 numParameters
updateRMSprop rmspropState parameters gradients

// Adamax
let adamaxState = createAdamax 0.002 0.9 0.999 1e-8 numParameters
updateAdamax adamaxState parameters gradients
```

---

## Algorithmic Differentiation

### Forward Mode

```fsharp
open Fowl.AD

// Compute derivative of f(x) = x² at x = 3
let f x = x * x
let df = diff f 3.0  // 6.0

// Gradient of f(x,y) = x² + y² at (3, 4)
let g (x: Dual) (y: Dual) = x * x + y * y
let grad = grad2 g 3.0 4.0  // [|6.0; 8.0|]

// Jacobian
let fVec (x: float[]) = [|x.[0] * x.[1]; x.[0] + x.[1]|]
let jac = jacobian fVec [|2.0; 3.0|]
```

### Reverse Mode

```fsharp
// Using computation graph (recommended)
let x = Graph.input "x" [||]
let y = Graph.mul x x
let grad = Graph.grad y x

// Higher-order derivatives
let f x = sin x
let f' = diff f  // First derivative
let f'' = diff f'  // Second derivative
```

---

## Optimization

### Gradient-Based Optimization

```fsharp
open Fowl.Optimization

// Define objective function
let rosenbrock (x: float[]) =
    (1.0 - x.[0]) ** 2.0 + 100.0 * (x.[1] - x.[0] ** 2.0) ** 2.0

// Gradient
let rosenbrockGrad (x: float[]) =
    let dx0 = -2.0 * (1.0 - x.[0]) - 400.0 * x.[0] * (x.[1] - x.[0] ** 2.0)
    let dx1 = 200.0 * (x.[1] - x.[0] ** 2.0)
    [|dx0; dx1|]

// Optimize
let options = { defaultOptions with MaxIter = 1000; LearningRate = 0.001 }
let result = GradientDescent.minimize rosenbrock rosenbrockGrad [|0.0; 0.0|] options

printfn "Optimum: x = %.6f, y = %.6f" result.X.[0] result.X.[1]
printfn "Function value: %.10f" result.Fun
```

### Global Optimization

```fsharp
// Simulated annealing for non-convex problems
let bounds = [|(-5.0, 5.0); (-5.0, 5.0)|]
let saResult = SimulatedAnnealing.minimize rosenbrock [|0.0; 0.0|] bounds options

printfn "SA found: x = %.6f, y = %.6f" saResult.X.[0] saResult.X.[1]
```

---

## Signal Processing

### FFT

```fsharp
open Fowl.FFT

// 1D FFT
let signal = Array.init 256 (fun i -> sin (2.0 * System.Math.PI * 50.0 * float i / 256.0))
let signalComplex = signal |> Array.map (fun x -> Complex(x, 0.0))
let fftResult = FFT.fft signalComplex |> Result.get

// Inverse FFT
let reconstructed = FFT.ifft fftResult |> Result.get

// 2D FFT for images
let image = Array2D.init 64 64 (fun i j -> ...)
let imageComplex = Array2D.map (fun x -> Complex(x, 0.0)) image
let fft2Result = FFT.fft2 imageComplex |> Result.get
```

### Filtering

```fsharp
open Fowl.FFT.SignalFilters

// Gaussian filter (smooth signal)
let noisySignal = Array.init 1000 (fun _ -> ...)
let smoothed = gaussianFilter1D noisySignal 2.0 |> Result.get

// Median filter (remove outliers)
let spikeSignal = [|...spikes...|]
let clean = medianFilter1D spikeSignal 5 |> Result.get

// Moving average
let averaged = movingAverage signal 10 |> Result.get
```

### Spectral Analysis

```fsharp
// Power spectral density
let psd = FFT.psd fftResult

// Frequencies
let freqs = FFT.fftfreq 256 0.001  // Sample rate 1000 Hz

// Spectrogram
let spectro = SignalProcessing.spectrogram signal 256 128 |> Result.get
```

---

## Performance Guide

### Automatic Optimization

Fowl automatically selects the best implementation:

```fsharp
open Fowl.Config

// Initialize (auto-detects hardware)
Config.initialize()

// All operations now use best available implementation
let result = Ndarray.add a b  // Uses SIMD if available
```

### Optimization Levels

| Level | Implementation | Speedup |
|-------|---------------|---------|
| 1 | Scalar (baseline) | 1x |
| 2 | Portable SIMD (Vector&lt;T&gt;) | 2-4x |
| 3 | Hardware SIMD (AVX2/SSE2) | 4-8x |
| 4 | Parallel + SIMD | 8-32x |

### Manual Control

```fsharp
// Use specific implementation
let simdResult = Fowl.SIMD.ElementWise.add a b
let avxResult = Fowl.Native.SIMD.Avx2Kernels.addArrays a b

// Parallel operations
let parallelResult = Fowl.Parallel.ParallelOps.parallelMap f a
```

### Memory Optimization

```fsharp
// Zero-copy slicing with NdarrayView
let view = NdarrayView.row matrix 0

// Use ArrayPool for temporary buffers
use buffer = ArrayPoolOps.rent 10000

// In-place operations to avoid allocations
Ndarray.addInPlace a b  // Modifies a in place
```

### Benchmarking

```bash
cd benchmarks
./run-benchmarks.sh
```

---

## Best Practices

### 1. Use Result Types

```fsharp
// Good: Handle errors explicitly
match Ndarray.add a b with
| Ok result -> useResult result
| Error e -> handleError e

// Good: Use computation expressions
let computation = result {
    let! x = someOperation
    let! y = anotherOperation x
    return y
}
```

### 2. Preallocate Where Possible

```fsharp
// Avoid repeated allocations in loops
let results = Array.zeroCreate n
for i = 0 to n - 1 do
    results.[i] <- compute i  // Reuse array
```

### 3. Use Appropriate Data Types

```fsharp
// Use float32 for memory efficiency when precision allows
let float32Array: Ndarray<float32> = ...

// Use int for discrete data
let intArray: Ndarray<int> = ...
```

### 4. Batch Operations

```fsharp
// Good: Process batches
let batchResults = Array.map processBatch batches

// Avoid: Processing one by one in tight loop
```

### 5. Profile Before Optimizing

```bash
# Use BenchmarkDotNet
dotnet run --project benchmarks/Fowl.Benchmarks -c Release
```

---

## Troubleshooting

### Common Errors

#### "Invalid shape for operation"

```fsharp
// Check shapes match
let shapeA = Ndarray.shape a
let shapeB = Ndarray.shape b
printfn "Shape A: %A, Shape B: %A" shapeA shapeB

// Use broadcasting or reshape
let aReshaped = Ndarray.reshape a [|3; 1|] |> Result.get
```

#### "Matrix is singular"

```fsharp
// Use pseudoinverse instead of inverse
let pinvA = AdvancedOps.pinv A |> Result.get

// Or add regularization
let A_reg = Ndarray.add A (Ndarray.eye n scalar=1e-6 |> Result.get) |> Result.get
```

#### "Out of memory"

```fsharp
// Use float32 instead of float64
// Process in chunks
// Use ArrayPool for temporaries
```

### Debugging Tips

1. **Enable verbose mode**
   ```fsharp
   Config.initialize(verbose=true)
   ```

2. **Check intermediate values**
   ```fsharp
   printfn "Intermediate: %A" intermediateResult
   ```

3. **Validate inputs**
   ```fsharp
   if Array.isEmpty data then
       failwith "Empty input"
   ```

### Getting Help

- **Documentation**: https://github.com/decoil/fowl/tree/main/docs
- **Issues**: https://github.com/decoil/fowl/issues
- **Examples**: See `examples/` directory

---

## Migration from Other Libraries

### From NumPy

| NumPy | Fowl |
|-------|------|
| `np.array([1,2,3])` | `Ndarray.ofArray [|1;2;3|] [\|3\|]` |
| `np.zeros((3,3))` | `Ndarray.zeros [\|3;3\|]` |
| `np.dot(a,b)` | `Matrix.dot a b` |
| `np.linalg.inv(a)` | `Factorizations.inv a` |
| `np.random.normal()` | `GaussianDistribution.rvs 0.0 1.0 None` |

### From Owl (OCaml)

Fowl follows Owl's API closely. Most function names are identical.

```fsharp
// Owl: Mat.uniform 3 3
// Fowl:
Ndarray.random [\|3;3\|] (Some 42) |> Result.get

// Owl: Linalg.D.inv a
// Fowl:
Factorizations.inv a |> Result.get
```

---

## License

MIT License - See [LICENSE](LICENSE) file

---

## Acknowledgments

- Inspired by [Owl](https://ocaml.xyz) - OCaml numerical library
- Architecture based on "Architecture of Advanced Numerical Analysis Systems"
- Built with [F#](https://fsharp.org) and [.NET](https://dotnet.microsoft.com)

---

_Last Updated: 2026-02-15_