# Chapter 1: Introduction to Fowl

## 1.1 What is Fowl?

Fowl is a high-performance numerical computing library for F#. It provides:

- **Ndarray**: N-dimensional arrays (tensors) with broadcasting
- **Linear Algebra**: Matrix operations, decompositions, solvers
- **Statistics**: Descriptive stats, probability distributions, hypothesis tests
- **Signal Processing**: FFT, filters, convolution
- **Optimization**: Gradient descent, Adam, L-BFGS
- **Neural Networks**: Computation graphs, automatic differentiation

## 1.2 Why Fowl?

### Type Safety
```fsharp
// NumPy: Runtime errors
import numpy as np
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])
c = a + b  # Runtime error!

// Fowl: Type-safe with Result types
let a = Ndarray.ofArray [|1.0; 2.0; 3.0|] [||] |> unwrap
let b = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0|] [|2; 2|] |> unwrap
match Ndarray.add a b with
| Ok c -> printfn "Success: %A" c
| Error e -> printfn "Error handled gracefully: %A" e
```

### Performance
```fsharp
// SIMD-accelerated operations
open Fowl.SIMD

let a = Array.init 1000000 (fun _ -> Random.Shared.NextDouble())
let b = Array.init 1000000 (fun _ -> Random.Shared.NextDouble())

// 20-50x faster than naive implementation
let sum = ElementWise.add a b
```

### Composability
```fsharp
// Railway-oriented programming
let result = result {
    let! a = Ndarray.zeros<Float64> [|3; 3|]
    let! b = Ndarray.ones<Float64> [|3; 3|]
    let! c = Ndarray.add a b
    let! d = Ndarray.mul c b
    return d
}
```

## 1.3 Installation

### Via NuGet
```bash
dotnet add package Fowl
```

### From Source
```bash
git clone https://github.com/decoil/fowl.git
cd fowl
dotnet build
```

## 1.4 Your First Program

```fsharp
// FirstProgram.fsx
#r "nuget: Fowl"

open Fowl
open Fowl.Core.Types

// Create a 3x3 matrix of zeros
let zeros = Ndarray.zeros<Float64> [|3; 3|]
match zeros with
| Ok arr ->
    printfn "Created array with shape: %A" (Ndarray.shape arr)
    
    // Create a 1D array
    let! data = Ndarray.linspace 0.0 10.0 100
    printfn "Created 100 points from 0 to 10"
    
    // Element-wise operations
    let! squared = Ndarray.map (fun x -> x * x) data
    printfn "Squared all elements"
    
    // Reduction
    let sum = Ndarray.fold (+) 0.0 squared
    printfn "Sum of squares: %f" sum
| Error e ->
    printfn "Error: %s" e.Message
```

Run it:
```bash
dotnet fsi FirstProgram.fsx
```

## 1.5 Core Concepts

### The Ndarray Type

```fsharp
// 1D array (vector)
let vector = Ndarray.ofArray [|1.0; 2.0; 3.0|] [||]

// 2D array (matrix)
let matrix = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0|] [|2; 2|]

// 3D array (tensor)
let tensor = Ndarray.zeros<Float64> [|3; 4; 5|]
```

### Shape and Broadcasting

```fsharp
// Broadcasting example
let a = Ndarray.ones<Float64> [|3; 3|] |> unwrap  // 3x3 matrix
let b = Ndarray.ofArray [|10.0|] [||] |> unwrap    // scalar

// b is broadcast to match a's shape
match Ndarray.add a b with
| Ok c ->
    // c is 3x3 matrix where every element is 11.0
    printfn "Broadcasting works!"
```

### Result Types

Every operation that can fail returns a `Result`:

```fsharp
type FowlResult<'T> = Result<'T, FowlError>

type FowlError =
    | InvalidShape of string
    | DimensionMismatch of string
    | IndexOutOfRange of string
    | InvalidArgument of string
    | ...
```

Handle errors explicitly:

```fsharp
match Ndarray.reshape [|9|] matrix with
| Ok reshaped -> // Use reshaped array
| Error (InvalidShape msg) -> printfn "Invalid shape: %s" msg
| Error (DimensionMismatch msg) -> printfn "Dimensions don't match: %s" msg
| Error e -> printfn "Other error: %A" e
```

## 1.6 Exercises

### Exercise 1.1: Basic Array Creation

Create the following arrays:
1. A 5x5 identity matrix
2. A vector of 100 random numbers between 0 and 1
3. A 3D tensor with shape [2, 3, 4] filled with 5.0

```fsharp
// Solution
open Fowl
open Fowl.Core.Types
open Fowl.Linalg

// 1. Identity matrix
let identity = Core.eye<Float64> 5

// 2. Random vector
let random = 
    let rng = System.Random()
    Ndarray.ofArray (Array.init 100 (fun _ -> rng.NextDouble())) [||]

// 3. Filled tensor
let tensor = Ndarray.create<Float64> [|2; 3; 4|] 5.0
```

### Exercise 1.2: Error Handling

Write a function that safely adds two arrays, returning `None` if they can't be added:

```fsharp
let tryAdd (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : Ndarray<'K, float> option =
    match Ndarray.add a b with
    | Ok result -> Some result
    | Error _ -> None
```

### Exercise 1.3: Performance Comparison

Compare the performance of Fowl's SIMD operations vs standard F# array operations:

```fsharp
open System.Diagnostics

let benchmark (name: string) (f: unit -> 'a) (iterations: int) =
    let sw = Stopwatch()
    sw.Start()
    for _ in 1..iterations do
        f() |> ignore
    sw.Stop()
    printfn "%s: %d ms" name sw.ElapsedMilliseconds

// Run benchmarks
let size = 1000000
let a = Array.init size (fun _ -> Random.Shared.NextDouble())
let b = Array.init size (fun _ -> Random.Shared.NextDouble())

benchmark "Standard Array.map2" (fun () -> Array.map2 (+) a b) 100
benchmark "SIMD ElementWise.add" (fun () -> Fowl.SIMD.ElementWise.add a b) 100
```

## 1.7 Summary

Key takeaways:
- Fowl provides NumPy-like functionality with F# type safety
- All operations return `Result` types for explicit error handling
- Performance is competitive through SIMD and parallelization
- Functional programming patterns enable composable numerical code

## 1.8 Further Reading

- [F# for Fun and Profit](https://fsharpforfunandprofit.com/)
- [Owl Tutorial](https://ocaml.xyz/tutorial/)
- [Architecture of Advanced Numerical Analysis Systems](https://link.springer.com/book/10.1007/978-3-030-97636-9)

---

*Next: [Chapter 2: Ndarray Basics](chapter02.md)*
