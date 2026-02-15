# Chapter 2: Ndarray Basics

## 2.1 Creating Arrays

### From Scratch

```fsharp
open Fowl
open Fowl.Core.Types

// Zeros and ones
let zeros = Ndarray.zeros<Float64> [|3; 4|]    // 3x4 matrix of zeros
let ones = Ndarray.ones<Float64> [|2; 2; 2|]  // 2x2x2 tensor of ones
let filled = Ndarray.create<Float64> [|5|] 3.14  // Vector of 3.14s

// Empty (uninitialized - use with caution!)
let empty = Ndarray.empty<Float64> [|100; 100|]
```

### From Data

```fsharp
// From flat array
let fromArray = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0|] [|2; 2|]

// From nested lists (convenience)
let fromList2D = 
    [[1.0; 2.0; 3.0]
     [4.0; 5.0; 6.0]
     [7.0; 8.0; 9.0]]
    |> List.concat
    |> List.toArray
    |> fun arr -> Ndarray.ofArray arr [|3; 3|]

// Arange (like Python's range)
let range = Ndarray.arange 0.0 10.0 2.0  // [|0; 2; 4; 6; 8|]

// Linspace
let spaced = Ndarray.linspace 0.0 1.0 5   // [|0; 0.25; 0.5; 0.75; 1|]
```

### Random Arrays

```fsharp
open Fowl.Stats.Random

let rng = RandomState.create 42  // Seeded RNG

// Uniform [0, 1)
let uniform = RandomState.rand rng [|1000|]

// Standard normal
let normal = RandomState.randn rng [|1000|]

// Integer random
let integers = RandomState.randint rng 0 10 [|100|]
```

## 2.2 Array Properties

```fsharp
let arr = Ndarray.zeros<Float64> [|3; 4; 5|] |> unwrap

// Shape
let shape = Ndarray.shape arr  // [|3; 4; 5|]

// Number of dimensions (rank)
let ndim = Ndarray.ndim arr    // 3

// Total elements
let size = Ndarray.numel arr   // 60

// Data type (via phantom type)
// arr : Ndarray<Float64, float>
```

## 2.3 Indexing and Slicing

### Basic Indexing

```fsharp
let arr = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0; 5.0|] [||] |> unwrap

// Get element
match Ndarray.get arr [|2|] with  // 3.0
| Ok v -> printfn "Element at index 2: %f" v
| Error e -> printfn "Error: %A" e

// Set element
match Ndarray.set arr [|2|] 10.0 with
| Ok () -> printfn "Set successfully"
| Error e -> printfn "Error: %A" e
```

### Slicing

```fsharp
open Fowl.Core.Slice

let matrix = 
    Ndarray.ofArray [|1.0 .. 25.0|] [|5; 5|] 
    |> unwrap

// Row slice (first row)
let row = Slice.slice matrix [|Range (Some 0, Some 1, None); All|]

// Column slice (middle column)
let col = Slice.slice matrix [|All; Index 2|]

// Sub-matrix
let sub = Slice.slice matrix 
    [|Range (Some 1, Some 4, None); 
      Range (Some 1, Some 4, None)|]

// Every other row
let strided = Slice.slice matrix 
    [|Range (Some 0, None, Some 2); All|]
```

### Advanced Indexing

```fsharp
// Boolean indexing (filtering)
let data = Ndarray.ofArray [|1.0; -2.0; 3.0; -4.0; 5.0|] [||] |> unwrap
let positive = 
    Ndarray.toArray data
    |> Array.filter (fun x -> x > 0.0)
    |> fun arr -> Ndarray.ofArray arr [||]

// Fancy indexing (select specific indices)
let indices = [|0; 2; 4|]
let selected = 
    indices
    |> Array.map (fun i -
        match Ndarray.get data [|i|] with
        | Ok v -> v
        | Error _ -> 0.0)
    |> fun arr -> Ndarray.ofArray arr [||]
```

## 2.4 Reshaping and Transposing

### Reshape

```fsharp
let arr = Ndarray.ofArray [|1.0 .. 12.0|] [||] |> unwrap

// Reshape to 3x4
let matrix = Ndarray.reshape [|3; 4|] arr |> unwrap

// Reshape to 2x2x3
let tensor = Ndarray.reshape [|2; 2; 3|] arr |> unwrap

// Flatten
let flat = Ndarray.reshape [|-1|] matrix |> unwrap
// Shape: [|12|]
```

### Transpose

```fsharp
open Fowl.Core.Matrix

let matrix = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0|] [|2; 2|] |> unwrap

// Transpose
let transposed = transpose matrix |> unwrap
// [|1; 3|
//  |2; 4|]

// Swap axes (for higher dimensions)
let tensor3D = Ndarray.zeros<Float64> [|2; 3; 4|] |> unwrap
// Shape: [2; 3; 4]

// Swap axis 0 and 2
let swapped = NdarrayOps.swapAxes tensor3D 0 2 |> unwrap
// Shape: [4; 3; 2]
```

## 2.5 Broadcasting

Broadcasting allows operations on arrays of different shapes:

```fsharp
// Scalar broadcasting
let a = Ndarray.ones<Float64> [|3; 3|] |> unwrap
let b = Ndarray.ofArray [|5.0|] [||] |> unwrap

// 5.0 is broadcast to 3x3
match Ndarray.add a b with
| Ok c -> 
    // c is 3x3 filled with 6.0
    printfn "Broadcast scalar: %A" (Ndarray.toArray c)
| Error e -> printfn "Error: %A" e

// Vector broadcasting
let matrix = Ndarray.ones<Float64> [|3; 4|] |> unwrap
let row = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0|] [||] |> unwrap

// row is broadcast across rows
match Ndarray.add matrix row with
| Ok c ->
    // Each row: [2; 3; 4; 5]
    printfn "Broadcast row"
| Error e -> printfn "Error: %A" e

// Column broadcasting
let col = Ndarray.ofArray [|10.0; 20.0; 30.0|] [||] |> unwrap
let colBroadcast = Ndarray.reshape [|-1; 1|] col |> unwrap

match Ndarray.add matrix colBroadcast with
| Ok c ->
    // Row 0: [11; 11; 11; 11]
    // Row 1: [21; 21; 21; 21]
    // Row 2: [31; 31; 31; 31]
    printfn "Broadcast column"
| Error e -> printfn "Error: %A" e
```

## 2.6 Element-wise Operations

### Arithmetic

```fsharp
let a = Ndarray.ofArray [|1.0; 2.0; 3.0|] [||] |> unwrap
let b = Ndarray.ofArray [|4.0; 5.0; 6.0|] [||] |> unwrap

let sum = Ndarray.add a b |> unwrap        // [|5; 7; 9|]
let diff = Ndarray.sub a b |> unwrap       // [|-3; -3; -3|]
let prod = Ndarray.mul a b |> unwrap       // [|4; 10; 18|]
let quot = Ndarray.div a b |> unwrap       // [|0.25; 0.4; 0.5|]
```

### Mathematical Functions

```fsharp
let x = Ndarray.linspace 0.0 (System.Math.PI) 100 |> unwrap

let sinX = Ndarray.map sin x
let cosX = Ndarray.map cos x
let expX = Ndarray.map exp x
let logX = Ndarray.map log (Ndarray.add x (Ndarray.ones<Float64> [|100|] |> unwrap) |> unwrap)
```

### Comparison Operations

```fsharp
let data = Ndarray.ofArray [|1.0; 5.0; 3.0; 8.0; 2.0|] [||] |> unwrap

// Element-wise comparison
let greaterThan3 = 
    Ndarray.toArray data
    |> Array.map (fun x -> if x > 3.0 then 1.0 else 0.0)
    |> fun arr -> Ndarray.ofArray arr [||]

// Clip values
let clipped = NdarrayOps.clip data 2.0 6.0 |> unwrap
// [|2; 5; 3; 6; 2|]
```

## 2.7 Reduction Operations

```fsharp
let matrix = Ndarray.ofArray [|1.0 .. 12.0|] [|3; 4|] |> unwrap
// [| 1  2  3  4 |
//  | 5  6  7  8 |
//  | 9 10 11 12 |]

// Sum all elements
let total = Ndarray.fold (+) 0.0 matrix  // 78.0

// Sum along axis
open Fowl.Core.NdarrayOps

let rowSums = sumAxis matrix 1 |> unwrap   // [|10; 26; 42|]
let colSums = sumAxis matrix 0 |> unwrap   // [|15; 18; 21; 24|]

// Other reductions
let prod = prodAxis matrix 1 |> unwrap
let mean = meanAxis matrix 1 |> unwrap
let max = maxAxis matrix 1 |> unwrap
let min = minAxis matrix 1 |> unwrap
```

## 2.8 Stacking and Concatenating

```fsharp
let a = Ndarray.ones<Float64> [|2; 3|] |> unwrap
let b = Ndarray.ones<Float64> [|2; 3|] |> unwrap

// Vertical stack (concatenate rows)
let vstack = Matrix.stack 0 [a; b] |> unwrap
// Shape: [|4; 3|]

// Horizontal stack (concatenate columns)
let hstack = Matrix.stack 1 [a; b] |> unwrap
// Shape: [|2; 6|]

// Depth stack (for 3D+)
let c = Ndarray.ones<Float64> [|2; 3|] |> unwrap
delete stack = Matrix.stack 2 [a; b; c] |> unwrap
// Shape: [|2; 3; 3|]
```

## 2.9 Splitting Arrays

```fsharp
let arr = Ndarray.ofArray [|1.0 .. 12.0|] [|3; 4|] |> unwrap

// Split into 3 equal parts along columns
let parts = NdarrayOps.split arr 1 3 |> unwrap
// 3 arrays of shape [|3; 1|] each

// Split at specific indices
let splitAt = NdarrayOps.splitAt arr 1 [|1; 3|] |> unwrap
// Splits at columns 1 and 3
```

## 2.10 Exercises

### Exercise 2.1: Array Creation Patterns

Create the following arrays without using explicit loops:

1. A 10x10 matrix where element (i,j) = i + j
2. A checkerboard pattern (alternating 0s and 1s)
3. A matrix with 1s on the diagonal and 0s elsewhere (without using `eye`)

```fsharp
// Solutions

// 1. i + j pattern
let pattern =
    Array.init 10 (fun i -
        Array.init 10 (fun j -
            float (i + j)))
    |> Array.concat
    |> fun arr -> Ndarray.ofArray arr [|10; 10|]

// 2. Checkerboard
let checkerboard =
    Array.init 100 (fun idx -
        let i = idx / 10
        let j = idx % 10
        if (i + j) % 2 = 0 then 1.0 else 0.0)
    |> fun arr -> Ndarray.ofArray arr [|10; 10|]

// 3. Identity-like
let identityLike =
    Array.init 100 (fun idx -
        let i = idx / 10
        let j = idx % 10
        if i = j then 1.0 else 0.0)
    |> fun arr -> Ndarray.ofArray arr [|10; 10|]
```

### Exercise 2.2: Normalization

Implement z-score normalization:

```fsharp
let zscoreNormalize (arr: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
    result {
        let data = Ndarray.toArray arr
        let mean = Array.average data
        let variance = 
            data
            |> Array.map (fun x -> (x - mean) ** 2.0)
            |> Array.average
        let std = sqrt variance
        
        let normalized =
            data
            |> Array.map (fun x -> (x - mean) / std)
        
        return! Ndarray.ofArray normalized (Ndarray.shape arr)
    }
```

### Exercise 2.3: Moving Average

Implement a moving average function:

```fsharp
let movingAverage (arr: Ndarray<'K, float>) (window: int) : FowlResult<Ndarray<'K, float>> =
    result {
        let data = Ndarray.toArray arr
        let n = data.Length
        
        if window <= 0 || window > n then
            return! Error.invalidArgument "Invalid window size"
        
        let result = Array.zeroCreate (n - window + 1)
        
        for i = 0 to n - window do
            let sum = data.[i .. i + window - 1] |> Array.sum
            result.[i] <- sum / float window
        
        return! Ndarray.ofArray result [||]
    }

// Usage
let data = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0; 5.0|] [||] |> unwrap
let ma = movingAverage data 3 |> unwrap
// [|2.0; 3.0; 4.0|]
```

## 2.11 Summary

Key concepts:
- Arrays are created with `zeros`, `ones`, `ofArray`, `arange`, `linspace`
- Shape can be inspected with `shape`, `ndim`, `numel`
- Indexing uses `get`/`set`, slicing uses `Slice.slice`
- Broadcasting enables operations between different shapes
- Reductions collapse dimensions with `fold`, `sumAxis`, etc.
- Stacking combines arrays along new or existing dimensions

## 2.12 Common Pitfalls

1. **Forgetting to unwrap Results**
```fsharp
// Wrong
let sum = Ndarray.add a b  // This is Result<Ndarray, _>, not Ndarray

// Right
let sum = Ndarray.add a b |> unwrap
```

2. **Shape mismatches in broadcasting**
```fsharp
// Wrong - incompatible shapes
let a = Ndarray.zeros<Float64> [|3; 4|] |> unwrap
let b = Ndarray.zeros<Float64> [|5; 4|] |> unwrap
// Ndarray.add a b will fail
```

3. **Modifying arrays in place**
```fsharp
// Arrays are not mutable by default
let arr = Ndarray.ones<Float64> [|3|] |> unwrap
// Ndarray.set modifies in place, but most operations return new arrays
let doubled = Ndarray.map (fun x -> x * 2.0) arr  // New array
```

---

*Next: [Chapter 3: Array Manipulation](chapter03.md)*
