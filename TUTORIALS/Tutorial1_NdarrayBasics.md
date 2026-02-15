# Tutorial 1: Ndarray Basics

Learn the fundamentals of Fowl's Ndarray type for numerical computing.

## Overview

The `Ndarray<'T>` type is Fowl's core data structure for n-dimensional arrays. It's similar to NumPy's ndarray but with F#'s type safety.

## Learning Objectives

By the end of this tutorial, you will:
- Create NDArrays of various shapes
- Perform element-wise operations
- Understand broadcasting
- Use slicing and indexing
- Apply mathematical functions

## Setup

```fsharp
open Fowl
open Fowl.Core.Types

// Helper to unwrap results
let unwrap = function Ok x -> x | Error e -> failwith e.Message
```

## Creating Arrays

### From F# Arrays

```fsharp
// 1D array
let arr1D = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0; 5.0|] [|5|] |> unwrap

// 2D array (matrix)
let arr2D = Ndarray.ofArray2D (array2D [[1.0; 2.0]; [3.0; 4.0]]) |> unwrap

// From existing data
let data = [|1.0 .. 100.0|]
let largeArray = Ndarray.ofArray data [|100|] |> unwrap
```

### Convenience Functions

```fsharp
// Zeros and ones
let zeros = Ndarray.zeros [|3; 4|] |> unwrap    // 3x4 matrix of zeros
let ones = Ndarray.ones [|2; 2; 2|] |> unwrap   // 2x2x2 tensor of ones

// Identity matrix
let eye3 = Ndarray.eye 3 |> unwrap  // 3x3 identity matrix

// Ranges
let range = Ndarray.arange 0.0 10.0 1.0 |> unwrap  // [|0; 1; 2; ...; 9|]

// Linear spacing
let linear = Ndarray.linspace 0.0 1.0 11 |> unwrap  // 11 points from 0 to 1
```

## Basic Operations

### Arithmetic

```fsharp
let a = Ndarray.ofArray [|1.0; 2.0; 3.0|] [|3|] |> unwrap
let b = Ndarray.ofArray [|4.0; 5.0; 6.0|] [|3|] |> unwrap

// Element-wise operations
let sum = Ndarray.add a b |> unwrap        // [|5; 7; 9|]
let diff = Ndarray.sub a b |> unwrap       // [|-3; -3; -3|]
let prod = Ndarray.mul a b |> unwrap       // [|4; 10; 18|]
let quot = Ndarray.div a b |> unwrap       // [|0.25; 0.4; 0.5|]

// Scalar operations
let doubled = Ndarray.mulScalar a 2.0 |> unwrap  // [|2; 4; 6|]
```

### Mathematical Functions

```fsharp
let x = Ndarray.linspace 0.0 (System.Math.PI) 100 |> unwrap

// Apply functions element-wise
let sinX = Ndarray.map sin x |> unwrap
let cosX = Ndarray.map cos x |> unwrap
let expX = Ndarray.map exp x |> unwrap
let logX = Ndarray.map log (Ndarray.addScalar x 1.0 |> unwrap) |> unwrap

// Powers
let squared = Ndarray.pow x 2.0 |> unwrap
let sqrted = Ndarray.map sqrt x |> unwrap
```

## Shape and Reshaping

```fsharp
let flat = Ndarray.ofArray [|1.0 .. 12.0|] [|12|] |> unwrap

// Reshape
let matrix = Ndarray.reshape flat [|3; 4|] |> unwrap  // 3x4 matrix
let tensor = Ndarray.reshape flat [|2; 2; 3|] |> unwrap  // 2x2x3 tensor

// Get shape
let shape = Ndarray.shape matrix  // [|3; 4|]
let numElements = Ndarray.size matrix  // 12

// Transpose
let transposed = Matrix.transpose matrix |> unwrap  // 4x3 matrix

// Flatten
let backToFlat = Ndarray.flatten matrix |> unwrap
```

## Indexing and Slicing

### Basic Indexing

```fsharp
let mat = Ndarray.ofArray2D (array2D [[1.0; 2.0; 3.0]; 
                                        [4.0; 5.0; 6.0]; 
                                        [7.0; 8.0; 9.0]]) |> unwrap

// Get element
let elem = Ndarray.get mat [|1; 2|] |> unwrap  // 6.0 (row 1, col 2)

// Set element
let modified = Ndarray.copy mat |> unwrap
Ndarray.set modified [|0; 0|] 99.0
```

### Slicing

```fsharp
// Get row
let row1 = Slice.slice mat [|(1, 1); Slice.All|] |> unwrap  // [|4; 5; 6|]

// Get column
let col1 = Slice.slice mat [|Slice.All; (1, 1)|] |> unwrap  // [|2; 5; 8|]

// Get sub-matrix
let sub = Slice.slice mat [|(0, 1); (1, 2)|] |> unwrap  // 2x2 matrix
```

## Broadcasting

```fsharp
// Arrays with different but compatible shapes
let a = Ndarray.ofArray [|1.0; 2.0; 3.0|] [|3|] |> unwrap  // Shape [|3|]
let b = Ndarray.ofArray [|10.0|] [|1|] |> unwrap              // Shape [|1|]

// Broadcasting: [|3|] + [|1|] -> [|3|]
let result = Ndarray.add a b |> unwrap  // [|11; 12; 13|]

// 2D broadcasting
let matrix = Ndarray.ones [|3; 4|] |> unwrap  // 3x4 matrix
let row = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0|] [|4|] |> unwrap  // 4 elements

// Add row to each row of matrix
// Shape [|3;4|] + [|4|] -> [|3;4|] (row is broadcast)
let added = Ndarray.add matrix row |> unwrap
```

## Reductions

```fsharp
let data = Ndarray.ofArray2D (array2D [[1.0; 2.0; 3.0]; 
                                        [4.0; 5.0; 6.0]]) |> unwrap

// Sum all elements
let total = Ndarray.sum data  // 21.0

// Sum along axis
let rowSums = Ndarray.sumAxis data 0 |> unwrap  // [|6; 15|] (sum columns)
let colSums = Ndarray.sumAxis data 1 |> unwrap  // [|5; 7; 9|] (sum rows)

// Other reductions
let maximum = Ndarray.max data  // 6.0
let minimum = Ndarray.min data  // 1.0
let avg = Ndarray.mean data     // 3.5
```

## Practical Example: Normalizing Data

```fsharp
// Sample data: exam scores
let scores = Ndarray.ofArray [|65.0; 72.0; 88.0; 91.0; 55.0; 78.0|] [|6|] |> unwrap

// Calculate mean and std
let meanVal = Ndarray.mean scores
let stdVal = Ndarray.std scores

// Z-score normalization
let normalized = 
    scores
    |> Ndarray.map (fun x -> (x - meanVal) / stdVal)
    |> unwrap

printfn "Original: %A" (Ndarray.toArray scores)
printfn "Normalized: %A" (Ndarray.toArray normalized)
```

## Exercises

1. Create a 5x5 identity matrix and multiply it by a scalar
2. Create two 3x3 matrices and compute their element-wise product
3. Extract the diagonal from a 4x4 matrix
4. Normalize a 1D array to have values between 0 and 1
5. Compute the mean of each column in a 5x3 matrix

## Solutions

```fsharp
// Exercise 1
let eye5 = Ndarray.eye 5 |> unwrap
let scaled = Ndarray.mulScalar eye5 3.0 |> unwrap

// Exercise 2
let a = Ndarray.ones [|3; 3|] |> unwrap
let b = Ndarray.ofArray2D (array2D [[1.0; 2.0; 3.0]; 
                                     [4.0; 5.0; 6.0]; 
                                     [7.0; 8.0; 9.0]]) |> unwrap
let elemProd = Ndarray.mul a b |> unwrap

// Exercise 3
let mat4x4 = Ndarray.ofArray2D (array2D [[1.0; 2.0; 3.0; 4.0];
                                          [5.0; 6.0; 7.0; 8.0];
                                          [9.0; 10.0; 11.0; 12.0];
                                          [13.0; 14.0; 15.0; 16.0]]) |> unwrap
let diag = [|for i in 0..3 do yield Ndarray.get mat4x4 [|i; i|] |> unwrap|]

// Exercise 4
let data = Ndarray.ofArray [|10.0; 20.0; 30.0; 40.0; 50.0|] [|5|] |> unwrap
let minVal = Ndarray.min data
let maxVal = Ndarray.max data
let range = maxVal - minVal
let normalized = Ndarray.map (fun x -> (x - minVal) / range) data |> unwrap

// Exercise 5
let mat5x3 = Ndarray.ofArray2D (array2D [[1.0; 2.0; 3.0];
                                          [4.0; 5.0; 6.0];
                                          [7.0; 8.0; 9.0];
                                          [10.0; 11.0; 12.0];
                                          [13.0; 14.0; 15.0]]) |> unwrap
let colMeans = Ndarray.meanAxis mat5x3 0 |> unwrap
```

## Next Steps

- [Tutorial 2: Linear Algebra](Tutorial2_LinearAlgebra.md)
- [Tutorial 3: Statistics](Tutorial3_Statistics.md)
- [User Guide](../USER_GUIDE.md)

---

*Estimated time: 30 minutes*