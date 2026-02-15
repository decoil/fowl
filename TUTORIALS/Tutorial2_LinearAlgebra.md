# Tutorial 2: Linear Algebra

Master matrix operations, factorizations, and solvers with Fowl.

## Overview

Linear algebra is fundamental to numerical computing. Fowl provides comprehensive support for matrix operations, factorizations, and linear system solvers.

## Learning Objectives

- Create and manipulate matrices
- Perform matrix operations (multiplication, transpose, inverse)
- Use matrix factorizations (LU, QR, SVD)
- Solve linear systems
- Compute matrix properties

## Setup

```fsharp
open Fowl
open Fowl.Core.Types
open Fowl.Linalg

let unwrap = function Ok x -> x | Error e -> failwith e.Message
```

## Creating Matrices

```fsharp
// From 2D array
let A = Ndarray.ofArray2D (array2D [[1.0; 2.0]; 
                                     [3.0; 4.0]]) |> unwrap

// Special matrices
let identity = Ndarray.eye 3 |> unwrap
let zeros = Ndarray.zeros [|3; 3|] |> unwrap
let ones = Ndarray.ones [|2; 4|] |> unwrap

// Diagonal matrix
let diag = Core.diag [|1.0; 2.0; 3.0|] |> unwrap

// Random matrix
let random = Ndarray.random [|3; 3|] (Some 42) |> unwrap
```

## Basic Matrix Operations

### Arithmetic

```fsharp
let A = Ndarray.ofArray2D (array2D [[1.0; 2.0]; [3.0; 4.0]]) |> unwrap
let B = Ndarray.ofArray2D (array2D [[5.0; 6.0]; [7.0; 8.0]]) |> unwrap

// Matrix multiplication (not element-wise!)
let C = Matrix.matmul A B |> unwrap
// C = [[19, 22], [43, 50]]

// Element-wise multiplication
let elemProd = Ndarray.mul A B |> unwrap
// elemProd = [[5, 12], [21, 32]]

// Transpose
let AT = Matrix.transpose A |> unwrap
// AT = [[1, 3], [2, 4]]

// Matrix addition/subtraction
let sum = Ndarray.add A B |> unwrap
let diff = Ndarray.sub A B |> unwrap
```

### Matrix-Vector Operations

```fsharp
let A = Ndarray.ofArray2D (array2D [[1.0; 2.0]; [3.0; 4.0]; [5.0; 6.0]]) |> unwrap
let v = Ndarray.ofArray [|7.0; 8.0|] [|2|] |> unwrap

// Matrix-vector multiplication
let Av = Matrix.matmul A v |> unwrap
// Av = [23, 53, 83] (3x2 * 2x1 = 3x1)

// Dot product
let u = Ndarray.ofArray [|1.0; 2.0; 3.0|] [|3|] |> unwrap
let dot = Matrix.dot u u |> unwrap  // 14.0

// Outer product
let outer = Matrix.outer u u |> unwrap  // 3x3 matrix
```

## Matrix Factorizations

### LU Decomposition

LU decomposition factors a matrix into lower and upper triangular matrices:

```fsharp
let A = Ndarray.ofArray2D (array2D [[4.0; 3.0]; 
                                     [6.0; 3.0]]) |> unwrap

let L, U, P = Factorizations.lu A |> unwrap

// L = [[1, 0], [1.5, 1]] (lower triangular)
// U = [[4, 3], [0, -1.5]] (upper triangular)
// P = permutation matrix

// Verify: P*A = L*U
```

### QR Decomposition

QR decomposition is useful for least squares and eigenvalue problems:

```fsharp
let A = Ndarray.ofArray2D (array2D [[1.0; 2.0; 3.0]; 
                                     [4.0; 5.0; 6.0]; 
                                     [7.0; 8.0; 9.0]]) |> unwrap

let Q, R = Factorizations.qr A |> unwrap

// Q is orthogonal: Q^T * Q = I
// R is upper triangular

// Verify: A = Q * R
```

### SVD (Singular Value Decomposition)

SVD is one of the most important matrix factorizations:

```fsharp
let A = Ndarray.ofArray2D (array2D [[3.0; 0.0]; 
                                     [0.0; 2.0]; 
                                     [0.0; 0.0]]) |> unwrap

let U, S, Vt = Factorizations.svd A |> unwrap

// U: 3x3 orthogonal matrix (left singular vectors)
// S: singular values [3, 2]
// Vt: 2x2 orthogonal matrix transpose (right singular vectors)

// Reconstruct: A = U * diag(S) * Vt
```

### Cholesky Decomposition

For positive definite matrices only:

```fsharp
// Must be symmetric positive definite
let SPD = Ndarray.ofArray2D (array2D [[4.0; 2.0]; 
                                       [2.0; 5.0]]) |> unwrap

let L = Factorizations.cholesky SPD |> unwrap

// L is lower triangular
// Verify: SPD = L * L^T
```

### Eigenvalue Decomposition

For symmetric matrices:

```fsharp
let A = Ndarray.ofArray2D (array2D [[4.0; 2.0]; 
                                     [2.0; 3.0]]) |> unwrap

let eigenvalues, eigenvectors = Factorizations.eigSymmetric A |> unwrap

// eigenvalues: [5.56, 1.44] (sorted descending)
// eigenvectors: columns are eigenvectors
// Verify: A * v = λ * v for each eigenpair
```

## Solving Linear Systems

### Square Systems

```fsharp
// Solve Ax = b
let A = Ndarray.ofArray2D (array2D [[3.0; 1.0]; 
                                     [1.0; 2.0]]) |> unwrap
let b = Ndarray.ofArray [|9.0; 8.0|] [|2|] |> unwrap

let x = Factorizations.solve A b |> unwrap
// x = [2, 3] (exact solution)

// Verify
let Ax = Matrix.matmul A x |> unwrap
// Ax should equal b
```

### Overdetermined Systems (Least Squares)

When you have more equations than unknowns:

```fsharp
// Fit a line: y = mx + b
// Data points: (1,2), (2,3), (3,5), (4,4)
let X = Ndarray.ofArray2D (array2D [[1.0; 1.0]; 
                                     [1.0; 2.0]; 
                                     [1.0; 3.0]; 
                                     [1.0; 4.0]]) |> unwrap  // Design matrix
let y = Ndarray.ofArray [|2.0; 3.0; 5.0; 4.0|] [|4|] |> unwrap

// Solve: (X^T * X) * beta = X^T * y
let beta, residual, rank, s = AdvancedOps.lstsq X y |> unwrap

// beta = [intercept, slope]
```

### Using Pseudoinverse

```fsharp
// When matrix is singular or rectangular
let A = Ndarray.ofArray2D (array2D [[1.0; 2.0]; 
                                     [2.0; 4.0]]) |> unwrap  // Singular!

let A_pinv = AdvancedOps.pinv A |> unwrap

// Solve using pseudoinverse
let b = Ndarray.ofArray [|3.0; 6.0|] [|2|] |> unwrap
let x = Matrix.matmul A_pinv b |> unwrap  // Minimum norm solution
```

## Matrix Properties

```fsharp
let A = Ndarray.ofArray2D (array2D [[1.0; 2.0]; 
                                     [3.0; 4.0]]) |> unwrap

// Determinant
let det = Factorizations.det A |> unwrap  // -2.0

// Trace
let tr = Core.trace A  // 5.0

// Rank
let r = AdvancedOps.rank A |> unwrap  // 2

// Condition number (high = ill-conditioned)
let cond = AdvancedOps.cond A |> unwrap

// Inverse
let invA = Factorizations.inv A |> unwrap
// Verify: A * invA = I
```

## Practical Example: Principal Component Analysis

```fsharp
// Sample data: 5 samples, 3 features
let data = Ndarray.ofArray2D (array2D [[1.0; 2.0; 3.0];
                                        [2.0; 3.0; 4.0];
                                        [3.0; 4.0; 5.0];
                                        [4.0; 5.0; 6.0];
                                        [5.0; 6.0; 7.0]]) |> unwrap

// 1. Center the data
let mean = Ndarray.meanAxis data 0 |> unwrap
let centered = Ndarray.sub data mean |> unwrap  // Simplified

// 2. Compute covariance matrix
let cov = Correlation.covarianceMatrix (Ndarray.toArray2D centered |> unwrap) |> Result.get

// 3. Eigendecomposition
let eigenvals, eigenvecs = Factorizations.eigSymmetric cov |> unwrap

// 4. Sort by eigenvalue magnitude
// 5. Project data onto principal components

printfn "Eigenvalues (variance explained): %A" (Ndarray.toArray eigenvals)
```

## Exercises

1. Create a 3x3 rotation matrix and verify it's orthogonal
2. Solve a system of 3 equations with 3 unknowns
3. Compute the SVD of a 4x3 matrix and reconstruct it
4. Find the least squares fit for a quadratic function
5. Compute the condition number of a Hilbert matrix

## Solutions

```fsharp
// Exercise 1: Rotation matrix
let theta = System.Math.PI / 4.0  // 45 degrees
let R = Ndarray.ofArray2D (array2D [[cos theta; -sin theta; 0.0];
                                     [sin theta; cos theta; 0.0];
                                     [0.0; 0.0; 1.0]]) |> unwrap
let RT = Matrix.transpose R |> unwrap
let shouldBeIdentity = Matrix.matmul R RT |> unwrap

// Exercise 2: Solve 3x3 system
let A = Ndarray.ofArray2D (array2D [[2.0; 1.0; -1.0];
                                     [-3.0; -1.0; 2.0];
                                     [-2.0; 1.0; 2.0]]) |> unwrap
let b = Ndarray.ofArray [|8.0; -11.0; -3.0|] [|3|] |> unwrap
let x = Factorizations.solve A b |> unwrap

// Exercise 3: SVD reconstruction
let A = Ndarray.random [|4; 3|] (Some 42) |> unwrap
let U, S, Vt = Factorizations.svd A |> unwrap
let Sdiag = Ndarray.zeros [|4; 3|] |> unwrap
// Fill diagonal with S values...
let reconstructed = Matrix.matmul (Matrix.matmul U Sdiag |> unwrap) Vt |> unwrap

// Exercise 4: Quadratic fit
let t = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0; 5.0|] [|5|] |> unwrap
let y = Ndarray.ofArray [|2.3; 4.5; 7.2; 9.1; 12.5|] [|5|] |> unwrap
let X = Ndarray.ofArray2D (array2D [[1.0; 1.0; 1.0];
                                     [1.0; 2.0; 4.0];
                                     [1.0; 3.0; 9.0];
                                     [1.0; 4.0; 16.0];
                                     [1.0; 5.0; 25.0]]) |> unwrap
let coeffs, _, _, _ = AdvancedOps.lstsq X y |> unwrap

// Exercise 5: Hilbert matrix condition number
let hilbert n =
    Array2D.init n n (fun i j -> 1.0 / float (i + j + 1))
    |> Ndarray.ofArray2D
    |> unwrap
let H5 = hilbert 5
let condH5 = AdvancedOps.cond H5 |> unwrap  // Very large!
```

## Next Steps

- [Tutorial 3: Statistics](Tutorial3_Statistics.md)
- [User Guide](../USER_GUIDE.md#linear-algebra)

---

*Estimated time: 45 minutes*