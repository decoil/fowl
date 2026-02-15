# Chapter 3: Linear Algebra Fundamentals

## 3.1 Introduction

Linear algebra is the foundation of numerical computing. Fowl provides comprehensive linear algebra operations including matrix creation, decomposition, and solving linear systems.

## 3.2 Matrix Creation

### Special Matrices

```fsharp
open Fowl
open Fowl.Core.Types
open Fowl.Linq

// Identity matrix
let I = Core.eye<Float64> 5
// 5x5 identity matrix

// Diagonal matrix
let diag = Core.diag [|1.0; 2.0; 3.0|]
// [|1 0 0|
//  |0 2 0|
//  |0 0 3|]

// Triangular matrices
let lower = Factorizations.tril matrix |> unwrap
let upper = Factorizations.triu matrix |> unwrap

// Random matrices
open Fowl.Stats.Random
let rng = RandomState.create 42
let uniform = RandomState.rand rng [|10; 10|]
```

### Matrix Properties

```fsharp
let A = Ndarray.ofArray [|1.0 .. 9.0|] [|3; 3|] |> unwrap

// Trace (sum of diagonal)
let tr = Core.trace A

// Determinant
let det = Factorizations.det A |> unwrap

// Rank
let rank = AdvancedOps.rank A |> unwrap

// Condition number
let cond = AdvancedOps.cond A |> unwrap
```

## 3.3 Matrix Operations

### Basic Operations

```fsharp
let A = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0|] [|2; 2|] |> unwrap
let B = Ndarray.ofArray [|5.0; 6.0; 7.0; 8.0|] [|2; 2|] |> unwrap

// Addition and subtraction
let C = Ndarray.add A B |> unwrap
let D = Ndarray.sub A B |> unwrap

// Element-wise multiplication (Hadamard product)
let E = Ndarray.mul A B |> unwrap

// Matrix multiplication
let F = Matrix.matmul A B |> unwrap

// Matrix power
let A3 = Factorizations.matrixPower A 3 |> unwrap
```

### Matrix-Vector Operations

```fsharp
let A = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0|] [|2; 2|] |> unwrap
let x = Ndarray.ofArray [|5.0; 6.0|] [||] |> unwrap

// Matrix-vector multiplication
let y = Matrix.matmul A (Ndarray.reshape [|2; 1|] x |> unwrap) |> unwrap

// Dot product
let dot = Matrix.dot x x

// Outer product
let outer = Matrix.outer x x |> unwrap
```

## 3.4 Matrix Decompositions

### LU Decomposition

```fsharp
open Fowl.Linq.Factorizations

let A = Ndarray.ofArray [|4.0; 3.0; 6.0; 3.0|] [|2; 2|] |> unwrap

// A = P * L * U
let! (ipiv, L, U) = lu A

// ipiv: pivot indices
// L: lower triangular with 1s on diagonal
// U: upper triangular

// Verify: P*L*U should equal A
```

### QR Decomposition

```fsharp
let A = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0|] [|3; 2|] |> unwrap

// A = Q * R
let! (Q, R) = qr A

// Q: orthogonal columns (Q^T * Q = I)
// R: upper triangular

// Verify orthogonality
let! qtq = Matrix.matmul (Matrix.transpose Q |> unwrap) Q
```

### SVD (Singular Value Decomposition)

```fsharp
let A = Ndarray.ofArray [|3.0; 2.0; 1.0; 4.0; 5.0; 6.0|] [|3; 2|] |> unwrap

// A = U * S * Vt
let! (U, S, Vt) = svd A

// U: left singular vectors (m x m)
// S: singular values (min(m,n))
// Vt: right singular vectors transposed (n x n)

// Low-rank approximation
let rank2Approx =
    result {
        let! u, s, vt = svd A
        // Keep top 2 singular values
        let s2 = Array.init 2 (fun i -> s.[i])
        // Reconstruct
        return! reconstructSVD u s2 vt
    }
```

### Cholesky Decomposition

```fsharp
// For symmetric positive definite matrices
let A = Ndarray.ofArray [|4.0; 2.0; 2.0; 3.0|] [|2; 2|] |> unwrap

// A = L * L^T
let! L = cholesky A

// L: lower triangular

// Useful for:
// 1. Solving linear systems
// 2. Generating correlated random variables
// 3. Checking positive definiteness
```

### Eigenvalue Decomposition

```fsharp
// For symmetric matrices
let A = Ndarray.ofArray [|4.0; 2.0; 2.0; 3.0|] [|2; 2|] |> unwrap

// A = Q * Λ * Q^T
let! (eigenvals, eigenvecs) = eigSymmetric A

// eigenvals: eigenvalues
// eigenvecs: eigenvectors as columns

// Verify: A * v = λ * v
let! Av = Matrix.matmul A eigenvecs
let! lambdaV = Ndarray.mul eigenvecs (Ndarray.reshape [|-1; 1|] eigenvals |> unwrap) |> unwrap
```

## 3.5 Solving Linear Systems

### Square Systems

```fsharp
// Solve A * x = b
let A = Ndarray.ofArray [|3.0; 1.0; 1.0; 2.0|] [|2; 2|] |> unwrap
let b = Ndarray.ofArray [|9.0; 8.0|] [||] |> unwrap

let! x = solve A b
// x = [2; 3] because 3*2 + 1*3 = 9, 1*2 + 2*3 = 8

// Verify
let! Ax = Matrix.matmul A (Ndarray.reshape [|-1; 1|] x |> unwrap)
```

### Overdetermined Systems (Least Squares)

```fsharp
open Fowl.Linq.AdvancedOps

// More equations than unknowns
let A = Ndarray.ofArray [|
    1.0; 1.0;
    1.0; 2.0;
    1.0; 3.0;
    1.0; 4.0
|] [|4; 2|] |> unwrap

let b = Ndarray.ofArray [|6.0; 5.0; 7.0; 10.0|] [||] |> unwrap

// Minimize ||A*x - b||^2
let! (x, residual, rank, s) = lstsq A b

// x gives best-fit line: y = x₀ + x₁*t
```

### Underdetermined Systems

```fsharp
// Fewer equations than unknowns (infinite solutions)
// Use pseudoinverse for minimum-norm solution
let! x = AdvancedOps.pinv A |> Result.bind (fun pinv -
    Matrix.matmul pinv b)
```

## 3.6 Matrix Inversion

```fsharp
let A = Ndarray.ofArray [|4.0; 7.0; 2.0; 6.0|] [|2; 2|] |> unwrap

// Regular inverse (for square, non-singular matrices)
let! Ainv = Factorizations.inv A

// Verify: A * Ainv = I
let! product = Matrix.matmul A Ainv

// Moore-Penrose pseudoinverse (works for any matrix)
let! Apinv = AdvancedOps.pinv A
```

## 3.7 Advanced Operations

### Matrix Exponential

```fsharp
// For solving systems of ODEs: dx/dt = A*x
let A = Ndarray.ofArray [|0.0; 1.0; -1.0; 0.0|] [|2; 2|] |> unwrap

// e^A
let! expA = AdvancedOps.expm A

// Useful for:
// 1. Solving linear ODEs
// 2. Continuous-time Markov chains
// 3. Lie group methods
```

### Kronecker Product

```fsharp
let A = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0|] [|2; 2|] |> unwrap
let B = Ndarray.ofArray [|0.0; 5.0; 6.0; 7.0|] [|2; 2|] |> unwrap

// A ⊗ B
let! K = AdvancedOps.kron A B
// 4x4 block matrix
```

### Vectorization

```fsharp
let A = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0|] [|2; 2|] |> unwrap

// vec(A): stack columns
let vecA = Ndarray.toArray A
// [|1; 2; 3; 4|] (column-major)

// Reshape back
let A2 = Ndarray.ofArray vecA [|2; 2|] |> unwrap
```

## 3.8 Exercises

### Exercise 3.1: Linear Regression via Normal Equations

```fsharp
let linearRegressionNormal (X: float[,]) (y: float[]) =
    result {
        let n = X.GetLength(0)
        let p = X.GetLength(1)
        
        // Add bias column
        let Xb = Array2D.init n (p + 1) (fun i j -
            if j = 0 then 1.0 else X.[i, j-1])
        
        let! xArr = Ndarray.ofArray2D Xb
        let! yArr = Ndarray.ofArray y [||]
        
        // β = (X^T X)^-1 X^T y
        let! xt = Matrix.transpose xArr
        let! xtx = Matrix.matmul xt xArr
        let! xtxInv = Factorizations.inv xtx
        let! xty = Matrix.matmul xt yArr
        let! beta = Matrix.matmul xtxInv xty
        
        return Ndarray.toArray beta
    }

// Usage
let X = array2D [|
    [|1.0; 2.0|]
    [|2.0; 3.0|]
    [|3.0; 4.0|]
|]
let y = [|5.0; 7.0; 9.0|]

match linearRegressionNormal X y with
| Ok beta -> printfn "Coefficients: %A" beta
| Error e -> printfn "Error: %A" e
```

### Exercise 3.2: Principal Component Analysis (PCA)

```fsharp
let pca (data: Ndarray<'K, float>) (nComponents: int) =
    result {
        // Center the data
        let dataArr = Ndarray.toArray data
        let mean = Array.average dataArr
        let centered = dataArr |> Array.map (fun x -> x - mean)
        let! centeredNd = Ndarray.ofArray centered (Ndarray.shape data)
        
        // Compute covariance matrix
        let! cov = Correlation.covarianceMatrix centeredNd
        
        // Eigendecomposition
        let! eigenvals, eigenvecs = eigSymmetric cov
        
        // Sort by eigenvalue (descending)
        let indexed = eigenvals |> Array.mapi (fun i v -> (v, i))
        let sorted = indexed |> Array.sortByDescending fst
        
        // Select top components
        let topIndices = sorted.[..nComponents-1] |> Array.map snd
        
        // Extract principal components
        let! components = 
            topIndices
            |> Array.map (fun idx -
                Ndarray.get eigenvecs [|idx|])
            |> Result.sequence
        
        return (eigenvals, components)
    }
```

### Exercise 3.3: Solving Markov Chains

```fsharp
// Find stationary distribution of Markov chain
let stationaryDistribution (P: float[,]) =
    result {
        // P is transition matrix (rows sum to 1)
        let n = P.GetLength(0)
        
        // Solve πP = π, or (P^T - I)π = 0
        let Pt = Array2D.init n n (fun i j -> P.[j, i])
        
        // Subtract identity
        for i = 0 to n-1 do
            Pt.[i,i] <- Pt.[i,i] - 1.0
        
        // Add normalization constraint: sum(π) = 1
        let A = Array2D.init (n+1) n (fun i j -
            if i < n then Pt.[i,j] else 1.0)
        let b = Array.init (n+1) (fun i -
            if i < n then 0.0 else 1.0)
        
        let! aArr = Ndarray.ofArray2D A
        let! bArr = Ndarray.ofArray b [||]
        let! pi = lstsq aArr bArr
        
        return fst pi
    }
```

## 3.9 Performance Tips

1. **Use appropriate decompositions**:
   - LU for general square matrices
   - Cholesky for symmetric positive definite (2x faster)
   - QR for least squares (more stable than normal equations)

2. **Avoid explicit inversion**:
```fsharp
// Slow
let! Ainv = Factorizations.inv A
let! x = Matrix.matmul Ainv b

// Fast
let! x = solve A b
```

3. **Reuse factorizations**:
```fsharp
// Factor once, solve many times
let! luFact = luFactor A
let! x1 = luSolve luFact b1
let! x2 = luSolve luFact b2
```

## 3.10 Summary

Key concepts:
- Matrix decompositions (LU, QR, SVD, Cholesky, Eig) reveal structure
- Linear systems can be solved efficiently using factorizations
- Least squares handles overdetermined systems
- Pseudoinverse provides minimum-norm solutions
- Matrix exponential solves systems of ODEs

---

*Next: [Chapter 4: Signal Processing](chapter04.md)*
