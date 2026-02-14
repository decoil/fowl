# Fowl Linear Algebra Module Design

## Overview

Interface to OpenBLAS for high-performance linear algebra operations.

## Dependencies

- OpenBLAS (system library)
- F# P/Invoke declarations

## Module Structure

```fsharp
namespace Fowl.Linalg

open Fowl
open Fowl.Core

/// LAPACK-style matrix types
module MatrixTypes =
    /// General matrix (no special structure)
    type GeneralMatrix = Ndarray<'K, float>
    
    /// Symmetric matrix (stored in upper/lower triangle)
    type SymmetricMatrix = Ndarray<'K, float>
    
    /// Triangular matrix (upper or lower)
    type TriangularMatrix = Ndarray<'K, float>
    
    /// Diagonal matrix (stored as 1D array)
    type DiagonalMatrix = Ndarray<'K, float>

/// Linear algebra operations
module LinearAlgebra =
    
    // ========================================
    // Matrix Factorizations
    // ========================================
    
    /// LU decomposition with partial pivoting
    /// Returns (P, L, U) where P*A = L*U
    val lu : Ndarray<'K, float> -> int array * Ndarray<'K, float> * Ndarray<'K, float>
    
    /// QR decomposition
    /// Returns (Q, R) where A = Q*R
    val qr : Ndarray<'K, float> -> Ndarray<'K, float> * Ndarray<'K, float>
    
    /// Singular Value Decomposition
    /// Returns (U, S, Vt) where A = U*diag(S)*Vt
    val svd : Ndarray<'K, float> -> Ndarray<'K, float> * Ndarray<'K, float> * Ndarray<'K, float>
    
    /// Cholesky decomposition for positive-definite matrices
    /// Returns L where A = L*L^T
    val cholesky : Ndarray<'K, float> -> Ndarray<'K, float>
    
    /// Eigenvalue decomposition
    /// Returns (eigenvalues, eigenvectors)
    val eig : Ndarray<'K, float> -> Ndarray<'K, complex> * Ndarray<'K, complex>
    
    /// Eigenvalues only (faster if eigenvectors not needed)
    val eigvals : Ndarray<'K, float> -> Ndarray<'K, complex>
    
    // ========================================
    // Linear System Solvers
    // ========================================
    
    /// Solve A*X = B for X
    val solve : Ndarray<'K, float> -> Ndarray<'K, float> -> Ndarray<'K, float>
    
    /// Solve using pre-computed LU factorization
    val solveLu : int array -> Ndarray<'K, float> -> Ndarray<'K, float> -> Ndarray<'K, float>
    
    /// Least squares solution: minimize ||A*X - B||
    val lstsq : Ndarray<'K, float> -> Ndarray<'K, float> -> Ndarray<'K, float>
    
    /// Solve triangular system
    val solveTriangular : bool -> Ndarray<'K, float> -> Ndarray<'K, float> -> Ndarray<'K, float>
    
    // ========================================
    // Matrix Operations
    // ========================================
    
    /// Matrix inverse
    val inv : Ndarray<'K, float> -> Ndarray<'K, float>
    
    /// Matrix determinant
    val det : Ndarray<'K, float> -> float
    
    /// Matrix rank
    val matrixRank : Ndarray<'K, float> -> int
    
    /// Condition number
    val cond : Ndarray<'K, float> -> float
    
    /// Matrix norm (1-norm, 2-norm, inf-norm, frobenius)
    val norm : string -> Ndarray<'K, float> -> float
    
    // ========================================
    // Special Matrices
    // ========================================
    
    /// Identity matrix
    val eye : int -> Ndarray<'K, float>
    
    /// Diagonal matrix from vector
    val diag : Ndarray<'K, float> -> Ndarray<'K, float>
    
    /// Extract diagonal from matrix
    val getDiag : Ndarray<'K, float> -> Ndarray<'K, float>
    
    /// Upper triangular part
    val triu : int -> Ndarray<'K, float> -> Ndarray<'K, float>
    
    /// Lower triangular part
    val tril : int -> Ndarray<'K, float> -> Ndarray<'K, float>

/// BLAS-level operations (lower level)
module Blas =
    /// Matrix-matrix multiply: C = alpha*A*B + beta*C
    val gemm : float -> Ndarray<'K, float> -> Ndarray<'K, float> -> float -> Ndarray<'K, float> -> Ndarray<'K, float>
    
    /// Matrix-vector multiply: y = alpha*A*x + beta*y
    val gemv : float -> Ndarray<'K, float> -> Ndarray<'K, float> -> float -> Ndarray<'K, float> -> Ndarray<'K, float>
    
    /// Vector dot product
    val dot : Ndarray<'K, float> -> Ndarray<'K, float> -> float
    
    /// Vector 2-norm
    val nrm2 : Ndarray<'K, float> -> float
    
    /// Vector 1-norm
    val asum : Ndarray<'K, float> -> float

/// LAPACK driver routines
module Lapack =
    /// General matrix solve (dgesv)
    val gesv : Ndarray<'K, float> -> Ndarray<'K, float> -> Ndarray<'K, float>
    
    /// Least squares (dgels)
    val gels : Ndarray<'K, float> -> Ndarray<'K, float> -> Ndarray<'K, float>
    
    /// Eigenvalue decomposition (dgeev)
    val geev : Ndarray<'K, float> -> Ndarray<'K, float> * Ndarray<'K, float>
    
    /// SVD (dgesvd)
    val gesvd : Ndarray<'K, float> -> Ndarray<'K, float> * Ndarray<'K, float> * Ndarray<'K, float>
    
    /// LU factorization (dgetrf)
    val getrf : Ndarray<'K, float> -> int array * Ndarray<'K, float>
    
    /// QR factorization (dgeqrf)
    val geqrf : Ndarray<'K, float> -> Ndarray<'K, float> * Ndarray<'K, float>
```

## Implementation Notes

### OpenBLAS Integration

OpenBLAS provides optimized BLAS and LAPACK routines:

```fsharp
module Native =
    [<DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl)>]
    extern void cblas_dgemm(
        int Order, int TransA, int TransB,
        int M, int N, int K,
        double alpha, double[] A, int lda,
        double[] B, int ldb,
        double beta, double[] C, int ldc)
    
    [<DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl)>]
    extern int LAPACKE_dgesv(
        int matrix_order, int n, int nrhs,
        double[] a, int lda, int[] ipiv,
        double[] b, int ldb)
```

### Matrix Layout

- OpenBLAS uses column-major (Fortran) order by default
- Our Ndarray uses row-major (C) order
- Need to handle transposition or use appropriate BLAS flags

### Performance Considerations

1. **Memory alignment**: Ensure arrays are aligned for SIMD
2. **Blocking**: Use cache-friendly block sizes for large matrices
3. **Parallelism**: OpenBLAS uses threads internally
4. **Copy avoidance**: Use appropriate strides when possible

## Usage Examples

```fsharp
open Fowl
open Fowl.Linalg

// Solve linear system
let A = Ndarray.random [|100; 100|]
let b = Ndarray.random [|100|]
let x = LinearAlgebra.solve A b

// LU decomposition
let p, l, u = LinearAlgebra.lu A

// SVD
let u, s, vt = LinearAlgebra.svd A

// Matrix inverse
let A_inv = LinearAlgebra.inv A

// Use BLAS directly
let C = Blas.gemm 1.0 A B 0.0 (Ndarray.zeros [|100; 100|])
```
