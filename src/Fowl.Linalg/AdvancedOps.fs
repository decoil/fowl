namespace Fowl.Linalg

open System
open Fowl
open Fowl.Core.Types

/// <summary>Advanced linear algebra operations.
/// Least squares, pseudoinverse, matrix properties.
/// </summary>module AdvancedOps =
    
    /// <summary>Least squares solution: minimize ||Ax - b||₂
    /// Uses SVD for numerical stability.
    /// </summary>/// <param name="A">Coefficient matrix (m x n).</param>
    /// <param name="b">Right-hand side vector (m).</param>
    /// <returns>Solution x (n), residual, rank, singular values.</returns>    let lstsq (A: Ndarray<'K, float>) (b: Ndarray<'K, float>) 
              : FowlResult<Ndarray<'K, float> * float * int * float[]> =
        result {
            // Use SVD: A = U * S * Vt
            // Solution: x = V * (1/S) * Ut * b
            
            let! u, s, vt = Factorizations.svd A
            let! uArr = Ndarray.toArray2D u
            let! vtArr = Ndarray.toArray2D vt
            let! bArr = Ndarray.toArray b
            
            let m = uArr.GetLength(0)
            let n = vtArr.GetLength(0)
            let r = min m n
            
            // Threshold for rank determination
            let eps = 1e-10
            let maxS = Array.max s
            let threshold = eps * maxS
            
            // Determine effective rank
            let rank = s |> Array.filter (fun si -> si > threshold) |> Array.length
            
            // Compute Ut * b
            let utb = Array.init r (fun i ->
                let mutable sum = 0.0
                for j = 0 to m - 1 do
                    sum <- sum + uArr.[j, i] * bArr.[j]
                sum)
            
            // Divide by singular values (with regularization for small values)
            let utbScaled = Array.init r (fun i ->
                if s.[i] > threshold then
                    utb.[i] / s.[i]
                else
                    0.0)
            
            // Multiply by V: x = V * utbScaled
            let xArr = Array.init n (fun i ->
                let mutable sum = 0.0
                for j = 0 to r - 1 do
                    sum <- sum + vtArr.[j, i] * utbScaled.[j]
                sum)
            
            let! x = Ndarray.ofArray xArr [||]
            
            // Compute residual ||Ax - b||²
            let! aArr = Ndarray.toArray2D A
            let Ax = Array.init m (fun i ->
                let mutable sum = 0.0
                for j = 0 to n - 1 do
                    sum <- sum + aArr.[i, j] * xArr.[j]
                sum)
            
            let residual = 
                Array.map2 (-) Ax bArr
                |> Array.sumBy (fun diff -> diff * diff)
            
            return (x, residual, rank, s)
        }
    
    /// <summary>Moore-Penrose pseudoinverse using SVD.
    /// A⁺ = V * Σ⁺ * Ut where Σ⁺ is pseudoinverse of singular values.
    /// </summary>/// <param name="A">Input matrix.</param>
    /// <returns>Pseudoinverse A⁺.</returns>    let pinv (A: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
        result {
            let! u, s, vt = Factorizations.svd A
            let! uArr = Ndarray.toArray2D u
            let! vtArr = Ndarray.toArray2D vt
            
            let m = uArr.GetLength(0)
            let n = vtArr.GetLength(0)
            
            // Threshold for numerical stability
            let eps = 1e-15
            let maxS = if s.Length > 0 then Array.max s else 0.0
            let threshold = eps * maxS * float (max m n)
            
            // Create pseudoinverse of S
            let sPinv = Array.init s.Length (fun i ->
                if s.[i] > threshold then 1.0 / s.[i] else 0.0)
            
            // A⁺ = V * S⁺ * Ut
            // V is transpose of Vt
            let v = Array2D.init n n (fun i j -> vtArr.[j, i])
            
            // V * S⁺ (scale columns)
            let vScaled = Array2D.init n n (fun i j -
                if j < sPinv.Length then
                    v.[i, j] * sPinv.[j]
                else
                    0.0)
            
            // (V * S⁺) * Ut
            let pinvArr = Array2D.init n m (fun i j ->
                let mutable sum = 0.0
                for k = 0 to n - 1 do
                    if k < sPinv.Length && sPinv.[k] > 0.0 then
                        sum <- sum + vScaled.[i, k] * uArr.[j, k]
                sum)
            
            return! Ndarray.ofArray2D pinvArr
        }
    
    /// <summary>Matrix rank using SVD.
    /// Number of singular values above threshold.
    /// </summary>/// <param name="A">Input matrix.</param>
    /// <returns>Rank of matrix.</returns>    let rank (A: Ndarray<'K, float>) : FowlResult<int> =
        result {
            let! _, s, _ = Factorizations.svd A
            
            let shape = Ndarray.shape A
            let maxDim = Array.max shape
            let eps = 1e-10
            let maxS = Array.max s
            let threshold = float maxDim * eps * maxS
            
            return s |> Array.filter (fun si -> si > threshold) |> Array.length
        }
    
    /// <summary>Condition number using SVD.
    /// κ(A) = σ_max / σ_min
    /// Large condition number indicates ill-conditioned matrix.
    /// </summary>/// <param name="A">Input matrix.</param>
    /// <returns>Condition number.</returns>    let cond (A: Ndarray<'K, float>) : FowlResult<float> =
        result {
            let! _, s, _ = Factorizations.svd A
            
            if s.Length = 0 then
                return! Error.invalidState "Cannot compute condition number of empty matrix"
            
            let maxS = Array.max s
            let minS = max (Array.min s) 1e-15
            
            return maxS / minS
        }
    
    /// <summary>Null space using SVD.
    /// Returns basis for null space (vectors x such that Ax = 0)
    /// </summary>/// <param name="A">Input matrix.</param>
    /// <returns>Basis for null space.</returns>    let nullSpace (A: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
        result {
            let! _, s, vt = Factorizations.svd A
            let! vtArr = Ndarray.toArray2D vt
            
            let n = vtArr.GetLength(0)
            let maxS = Array.max s
            let threshold = 1e-10 * maxS
            
            // Find columns corresponding to near-zero singular values
            let nullIndices = 
                s
                |> Array.mapi (fun i si -> if si <= threshold then Some i else None)
                |> Array.choose id
            
            if nullIndices.Length = 0 then
                // Full rank - null space is empty (return zero vector)
                return! Ndarray.zeros<'K> [||]
            else
                // Extract null space basis from V
                let basis = 
                    Array2D.init n nullIndices.Length (fun i j ->
                        // V is transpose of Vt
                        vtArr.[nullIndices.[j], i])
                
                return! Ndarray.ofArray2D basis
        }
    
    /// <summary>Orthonormal basis for range (column space) using SVD.
    /// </summary>/// <param name="A">Input matrix.</param>
    /// <returns>Orthonormal basis for range of A.</returns>    let orth (A: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
        result {
            let! u, s, _ = Factorizations.svd A
            let! uArr = Ndarray.toArray2D u
            
            let m = uArr.GetLength(0)
            let maxS = Array.max s
            let threshold = 1e-10 * maxS
            
            // Find columns corresponding to non-zero singular values
            let rangeIndices = 
                s
                |> Array.mapi (fun i si -> if si > threshold then Some i else None)
                |> Array.choose id
            
            // Extract range basis from U
            let basis = 
                Array2D.init m rangeIndices.Length (fun i j ->
                    uArr.[i, rangeIndices.[j]])
            
            return! Ndarray.ofArray2D basis
        }
    
    /// <summary>Matrix exponential using Pade approximation.
    /// Computes e^A for square matrix A.
    /// </summary>/// <param name="A">Square matrix.</param>
    /// <returns>e^A.</returns>    let expm (A: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
        result {
            let! aArr = Ndarray.toArray2D A
            let n = aArr.GetLength(0)
            
            if aArr.GetLength(1) <> n then
                return! Error.invalidShape "expm requires square matrix"
            
            // Simple implementation: use eigendecomposition if symmetric
            // For general matrices, use scaling and squaring with Pade approximation
            // This is a basic implementation - can be improved
            
            // Check if symmetric
            let isSymmetric = 
                [| for i = 0 to n-1 do
                     for j = i+1 to n-1 do
                         yield abs (aArr.[i,j] - aArr.[j,i]) < 1e-10 |]
                |> Array.forall id
            
            if isSymmetric then
                // Use eigendecomposition: A = Q * D * Qt, e^A = Q * e^D * Qt
                let! eigenvals, eigenvecs = Factorizations.eigSymmetric A
                let! evecs = Ndarray.toArray2D eigenvecs
                let! evals = Ndarray.toArray eigenvals
                
                // e^D
                let expD = Array2D.init n n (fun i j ->
                    if i = j then exp evals.[i] else 0.0)
                
                // Q * e^D
                let qExpD = Array2D.init n n (fun i j ->
                    let mutable sum = 0.0
                    for k = 0 to n-1 do
                        sum <- sum + evecs.[i,k] * expD.[k,j]
                    sum)
                
                // (Q * e^D) * Qt
                let result = Array2D.init n n (fun i j ->
                    let mutable sum = 0.0
                    for k = 0 to n-1 do
                        sum <- sum + qExpD.[i,k] * evecs.[j,k]
                    sum)
                
                return! Ndarray.ofArray2D result
            else
                // For non-symmetric, use Taylor series approximation (basic)
                let mutable result = Array2D.init n n (fun i j -> if i = j then 1.0 else 0.0)
                let mutable term = Array2D.init n n (fun i j -> if i = j then 1.0 else 0.0)
                
                for k = 1 to 20 do
                    // term = term * A / k
                    let newTerm = Array2D.zeroCreate n n
                    for i = 0 to n-1 do
                        for j = 0 to n-1 do
                            for l = 0 to n-1 do
                                newTerm.[i,j] <- newTerm.[i,j] + term.[i,l] * aArr.[l,j] / float k
                    term <- newTerm
                    
                    // result += term
                    for i = 0 to n-1 do
                        for j = 0 to n-1 do
                            result.[i,j] <- result.[i,j] + term.[i,j]
                
                return! Ndarray.ofArray2D result
        }
    
    /// <summary>Frobenius norm of a matrix.
    /// ||A||_F = sqrt(sum(A_ij^2))
    /// </summary>/// <param name="A">Input matrix.</param>
    /// <returns>Frobenius norm.</returns>    let normFrobenius (A: Ndarray<'K, float>) : FowlResult<float> =
        result {
            let! data = Ndarray.toArray A
            let sumSq = data |> Array.sumBy (fun x -> x * x)
            return sqrt sumSq
        }
