namespace Fowl.Linalg

open System
open Fowl
open Fowl.Core.Types

/// <summary>Advanced linear algebra operations.
/// Least squares, pseudoinverse, matrix properties.
/// </summary>module AdvancedOps =
    
    /// <summary>Least squares solution: minimize ||Ax - b||₂
/// Uses SVD for numerical stability.
/// Returns (x, residuals, rank, singular_values)
/// </summary>let lstsq (A: Ndarray<float>) (b: Ndarray<float>) 
              : FowlResult<Ndarray<float> * float * int * float[]> =
        result {
            // Use SVD: A = U * S * Vt
            // Solution: x = V * (1/S) * Ut * b
            
            let! u, s, vt = Factorizations.svd A
            let! uArr = Ndarray.toArray2D u
            let! vtArr = Ndarray.toArray2D vt
            let! bArr = Ndarray.toArray b
            
            let m = uArr.GetLength(0)
            let n = vtArr.GetLength(0)
            let r = min m n  // Rank (number of singular values)
            
            // Threshold for rank determination
            let eps = 1e-10
            let maxS = Array.max s
            let threshold = eps * maxS
            
            // Determine effective rank
            let rank = s |> Array.filter (fun si -> si > threshold) |> Array.length
            
            // Compute Ut * b
            let utb = Array.init r (fun i -
                let mutable sum = 0.0
                for j = 0 to m - 1 do
                    sum <- sum + uArr.[j, i] * bArr.[j]
                sum)
            
            // Divide by singular values (with regularization for small values)
            let utbScaled = Array.init r (fun i -
                if s.[i] > threshold then
                    utb.[i] / s.[i]
                else
                    0.0)
            
            // Multiply by V: x = V * utbScaled
            let xArr = Array.init n (fun i -
                let mutable sum = 0.0
                for j = 0 to r - 1 do
                    sum <- sum + vtArr.[j, i] * utbScaled.[j]
                sum)
            
            let! x = Ndarray.ofArray xArr [|n|]
            
            // Compute residual
            let! aArr = Ndarray.toArray2D A
            let Ax = Array.init m (fun i -
                let mutable sum = 0.0
                for j = 0 to n - 1 do
                    sum <- sum + aArr.[i, j] * xArr.[j]
                sum)
            
            let residual = 
                Array.map2 (-) Ax bArr
                |> Array.sumBy (fun diff -> diff * diff)
                |> sqrt
            
            return (x, residual * residual, rank, s)
        }
    
    /// <summary>Moore-Penrose pseudoinverse using SVD.
/// A⁺ = V * Σ⁺ * Ut
/// where Σ⁺ is pseudoinverse of singular value matrix.
/// </summary>let pinv (A: Ndarray<float>) : FowlResult<Ndarray<float>> =
        result {
            // SVD: A = U * S * Vt
            // Pseudoinverse: A⁺ = V * S⁺ * Ut
            
            let! u, s, vt = Factorizations.svd A
            let! uArr = Ndarray.toArray2D u
            let! vtArr = Ndarray.toArray2D vt
            
            let m = uArr.GetLength(0)
            let n = vtArr.GetLength(0)
            
            // Threshold for numerical stability
            let eps = 1e-15
            let maxS = if s.Length > 0 then Array.max s else 0.0
            let threshold = eps * maxS * float (max m n)
            
            // Create pseudoinverse of S: reciprocal of non-zero singular values
            let sPinv = Array.init s.Length (fun i -
                if s.[i] > threshold then 1.0 / s.[i] else 0.0)
            
            // Compute A⁺ = V * S⁺ * Ut
            // First: V * S⁺ (scale columns of Vt^T = rows of Vt)
            let vScaled = Array2D.init n n (fun i j -
                if j < sPinv.Length then
                    vtArr.[j, i] * sPinv.[j]  // Note: vt is transposed
                else
                    0.0)
            
            // Then: (V * S⁺) * Ut
            let pinvArr = Array2D.init n m (fun i j -
                let mutable sum = 0.0
                for k = 0 to n - 1 do
                    if k < sPinv.Length && sPinv.[k] > 0.0 then
                        sum <- sum + vScaled.[i, k] * uArr.[j, k]
                sum)
            
            return! Ndarray.ofArray2D pinvArr
        }
    
    /// <summary>Matrix rank using SVD.
/// Number of singular values above threshold.
/// </summary>let rank (A: Ndarray<float>) : FowlResult<int> =
        result {
            let! _, s, _ = Factorizations.svd A
            
            // Threshold: max(m,n) * eps * max(s)
            let! shape = Ndarray.shape A |> Result.ofOption (Error.invalidState "Cannot get shape")
            let maxDim = Array.max shape
            let eps = 1e-10
            let maxS = Array.max s
            let threshold = float maxDim * eps * maxS
            
            return s |> Array.filter (fun si -> si > threshold) |> Array.length
        }
    
    /// <summary>Condition number using SVD.
/// κ(A) = σ_max / σ_min
/// Large condition number indicates ill-conditioned matrix.
/// </summary>let cond (A: Ndarray<float>) : FowlResult<float> =
        result {
            let! _, s, _ = Factorizations.svd A
            
            if s.Length = 0 then
                return! Error.invalidState "Cannot compute condition number of empty matrix"
            
            let maxS = Array.max s
            let minS = Array.max [|Array.min s; 1e-15|]
            
            return maxS / minS
        }
    
    /// <summary>Null space using SVD.
/// Returns basis for null space (vectors x such that Ax = 0)
/// </summary>let nullSpace (A: Ndarray<float>) : FowlResult<Ndarray<float>> =
        result {
            // SVD: A = U * S * Vt
            // Null space is spanned by columns of V corresponding to zero singular values
            
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
                // No null space (full rank)
                return! Ndarray.zeros [||]
            else
                // Extract null space basis
                let basis = 
                    Array2D.init n nullIndices.Length (fun i j -
                        vtArr.[nullIndices.[j], i])
                
                return! Ndarray.ofArray2D basis
        }
    
    /// <summary>Orthonormal basis for range (column space) using SVD.
/// Returns orthonormal basis for range of A.
/// </summary>let orth (A: Ndarray<float>) : FowlResult<Ndarray<float>> =
        result {
            // SVD: A = U * S * Vt
            // Range is spanned by columns of U corresponding to non-zero singular values
            
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
            
            // Extract range basis
            let basis = 
                Array2D.init m rangeIndices.Length (fun i j -
                    uArr.[i, rangeIndices.[j]])
            
            return! Ndarray.ofArray2D basis
        }