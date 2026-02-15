/// <summary>Fowl Optimized Operations - High-Performance Primitives</summary>
/// <remarks>
/// High-performance element-wise operations using SIMD, parallelization,
/// and automatic fallback to scalar for small arrays.
/// 
/// These operations automatically select the best implementation based on:
/// - Array size (thresholds for SIMD/Parallel)
/// - Hardware capabilities (AVX2, SSE2, Vector<T>)
/// - Configuration settings
/// 
/// Example:
/// <code>
/// open Fowl.Optimized
/// 
/// // Automatically uses best implementation
/// let result = Optimized.add a b
/// </code>
/// </remarks>
module Fowl.Optimized

open System
open Fowl.Config

// ============================================================================
// Element-wise Operations with Auto-Selection
// ============================================================================

/// <summary>Element-wise addition with automatic optimization selection.</summary>
/// <param name="a">First array.</param>
/// <param name="b">Second array.</param>
/// <returns>New array with element-wise sum.</returns>
/// <remarks>
/// Automatically selects from (fastest to slowest):
/// 1. Parallel + Hardware SIMD (AVX2)
/// 2. Hardware SIMD (AVX2/SSE2)
/// 3. Portable SIMD (Vector<T>)
/// 4. Parallel scalar
/// 5. Sequential scalar
/// </remarks>
let add (a: double[]) (b: double[]) : double[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    
    let length = a.Length
    let result = Array.zeroCreate length
    
    // Strategy selection based on size and config
    if shouldUseParallel length && shouldUseHardwareIntrinsics() then
        // Parallel + Hardware SIMD (fastest)
        Fowl.Native.SIMD.KernelSelector.Add(a, b, result)
        // TODO: Add true parallel+simd version
    elif shouldUseHardwareIntrinsics() then
        // Hardware SIMD only
        Fowl.Native.SIMD.KernelSelector.Add(a, b, result)
    elif shouldUseSimd length then
        // Portable SIMD
        Fowl.SIMD.ElementWise.add a b
    elif shouldUseParallel length then
        // Parallel scalar
        Fowl.Parallel.ParallelOps.add a b
    else
        // Sequential scalar (fallback)
        for i = 0 to length - 1 do
            result.[i] <- a.[i] + b.[i]
        result

/// <summary>Element-wise subtraction with automatic optimization selection.</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>New array with element-wise difference.</returns>let sub (a: double[]) (b: double[]) : double[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    
    let length = a.Length
    let result = Array.zeroCreate length
    
    if shouldUseHardwareIntrinsics() && length >= current.simd.threshold then
        Fowl.Native.SIMD.KernelSelector.Subtract(a, b, result)
    elif shouldUseSimd length then
        Fowl.SIMD.ElementWise.sub a b
    elif shouldUseParallel length then
        Fowl.Parallel.ParallelOps.sub a b
    else
        for i = 0 to length - 1 do
            result.[i] <- a.[i] - b.[i]
        result

/// <summary>Element-wise multiplication with automatic optimization selection.</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>New array with element-wise product.</returns>let mul (a: double[]) (b: double[]) : double[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    
    let length = a.Length
    let result = Array.zeroCreate length
    
    if shouldUseHardwareIntrinsics() && length >= current.simd.threshold then
        Fowl.Native.SIMD.KernelSelector.Multiply(a, b, result)
    elif shouldUseSimd length then
        Fowl.SIMD.ElementWise.mul a b
    elif shouldUseParallel length then
        Fowl.Parallel.ParallelOps.mul a b
    else
        for i = 0 to length - 1 do
            result.[i] <- a.[i] * b.[i]
        result

/// <summary>Element-wise division with automatic optimization selection.</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>New array with element-wise quotient.</returns>let div (a: double[]) (b: double[]) : double[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    
    let length = a.Length
    let result = Array.zeroCreate length
    
    if shouldUseHardwareIntrinsics() && length >= current.simd.threshold then
        Fowl.Native.SIMD.KernelSelector.Divide(a, b, result)
    elif shouldUseSimd length then
        Fowl.SIMD.ElementWise.div a b
    else
        for i = 0 to length - 1 do
            result.[i] <- a.[i] / b.[i]
        result

/// <summary>Multiply array by scalar with automatic optimization.</summary>/// <param name="a">Input array.</param>/// <param name="scalar">Scalar value.</param>/// <returns>New array with multiplied values.</returns>let mulScalar (a: double[]) (scalar: double) : double[] =
    let length = a.Length
    let result = Array.zeroCreate length
    
    if shouldUseHardwareIntrinsics() && length >= current.simd.threshold then
        Fowl.Native.SIMD.KernelSelector.MultiplyScalar(a, scalar, result)
    elif shouldUseSimd length then
        Fowl.SIMD.ElementWise.mulScalar a scalar
    elif shouldUseParallel length then
        Fowl.Parallel.ParallelOps.mulScalar a scalar
    else
        for i = 0 to length - 1 do
            result.[i] <- a.[i] * scalar
        result

/// <summary>Negate array with automatic optimization.</summary>/// <param name="a">Input array.</param>/// <returns>New array with negated values.</returns>let negate (a: double[]) : double[] =
    let length = a.Length
    let result = Array.zeroCreate length
    
    if shouldUseHardwareIntrinsics() && length >= current.simd.threshold then
        Fowl.Native.SIMD.KernelSelector.Negate(a, result)
    elif shouldUseSimd length then
        Fowl.SIMD.ElementWise.negate a
    else
        for i = 0 to length - 1 do
            result.[i] <- -a.[i]
        result

/// <summary>Map function with automatic parallelization.</summary>/// <param name="mapping">Mapping function.</param>/// <param name="a">Input array.</param>/// <returns>New array with mapped values.</returns>let map (mapping: double -> double) (a: double[]) : double[] =
    if shouldUseParallel a.Length then
        Fowl.Parallel.ParallelOps.map mapping a
    else
        Array.map mapping a

// ============================================================================
// Reduction Operations with Auto-Selection
// ============================================================================

/// <summary>Sum all elements with automatic optimization selection.</summary>/// <param name="a">Input array.</param>/// <returns>Sum of all elements.</returns>let sum (a: double[]) : double =
    if shouldUseHardwareIntrinsics() && a.Length >= current.simd.threshold then
        Fowl.Native.SIMD.KernelSelector.Sum(a)
    elif shouldUseSimd a.Length then
        Fowl.SIMD.Reductions.sum a
    elif shouldUseParallel a.Length then
        Fowl.Parallel.ParallelReductions.sum a
    else
        Array.sum a

/// <summary>Calculate mean with automatic optimization.</summary>/// <param name="a">Input array.</param>/// <returns>Mean of all elements.</returns>let mean (a: double[]) : double =
    if a.Length = 0 then
        invalidArg "a" "Cannot compute mean of empty array"
    sum a / double a.Length

/// <summary>Dot product with automatic optimization selection.</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>Dot product.</returns>let dot (a: double[]) (b: double[]) : double =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    
    if shouldUseHardwareIntrinsics() && a.Length >= current.simd.threshold then
        Fowl.Native.SIMD.KernelSelector.Dot(a, b)
    elif shouldUseSimd a.Length then
        Fowl.SIMD.Reductions.dot a b
    elif shouldUseParallel a.Length then
        Fowl.Parallel.ParallelReductions.dot a b
    else
        let mutable s = 0.0
        for i = 0 to a.Length - 1 do
            s <- s + a.[i] * b.[i]
        s

/// <summary>Find minimum with automatic optimization.</summary>/// <param name="a">Input array.</param>/// <returns>Minimum value.</returns>let min (a: double[]) : double =
    if a.Length = 0 then
        invalidArg "a" "Cannot find min of empty array"
    
    if shouldUseHardwareIntrinsics() && a.Length >= current.simd.threshold then
        Fowl.Native.SIMD.KernelSelector.Min(a)
    elif shouldUseParallel a.Length then
        Fowl.Parallel.ParallelReductions.min a
    else
        Array.min a

/// <summary>Find maximum with automatic optimization.</summary>/// <param name="a">Input array.</param>/// <returns>Maximum value.</returns>let max (a: double[]) : double =
    if a.Length = 0 then
        invalidArg "a" "Cannot find max of empty array"
    
    if shouldUseHardwareIntrinsics() && a.Length >= current.simd.threshold then
        Fowl.Native.SIMD.KernelSelector.Max(a)
    elif shouldUseParallel a.Length then
        Fowl.Parallel.ParallelReductions.max a
    else
        Array.max a

// ============================================================================
// Matrix Operations with Cache Optimization
// ============================================================================

/// <summary>Matrix multiplication with cache optimization.</summary>/// <param name="a">Left matrix (m x n).</param>/// <param name="b">Right matrix (n x p).</param>/// <returns>Result matrix (m x p).</returns>let matmul (a: double[,]) (b: double[,]) : double[,] =
    let m = a.GetLength(0)
    let n = a.GetLength(1)
    let p = b.GetLength(1)
    
    if n <> b.GetLength(0) then
        invalidArg "b" "Matrix dimensions incompatible"
    
    // Use cache-optimized version for large matrices
    if current.cache.enabled && (m > 100 || n > 100 || p > 100) then
        Fowl.Cache.CacheMatrixOps.matmulTiled a b
    elif current.parallel.enabled && (m * p >= current.parallel.threshold) then
        Fowl.Parallel.ParallelMatrixOps.matmul a b
    else
        // Naive sequential
        let result = Array2D.zeroCreate m p
        for i = 0 to m - 1 do
            for j = 0 to p - 1 do
                let mutable sum = 0.0
                for k = 0 to n - 1 do
                    sum <- sum + a.[i, k] * b.[k, j]
                result.[i, j] <- sum
        result

/// <summary>Matrix transpose with cache optimization.</summary>/// <param name="a">Input matrix.</param>/// <returns>Transposed matrix.</returns>let transpose (a: double[,]) : double[,] =
    let m = a.GetLength(0)
    let n = a.GetLength(1)
    
    if current.cache.enabled && (m > 64 || n > 64) then
        Fowl.Cache.CacheMatrixOps.transposeBlocked a
    else
        let result = Array2D.zeroCreate n m
        for i = 0 to m - 1 do
            for j = 0 to n - 1 do
                result.[j, i] <- a.[i, j]
        result

// ============================================================================
// Convenience Functions
// ============================================================================

/// <summary>Check what implementation would be used for given size.</summary>/// <param name="length">Array length.</param>/// <returns>Description of implementation that would be used.</returns>let getImplementationInfo (length: int) : string =
    if shouldUseParallel length && shouldUseHardwareIntrinsics() then
        "Parallel + Hardware SIMD (AVX2/SSE2)"
    elif shouldUseHardwareIntrinsics() then
        "Hardware SIMD (AVX2/SSE2)"
    elif shouldUseSimd length then
        "Portable SIMD (Vector<T>)"
    elif shouldUseParallel length then
        "Parallel Scalar"
    else
        "Sequential Scalar"

/// <summary>Print optimization information for debugging.</summary>/// <param name="length">Array length.</param>let printOptimizationInfo (length: int) : unit =
    printfn "Array size: %d" length
    printfn "Implementation: %s" (getImplementationInfo length)
    printfn ""
