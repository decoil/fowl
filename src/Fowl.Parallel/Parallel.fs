/// <summary>
/// Fowl Parallel Module - Multi-Core Parallelization
/// </summary>
/// <remarks>
/// Provides parallel operations for multi-core speedup on large arrays.
/// 
/// Automatically selects parallel vs sequential based on array size threshold.
/// Combines with SIMD for maximum performance.
/// 
/// Example:
/// <code>
/// open Fowl.Parallel
/// 
/// // Parallel addition (auto-detects array size)
/// let result = ParallelOps.add a b
/// 
/// // Force parallel with custom threshold
/// let result = ParallelOps.addWithThreshold 5000 a b
/// </code>
/// </remarks>
module Fowl.Parallel

open System
open System.Threading.Tasks

// ============================================================================
// Configuration
// ============================================================================

/// <summary>
/// Minimum array size to benefit from parallelization.
/// </summary>
/// <remarks>
/// Arrays smaller than this use sequential processing because
/// parallel overhead dominates for small arrays.
/// Default: 10,000 elements
/// </remarks>
let mutable parallelThreshold = 10000

/// <summary>
/// Get current parallel threshold.
/// </summary>
let getParallelThreshold () = parallelThreshold

/// <summary>
/// Set parallel threshold.
/// </summary>
/// <param name="threshold">New threshold value.</param>
let setParallelThreshold (threshold: int) =
    parallelThreshold <- max 1000 threshold  // Minimum 1000 to avoid overhead

/// <summary>
/// Check if array is large enough for parallel processing.
/// </summary>
/// <param name="length">Array length.</param>
/// <returns>true if parallel should be used.</returns>
let shouldParallelize (length: int) =
    length >= parallelThreshold

/// <summary>
/// Get number of logical processors.
/// </summary>
/// <returns>Processor count.</returns>
let processorCount = Environment.ProcessorCount

// ============================================================================
// Helper Functions
// ============================================================================

/// <summary>
/// Calculate chunk size for parallel processing.
/// </summary>
/// <param name="totalLength">Total array length.</param>
/// <param name="numChunks">Number of chunks (typically processor count).</param>
/// <returns>Chunk size.</returns>
let private calculateChunkSize (totalLength: int) (numChunks: int) =
    (totalLength + numChunks - 1) / numChunks  // Ceiling division

/// <summary>
/// Execute action in parallel over array range.
/// </summary>
/// <param name="length">Array length.</param>
/// <param name="action">Action to execute for each index.</param>
let private parallelFor (length: int) (action: int -> unit) =
    if shouldParallelize length then
        let numChunks = min processorCount 8  // Max 8 chunks to reduce overhead
        let chunkSize = calculateChunkSize length numChunks
        
        Parallel.For(0, numChunks, fun chunkIdx ->
            let start = chunkIdx * chunkSize
            let end' = min (start + chunkSize) length
            for i = start to end' - 1 do
                action i
        ) |> ignore
    else
        // Sequential fallback
        for i = 0 to length - 1 do
            action i

/// <summary>
/// Execute action in parallel with SIMD per chunk.
/// </summary>
/// <param name="length">Array length.</param>
/// <param name="simdAction">SIMD-optimized action for chunk.</param>
/// <param name="scalarAction">Scalar action for remainder.</param>
let private parallelSimd (length: int) 
                         (simdAction: int -> int -> unit)  // start, end
                         (scalarAction: int -> unit) =       // index
    if shouldParallelize length then
        let numChunks = min processorCount 8
        let chunkSize = calculateChunkSize length numChunks
        
        Parallel.For(0, numChunks, fun chunkIdx ->
            let start = chunkIdx * chunkSize
            let end' = min (start + chunkSize) length
            simdAction start end'
        ) |> ignore
        
        // Handle remainder sequentially (usually small)
        let processed = numChunks * chunkSize
        for i = processed to length - 1 do
            scalarAction i
    else
        simdAction 0 length

// ============================================================================
// Element-wise Parallel Operations
// ============================================================================

/// <summary>
/// Module for parallel element-wise operations.
/// </summary>
module ParallelOps =
    /// <summary>
    /// Add two arrays in parallel.
    /// </summary>
    /// <param name="a">First array.</param>
    /// <param name="b">Second array.</param>
    /// <returns>New array with element-wise sum.</returns>
    let add (a: double[]) (b: double[]) : double[] =
        if a.Length <> b.Length then
            invalidArg "b" "Arrays must have same length"
        
        let result = Array.zeroCreate a.Length
        
        parallelFor a.Length (fun i ->
            result.[i] <- a.[i] + b.[i]
        )
        
        result
    
    /// <summary>
    /// Add two arrays with custom threshold.
    /// </summary>
    /// <param name="threshold">Custom parallel threshold.</param>
    /// <param name="a">First array.</param>
    /// <param name="b">Second array.</param>
    /// <returns>New array with element-wise sum.</returns>
    let addWithThreshold (threshold: int) (a: double[]) (b: double[]) : double[] =
        let oldThreshold = parallelThreshold
        parallelThreshold <- threshold
        try
            add a b
        finally
            parallelThreshold <- oldThreshold
    
    /// <summary>
    /// Subtract two arrays in parallel.
    /// </summary>
    /// <param name="a">First array.</param>
    /// <param name="b">Second array.</param>
    /// <returns>New array with element-wise difference.</returns>
    let sub (a: double[]) (b: double[]) : double[] =
        if a.Length <> b.Length then
            invalidArg "b" "Arrays must have same length"
        
        let result = Array.zeroCreate a.Length
        
        parallelFor a.Length (fun i ->
            result.[i] <- a.[i] - b.[i]
        )
        
        result
    
    /// <summary>
    /// Multiply two arrays in parallel.
    /// </summary>
    /// <param name="a">First array.</param>
    /// <param name="b">Second array.</param>
    /// <returns>New array with element-wise product.</returns>
    let mul (a: double[]) (b: double[]) : double[] =
        if a.Length <> b.Length then
            invalidArg "b" "Arrays must have same length"
        
        let result = Array.zeroCreate a.Length
        
        parallelFor a.Length (fun i ->
            result.[i] <- a.[i] * b.[i]
        )
        
        result
    
    /// <summary>
    /// Multiply array by scalar in parallel.
    /// </summary>
    /// <param name="a">Input array.</param>
    /// <param name="scalar">Scalar value.</param>
    /// <returns>New array with multiplied values.</returns>
    let mulScalar (a: double[]) (scalar: double) : double[] =
        let result = Array.zeroCreate a.Length
        
        parallelFor a.Length (fun i ->
            result.[i] <- a.[i] * scalar
        )
        
        result
    
    /// <summary>
    /// Apply function to each element in parallel.
    /// </summary>
    /// <param name="mapping">Mapping function.</param>
    /// <param name="a">Input array.</param>
    /// <returns>New array with mapped values.</returns>
    let map (mapping: double -> double) (a: double[]) : double[] =
        let result = Array.zeroCreate a.Length
        
        parallelFor a.Length (fun i ->
            result.[i] <- mapping a.[i]
        )
        
        result

// ============================================================================
// Parallel + SIMD Operations
// ============================================================================

/// <summary>
/// Module for combined Parallel + SIMD operations.
/// </summary>
/// <remarks>
/// Maximum performance: uses multiple cores with SIMD per core.
/// </remarks>
module ParallelSimdOps =
    open System.Numerics
    
    /// <summary>
    /// Add two arrays using Parallel + Vector<T> SIMD.
    /// </summary>
    /// <param name="a">First array.</param>
    /// <param name="b">Second array.</param>
    /// <returns>New array with element-wise sum.</returns>
    let add (a: double[]) (b: double[]) : double[] =
        if a.Length <> b.Length then
            invalidArg "b" "Arrays must have same length"
        
        let result = Array.zeroCreate a.Length
        let vecSize = Vector<double>.Count
        
        parallelSimd a.Length
            (fun start end' ->
                // SIMD per chunk
                let mutable i = start
                while i <= end' - vecSize do
                    let va = Vector(a, i)
                    let vb = Vector(b, i)
                    let vr = va + vb
                    vr.CopyTo(result, i)
                    i <- i + vecSize
                
                // Scalar remainder in this chunk
                while i < end' do
                    result.[i] <- a.[i] + b.[i]
                    i <- i + 1
            )
            (fun i ->
                result.[i] <- a.[i] + b.[i]
            )
        
        result
    
    /// <summary>
    /// Multiply two arrays using Parallel + Vector<T> SIMD.
    /// </summary>
    /// <param name="a">First array.</param>
    /// <param name="b">Second array.</param>
    /// <returns>New array with element-wise product.</returns>
    let mul (a: double[]) (b: double[]) : double[] =
        if a.Length <> b.Length then
            invalidArg "b" "Arrays must have same length"
        
        let result = Array.zeroCreate a.Length
        let vecSize = Vector<double>.Count
        
        parallelSimd a.Length
            (fun start end' ->
                let mutable i = start
                while i <= end' - vecSize do
                    let va = Vector(a, i)
                    let vb = Vector(b, i)
                    let vr = va * vb
                    vr.CopyTo(result, i)
                    i <- i + vecSize
                
                while i < end' do
                    result.[i] <- a.[i] * b.[i]
                    i <- i + 1
            )
            (fun i ->
                result.[i] <- a.[i] * b.[i]
            )
        
        result
    
    /// <summary>
    /// Add two arrays using Parallel + AVX2 (hardware SIMD).
    /// </summary>
    /// <param name="a">First array.</param>
    /// <param name="b">Second array.</param>
    /// <returns>New array with element-wise sum.</returns>
    let addAvx2 (a: double[]) (b: double[]) : double[] =
        if a.Length <> b.Length then
            invalidArg "b" "Arrays must have same length"
        
        let result = Array.zeroCreate a.Length
        
        // Use hardware-specific SIMD in chunks
        if shouldParallelize a.Length && Fowl.Native.SIMD.KernelSelector.IsAvx2Supported then
            let numChunks = min processorCount 8
            let chunkSize = (a.Length + numChunks - 1) / numChunks
            
            Parallel.For(0, numChunks, fun chunkIdx ->
                let start = chunkIdx * chunkSize
                let len = min chunkSize (a.Length - start)
                if len > 0 then
                    // Use AVX2 for this chunk
                    let chunkA: double[] = Array.zeroCreate len
                    let chunkB: double[] = Array.zeroCreate len
                    let chunkResult: double[] = Array.zeroCreate len
                    Array.Copy(a, start, chunkA, 0, len)
                    Array.Copy(b, start, chunkB, 0, len)
                    Fowl.Native.SIMD.Avx2Kernels.Add(chunkA, chunkB, chunkResult)
                    Array.Copy(chunkResult, 0, result, start, len)
            ) |> ignore
        else
            // Sequential fallback
            Fowl.Native.SIMD.KernelSelector.Add(a, b, result)
        
        result

// ============================================================================
// Parallel Reductions
// ============================================================================

/// <summary>
/// Module for parallel reduction operations.
/// </summary>
/// <remarks>
/// Parallel reductions use tree-based aggregation for better performance.
/// </remarks>
module ParallelReductions =
    /// <summary>
    /// Sum all elements in parallel.
    /// </summary>
    /// <param name="a">Input array.</param>
    /// <returns>Sum of all elements.</returns>
    let sum (a: double[]) : double =
        if a.Length < parallelThreshold then
            Array.sum a
        else
            let numChunks = min processorCount 8
            let chunkSize = calculateChunkSize a.Length numChunks
            let partialSums = Array.zeroCreate numChunks
            
            Parallel.For(0, numChunks, fun chunkIdx ->
                let start = chunkIdx * chunkSize
                let end' = System.Math.Min(start + chunkSize, a.Length)
                let mutable s = 0.0
                for i = start to end' - 1 do
                    s <- s + a.[i]
                partialSums.[chunkIdx] <- s
            ) |> ignore
            
            Array.sum partialSums
    
    /// <summary>
    /// Calculate mean in parallel.
    /// </summary>
    /// <param name="a">Input array.</param>
    /// <returns>Mean of all elements.</returns>
    let mean (a: double[]) : double =
        if a.Length = 0 then
            invalidArg "a" "Cannot compute mean of empty array"
        sum a / double a.Length
    
    /// <summary>
    /// Dot product in parallel.
    /// </summary>
    /// <param name="a">First array.</param>
    /// <param name="b">Second array.</param>
    /// <returns>Dot product of a and b.</returns>
    let dot (a: double[]) (b: double[]) : double =
        if a.Length <> b.Length then
            invalidArg "b" "Arrays must have same length"
        
        if a.Length < parallelThreshold then
            let mutable s = 0.0
            for i = 0 to a.Length - 1 do
                s <- s + a.[i] * b.[i]
            s
        else
            let numChunks = min processorCount 8
            let chunkSize = calculateChunkSize a.Length numChunks
            let partialSums = Array.zeroCreate numChunks
            
            Parallel.For(0, numChunks, fun chunkIdx ->
                let start = chunkIdx * chunkSize
                let end' = System.Math.Min(start + chunkSize, a.Length)
                let mutable s = 0.0
                for i = start to end' - 1 do
                    s <- s + a.[i] * b.[i]
                partialSums.[chunkIdx] <- s
            ) |> ignore
            
            Array.sum partialSums
    
    /// <summary>
    /// Find minimum in parallel.
    /// </summary>
    /// <param name="a">Input array.</param>
    /// <returns>Minimum value.</returns>
    let min (a: double[]) : double =
        if a.Length = 0 then
            invalidArg "a" "Cannot find min of empty array"
        
        if a.Length < parallelThreshold then
            let mutable m = a.[0]
            for i = 1 to a.Length - 1 do
                if a.[i] < m then m <- a.[i]
            m
        else
            let numChunks = System.Math.Min(processorCount, 8)
            let chunkSize = calculateChunkSize a.Length numChunks
            let partialMins = Array.zeroCreate numChunks
            
            Parallel.For(0, numChunks, fun chunkIdx ->
                let start = chunkIdx * chunkSize
                let end' = System.Math.Min(start + chunkSize, a.Length)
                let mutable m = a.[start]
                for i = start + 1 to end' - 1 do
                    if a.[i] < m then m <- a.[i]
                partialMins.[chunkIdx] <- m
            ) |> ignore
            
            let mutable result = partialMins.[0]
            for i = 1 to partialMins.Length - 1 do
                if partialMins.[i] < result then result <- partialMins.[i]
            result
    
    /// <summary>
    /// Find maximum in parallel.
    /// </summary>
    /// <param name="a">Input array.</param>
    /// <returns>Maximum value.</returns>
    let max (a: double[]) : double =
        if a.Length = 0 then
            invalidArg "a" "Cannot find max of empty array"
        
        if a.Length < parallelThreshold then
            let mutable m = a.[0]
            for i = 1 to a.Length - 1 do
                if a.[i] > m then m <- a.[i]
            m
        else
            let numChunks = System.Math.Min(processorCount, 8)
            let chunkSize = calculateChunkSize a.Length numChunks
            let partialMaxs = Array.zeroCreate numChunks
            
            Parallel.For(0, numChunks, fun chunkIdx ->
                let start = chunkIdx * chunkSize
                let end' = System.Math.Min(start + chunkSize, a.Length)
                let mutable m = a.[start]
                for i = start + 1 to end' - 1 do
                    if a.[i] > m then m <- a.[i]
                partialMaxs.[chunkIdx] <- m
            ) |> ignore
            
            let mutable result = partialMaxs.[0]
            for i = 1 to partialMaxs.Length - 1 do
                if partialMaxs.[i] > result then result <- partialMaxs.[i]
            result

// ============================================================================
// Matrix Operations
// ============================================================================

/// <summary>
/// Module for parallel matrix operations.
/// </summary>
module ParallelMatrixOps =
    /// <summary>
    /// Matrix multiplication with parallel outer loops.
    /// </summary>
    /// <param name="a">Left matrix (m x n).</param>
    /// <param name="b">Right matrix (n x p).</param>
    /// <returns>Result matrix (m x p).</returns>
    /// <remarks>
    /// Parallelizes over rows of result matrix.
    /// Inner loop is sequential for cache efficiency.
    /// </remarks>
    let matmul (a: double[,]) (b: double[,]) : double[,] =
        let m = a.GetLength(0)
        let n = a.GetLength(1)
        let p = b.GetLength(1)
        
        if n <> b.GetLength(0) then
            invalidArg "b" "Matrix dimensions incompatible"
        
        let result = Array2D.zeroCreate m p
        
        if m * p >= parallelThreshold then
            Parallel.For(0, m, fun i ->
                for j = 0 to p - 1 do
                    let mutable sum = 0.0
                    for k = 0 to n - 1 do
                        sum <- sum + a.[i, k] * b.[k, j]
                    result.[i, j] <- sum
            ) |> ignore
        else
            // Sequential fallback
            for i = 0 to m - 1 do
                for j = 0 to p - 1 do
                    let mutable sum = 0.0
                    for k = 0 to n - 1 do
                        sum <- sum + a.[i, k] * b.[k, j]
                    result.[i, j] <- sum
        
        result

// ============================================================================
// Diagnostics
// ============================================================================

/// <summary>
/// Print parallelization configuration.
/// </summary>
let printParallelInfo () : unit =
    printfn "\n=== Parallel Configuration ==="
    printfn "Processor Count: %d" processorCount
    printfn "Parallel Threshold: %d elements" parallelThreshold
    printfn "Parallel Enabled: %b" (parallelThreshold < Int32.MaxValue)
    printfn ""

/// <summary>
/// Test parallel vs sequential performance.
/// </summary>
/// <param name="size">Array size for test.</param>
let performanceTest (size: int) : unit =
    printfn "\n=== Parallel Performance Test ==="
    printfn "Array size: %d elements" size
    printParallelInfo()
    
    let rng = Random()
    let a = Array.init size (fun _ -> rng.NextDouble())
    let b = Array.init size (fun _ -> rng.NextDouble())
    
    // Warmup
    ParallelOps.add a b |> ignore
    
    let stopwatch = Diagnostics.Stopwatch()
    
    // Sequential
    let result1 = Array.zeroCreate size
    stopwatch.Start()
    for _ = 1 to 10 do
        for i = 0 to size - 1 do
            result1.[i] <- a.[i] + b.[i]
    stopwatch.Stop()
    let sequentialTime = stopwatch.ElapsedMilliseconds
    
    // Parallel
    stopwatch.Restart()
    for _ = 1 to 10 do
        ParallelOps.add a b |> ignore
    stopwatch.Stop()
    let parallelTime = stopwatch.ElapsedMilliseconds
    
    // Parallel + SIMD
    stopwatch.Restart()
    for _ = 1 to 10 do
        ParallelSimdOps.add a b |> ignore
    stopwatch.Stop()
    let parallelSimdTime = stopwatch.ElapsedMilliseconds
    
    printfn "Addition (10 iterations):"
    printfn "  Sequential:   %4d ms" sequentialTime
    printfn "  Parallel:     %4d ms (%.1fx speedup)" parallelTime (float sequentialTime / float parallelTime)
    printfn "  Parallel+SIMD:%4d ms (%.1fx speedup)" parallelSimdTime (float sequentialTime / float parallelSimdTime)
    printfn ""
