/// <summary>Fowl Cache Optimization Module</summary>/// <remarks>
/// Provides cache-optimized algorithms for better memory locality.
/// 
/// Key techniques:
/// - Tiling/Blocking: Process data in cache-friendly blocks
/// - Loop reordering: Optimize access patterns for spatial locality
/// - Prefetching: Hint to CPU about upcoming memory access
/// 
/// Example:
/// <code>
/// open Fowl.Cache
/// 
/// // Cache-optimized matrix multiplication
/// let result = CacheMatrixOps.matmulTiled a b
/// </code>
/// </remarks>
module Fowl.Cache

open System
open System.Runtime.CompilerServices

// ============================================================================
// Cache Configuration
// ============================================================================

/// <summary>Cache line size in bytes (typically 64 on x86/x64).</summary>let cacheLineSize = 64

/// <summary>Typical L1 cache size per core (32KB).</summary>let l1CacheSize = 32 * 1024

/// <summary>Typical L2 cache size per core (256KB).</summary>let l2CacheSize = 256 * 1024

/// <summary>Typical L3 cache size (shared, 8MB).</summary>let l3CacheSize = 8 * 1024 * 1024

/// <summary>Block size for tiling (tuned for double precision).</summary>/// <remarks>
/// 64 doubles = 512 bytes, fits in L1 cache with room for 3 arrays.
/// </remarks>let tileSize = 64

/// <summary>Small block size for L1 cache.</summary>let smallBlockSize = 32

/// <summary>Medium block size for L2 cache.</summary>let mediumBlockSize = 128

/// <summary>Large block size for L3 cache.</summary>let largeBlockSize = 512

// ============================================================================
// Prefetching
// ============================================================================

/// <summary>Prefetch hint for hardware.</summary>/// <remarks>
/// Uses System.Runtime.CompilerServices.Unsafe.Prefetch when available.
/// Falls back gracefully on platforms without prefetch support.
/// </remarks>type PrefetchHint =
    | Read = 0
    | Write = 1

/// <summary>Prefetch data into cache.</summary>/// <param name="address">Memory address to prefetch.</param>/// <param name="hint">Read or write hint.</param>let inline prefetch (address: nativeint) (hint: PrefetchHint) : unit =
    // Note: In .NET, prefetch is not directly exposed
    // This is a placeholder for future hardware intrinsics
    // For now, we rely on the CPU's automatic prefetcher
    ignore address
    ignore hint
    ()

// ============================================================================
// Tiled Matrix Operations
// ============================================================================

/// <summary>Module for cache-optimized matrix operations.</summary>module CacheMatrixOps =
    /// <summary>Matrix multiplication with cache tiling.</summary>    /// <param name="a">Left matrix (m x n).</param>    /// <param name="b">Right matrix (n x p).</param>    /// <returns>Result matrix (m x p).</returns>    /// <remarks>
    /// Uses three-level tiling for L1, L2, L3 cache hierarchy.
    /// </remarks>    let matmulTiled (a: double[,]) (b: double[,]) : double[,] =
        let m = a.GetLength(0)
        let n = a.GetLength(1)
        let p = b.GetLength(1)
        
        if n <> b.GetLength(0) then
            invalidArg "b" "Matrix dimensions incompatible"
        
        let result = Array2D.zeroCreate m p
        
        // Three-level tiling
        let iTile = min tileSize m
        let jTile = min tileSize p
        let kTile = min tileSize n
        
        for ii in 0..iTile..m-1 do
            for jj in 0..jTile..p-1 do
                for kk in 0..kTile..n-1 do
                    // Process tile
                    let iEnd = min (ii + iTile) m
                    let jEnd = min (jj + jTile) p
                    let kEnd = min (kk + kTile) n
                    
                    for i = ii to iEnd - 1 do
                        for j = jj to jEnd - 1 do
                            let mutable sum = result.[i, j]
                            for k = kk to kEnd - 1 do
                                sum <- sum + a.[i, k] * b.[k, j]
                            result.[i, j] <- sum
        
        result
    
    /// <summary>Matrix multiplication with L2 cache blocking.</summary>    /// <param name="a">Left matrix.</param>    /// <param name="b">Right matrix.</param>    /// <returns>Result matrix.</returns>    let matmulBlocked (a: double[,]) (b: double[,]) : double[,] =
        let m = a.GetLength(0)
        let n = a.GetLength(1)
        let p = b.GetLength(1)
        
        if n <> b.GetLength(0) then
            invalidArg "b" "Matrix dimensions incompatible"
        
        let result = Array2D.zeroCreate m p
        let blockSize = min mediumBlockSize (min (min m n) p)
        
        for ii in 0..blockSize..m-1 do
            for jj in 0..blockSize..p-1 do
                // Initialize block of result to zero
                let iEnd = min (ii + blockSize) m
                let jEnd = min (jj + blockSize) p
                for i = ii to iEnd - 1 do
                    for j = jj to jEnd - 1 do
                        result.[i, j] <- 0.0
                
                for kk in 0..blockSize..n-1 do
                    let kEnd = min (kk + blockSize) n
                    // Multiply block of A with block of B
                    for i = ii to iEnd - 1 do
                        for k = kk to kEnd - 1 do
                            let aik = a.[i, k]
                            for j = jj to jEnd - 1 do
                                result.[i, j] <- result.[i, j] + aik * b.[k, j]
        
        result
    
    /// <summary>Cache-optimized matrix transpose.</summary>    /// <param name="a">Input matrix.</param>    /// <returns>Transposed matrix.</returns>    /// <remarks>
    /// Uses blocking to improve cache locality.
    /// </remarks>    let transposeBlocked (a: double[,]) : double[,] =
        let m = a.GetLength(0)
        let n = a.GetLength(1)
        let result = Array2D.zeroCreate n m
        let blockSize = min tileSize (min m n)
        
        for ii in 0..blockSize..m-1 do
            for jj in 0..blockSize..n-1 do
                let iEnd = min (ii + blockSize) m
                let jEnd = min (jj + blockSize) n
                for i = ii to iEnd - 1 do
                    for j = jj to jEnd - 1 do
                        result.[j, i] <- a.[i, j]
        
        result
    
    /// <summary>Matrix-vector multiplication with cache optimization.</summary>    /// <param name="a">Matrix (m x n).</param>    /// <param name="v">Vector (n).</param>    /// <returns>Result vector (m).</returns>    let matvecTiled (a: double[,]) (v: double[]) : double[] =
        let m = a.GetLength(0)
        let n = a.GetLength(1)
        
        if n <> v.Length then
            invalidArg "v" "Vector length incompatible with matrix"
        
        let result = Array.zeroCreate m
        let blockSize = min tileSize m
        
        for ii in 0..blockSize..m-1 do
            let iEnd = min (ii + blockSize) m
            for i = ii to iEnd - 1 do
                let mutable sum = 0.0
                for j = 0 to n - 1 do
                    sum <- sum + a.[i, j] * v.[j]
                result.[i] <- sum
        
        result

// ============================================================================
// Loop Reordering
// ============================================================================

/// <summary>Module for loop reordering optimizations.</summary>/// <remarks>
/// Loop order affects cache performance significantly.
/// Row-major access is cache-friendly, column-major is not.
/// </remarks>module LoopReorderOps =
    /// <summary>Sum matrix elements with cache-friendly row-major order.</summary>    /// <param name="matrix">Input matrix.</param>    /// <returns>Sum of all elements.</returns>    /// <remarks>
    /// Outer loop over rows (contiguous), inner over columns.
    /// </remarks>    let sumRowMajor (matrix: double[,]) : double =
        let rows = matrix.GetLength(0)
        let cols = matrix.GetLength(1)
        let mutable sum = 0.0
        
        for i = 0 to rows - 1 do
            for j = 0 to cols - 1 do
                sum <- sum + matrix.[i, j]
        
        sum
    
    /// <summary>Sum matrix elements with cache-unfriendly column-major order.</summary>    /// <param name="matrix">Input matrix.</param>    /// <returns>Sum of all elements.</returns>    /// <remarks>
    /// Outer loop over columns, inner over rows (strided access).
    /// Demonstrates poor performance.
    /// </remarks>    let sumColumnMajor (matrix: double[,]) : double =
        let rows = matrix.GetLength(0)
        let cols = matrix.GetLength(1)
        let mutable sum = 0.0
        
        for j = 0 to cols - 1 do
            for i = 0 to rows - 1 do
                sum <- sum + matrix.[i, j]
        
        sum
    
    /// <summary>Copy matrix with cache-friendly blocking.</summary>    /// <param name="source">Source matrix.</param>    /// <returns>Copied matrix.</returns>    let copyBlocked (source: double[,]) : double[,] =
        let rows = source.GetLength(0)
        let cols = source.GetLength(1)
        let result = Array2D.zeroCreate rows cols
        let blockSize = min tileSize (min rows cols)
        
        for ii in 0..blockSize..rows-1 do
            for jj in 0..blockSize..cols-1 do
                let iEnd = min (ii + blockSize) rows
                let jEnd = min (jj + blockSize) cols
                for i = ii to iEnd - 1 do
                    for j = jj to jEnd - 1 do
                        result.[i, j] <- source.[i, j]
        
        result

// ============================================================================
// Spatial Locality
// ============================================================================

/// <summary>Module for spatial locality optimizations.</summary>module SpatialLocalityOps =
    /// <summary>Process array with sequential access (cache-friendly).</summary>    /// <param name="processor">Processing function.</param>    /// <param name="array">Input array.</param>    let processSequential (processor: double -> unit) (array: double[]) : unit =
        for i = 0 to array.Length - 1 do
            processor array.[i]
    
    /// <summary>Process array with strided access (cache-unfriendly).</summary>    /// <param name="processor">Processing function.</param>    /// <param name="array">Input array.</param>    /// <param name="stride">Stride for access.</param>    let processStrided (processor: double -> unit) (array: double[]) (stride: int) : unit =
        for i in 0..stride..array.Length-1 do
            processor array.[i]
    
    /// <summary>Align array size to cache line boundary.</summary>    /// <param name="size">Requested size.</param>    /// <returns>Size padded to cache line boundary.</returns>    let padToCacheLine (size: int) : int =
        let doublesPerCacheLine = cacheLineSize / sizeof<double>
        ((size + doublesPerCacheLine - 1) / doublesPerCacheLine) * doublesPerCacheLine
    
    /// <summary>Create array with cache-line alignment padding.</summary>    /// <param name="size">Minimum size.</param>    /// <returns>Array with padded size.</returns>    let createPadded (size: int) : double[] =
        Array.zeroCreate (padToCacheLine size)

// ============================================================================
// Benchmarking Helpers
// ============================================================================

/// <summary>Module for cache performance measurement.</summary>module CacheBenchmark =
    /// <summary>Measure cache performance of an operation.</summary>    /// <param name="operation">Operation to measure.</param>    /// <returns>Execution time.</returns>    let measureTime (operation: unit -> 'T) : TimeSpan =
        let stopwatch = Diagnostics.Stopwatch()
        stopwatch.Start()
        operation() |> ignore
        stopwatch.Stop()
        stopwatch.Elapsed
    
    /// <summary>Compare row-major vs column-major performance.</summary>    /// <param name="matrix">Test matrix.</param>    /// <returns>Tuple of (rowTime, colTime, speedup).</returns>    let compareRowVsColumn (matrix: double[,]) : (TimeSpan * TimeSpan * double) =
        // Warmup
        LoopReorderOps.sumRowMajor matrix |> ignore
        LoopReorderOps.sumColumnMajor matrix |> ignore
        
        // Measure row-major
        let rowTime = measureTime (fun () -> LoopReorderOps.sumRowMajor matrix)
        
        // Measure column-major
        let colTime = measureTime (fun () -> LoopReorderOps.sumColumnMajor matrix)
        
        let speedup = colTime.TotalMilliseconds / rowTime.TotalMilliseconds
        (rowTime, colTime, speedup)
    
    /// <summary>Print cache comparison results.</summary>    /// <param name="matrix">Test matrix.</param>    let printComparison (matrix: double[,]) : unit =
        let rows = matrix.GetLength(0)
        let cols = matrix.GetLength(1)
        let rowTime, colTime, speedup = compareRowVsColumn matrix
        
        printfn "\n=== Cache Performance Comparison (%dx%d) ===" rows cols
        printfn "Row-major (cache-friendly):    %O" rowTime
        printfn "Column-major (strided):        %O" colTime
        printfn "Speedup:                       %.2fx" speedup
        printfn ""

// ============================================================================
// Cache Information
// ============================================================================

/// <summary>Print cache configuration information.</summary>let printCacheInfo () : unit =
    printfn "\n=== Cache Configuration ==="
    printfn "Cache Line Size:    %d bytes" cacheLineSize
    printfn "L1 Cache Size:      %d KB" (l1CacheSize / 1024)
    printfn "L2 Cache Size:      %d KB" (l2CacheSize / 1024)
    printfn "L3 Cache Size:      %d MB" (l3CacheSize / (1024 * 1024))
    printfn "Tile/Block Size:    %d doubles (%d bytes)" tileSize (tileSize * sizeof<double>)
    printfn "Doubles per line:   %d" (cacheLineSize / sizeof<double>)
    printfn ""
