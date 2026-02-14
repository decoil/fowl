/// <summary>Fowl Configuration - Global Optimization Settings</summary>/// <remarks>
/// Central configuration for all Fowl optimization settings.
/// 
/// Example:
/// <code>
/// open Fowl.Config
/// 
/// // Enable all optimizations
/// Optimizations.enableAll()
/// 
/// // Or fine-tune individually
/// Optimizations.SIMD.enabled <- true
/// Optimizations.Parallel.threshold <- 5000
/// </code>
/// </remarks>
module Fowl.Config

open System

// ============================================================================
// SIMD Configuration
// ============================================================================

/// <summary>SIMD optimization settings.</summary>type SimdConfig = {
    mutable enabled: bool
    mutable useHardwareIntrinsics: bool  // AVX2/SSE2 vs Vector<T>
    mutable threshold: int  // Minimum array size for SIMD
}

let simdDefaults = {
    enabled = true
    useHardwareIntrinsics = true
    threshold = 16
}

// ============================================================================
// Parallel Configuration
// ============================================================================

/// <summary>Parallel optimization settings.</summary>type ParallelConfig = {
    mutable enabled: bool
    mutable threshold: int  // Minimum array size for parallelization
    mutable maxDegreeOfParallelism: int
}

let parallelDefaults = {
    enabled = true
    threshold = 10000
    maxDegreeOfParallelism = Environment.ProcessorCount
}

// ============================================================================
// Memory Configuration
// ============================================================================

/// <summary>Memory optimization settings.</summary>type MemoryConfig = {
    mutable useArrayPool: bool
    mutable useZeroCopyViews: bool
}

let memoryDefaults = {
    useArrayPool = true
    useZeroCopyViews = true
}

// ============================================================================
// Cache Configuration
// ============================================================================

/// <summary>Cache optimization settings.</summary>type CacheConfig = {
    mutable enabled: bool
    mutable tileSize: int
    mutable blockSize: int
}

let cacheDefaults = {
    enabled = true
    tileSize = 64
    blockSize = 128
}

// ============================================================================
// Global Configuration
// ============================================================================

/// <summary>Global optimization configuration.</summary>type OptimizationConfig = {
    simd: SimdConfig
    parallel: ParallelConfig
    memory: MemoryConfig
    cache: CacheConfig
}

/// <summary>Current optimization configuration.</summary>/// <remarks>
/// Modify this to change optimization behavior globally.
/// </remarks>let mutable current = {
    simd = simdDefaults
    parallel = parallelDefaults
    memory = memoryDefaults
    cache = cacheDefaults
}

// ============================================================================
// Configuration Management
// ============================================================================

/// <summary>Reset all optimizations to defaults.</summary>let resetToDefaults () =
    current.simd <- simdDefaults
    current.parallel <- parallelDefaults
    current.memory <- memoryDefaults
    current.cache <- cacheDefaults

/// <summary>Enable all optimizations.</summary>let enableAll () =
    current.simd.enabled <- true
    current.simd.useHardwareIntrinsics <- true
    current.parallel.enabled <- true
    current.memory.useArrayPool <- true
    current.memory.useZeroCopyViews <- true
    current.cache.enabled <- true

/// <summary>Disable all optimizations (use scalar/sequential only).</summary>let disableAll () =
    current.simd.enabled <- false
    current.parallel.enabled <- false
    current.memory.useArrayPool <- false
    current.memory.useZeroCopyViews <- false
    current.cache.enabled <- false

/// <summary>Print current configuration.</summary>let printConfig () =
    printfn "\n=== Fowl Optimization Configuration ==="
    printfn ""
    printfn "SIMD:"
    printfn "  Enabled:              %b" current.simd.enabled
    printfn "  Hardware Intrinsics:  %b" current.simd.useHardwareIntrinsics
    printfn "  Threshold:            %d elements" current.simd.threshold
    printfn ""
    printfn "Parallel:"
    printfn "  Enabled:              %b" current.parallel.enabled
    printfn "  Threshold:            %d elements" current.parallel.threshold
    printfn "  Max Parallelism:      %d cores" current.parallel.maxDegreeOfParallelism
    printfn ""
    printfn "Memory:"
    printfn "  Use ArrayPool:        %b" current.memory.useArrayPool
    printfn "  Use Zero-Copy Views:  %b" current.memory.useZeroCopyViews
    printfn ""
    printfn "Cache:"
    printfn "  Enabled:              %b" current.cache.enabled
    printfn "  Tile Size:            %d" current.cache.tileSize
    printfn "  Block Size:           %d" current.cache.blockSize
    printfn ""

// ============================================================================
// Auto-Detection
// ============================================================================

/// <summary>Detect hardware capabilities and configure optimally.</summary>/// <returns>String describing detected configuration.</returns>let autoDetect () : string =
    let sb = System.Text.StringBuilder()
    
    // Check SIMD support
    let simdSupported =
        if Fowl.Native.SIMD.KernelSelector.IsAvx2Supported then
            current.simd.useHardwareIntrinsics <- true
            "AVX2 (256-bit)"
        elif Fowl.Native.SIMD.KernelSelector.IsSse2Supported then
            current.simd.useHardwareIntrinsics <- true
            "SSE2 (128-bit)"
        else
            current.simd.useHardwareIntrinsics <- false
            "Vector<T> only"
    
    sb.AppendLine(sprintf "SIMD: %s" simdSupported) |> ignore
    
    // Check processor count
    let cores = Environment.ProcessorCount
    current.parallel.maxDegreeOfParallelism <- cores
    sb.AppendLine(sprintf "Cores: %d" cores) |> ignore
    
    // Enable all by default
    enableAll()
    
    sb.AppendLine("Optimizations: Enabled all") |> ignore
    sb.ToString()

/// <summary>Initialize Fowl with optimal settings for current hardware.</summary>let initialize () : unit =
    printfn "%s" (autoDetect())
    printConfig()

// ============================================================================
// Helper Functions for Internal Use
// ============================================================================

/// <summary>Check if SIMD should be used for given array size.</summary>/// <param name="length">Array length.</param>/// <returns>true if SIMD should be used.</returns>let shouldUseSimd (length: int) : bool =
    current.simd.enabled && length >= current.simd.threshold

/// <summary>Check if parallel should be used for given array size.</summary>/// <param name="length">Array length.</param>/// <returns>true if parallel should be used.</returns>let shouldUseParallel (length: int) : bool =
    current.parallel.enabled && length >= current.parallel.threshold

/// <summary>Check if hardware intrinsics should be used.</summary>/// <returns>true if hardware intrinsics should be used.</returns>let shouldUseHardwareIntrinsics () : bool =
    current.simd.enabled && current.simd.useHardwareIntrinsics
