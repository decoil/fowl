/// <summary>Fowl SIMD - Hardware-Specific Intrinsics Wrapper</summary>
/// <remarks>
/// Provides F# wrappers around Fowl.Native.SIMD (C#) hardware intrinsics.
/// Automatically selects AVX2, SSE2, or scalar based on hardware capabilities.
/// 
/// This module extends Fowl.SIMD with hardware-specific optimizations that
/// provide 20-50% better performance than portable Vector&lt;T&gt; on AVX2 hardware.
/// </remarks>
module Fowl.SIMD.Hardware

open System
open Fowl.Native.SIMD

// Re-export hardware detection
let isAvx2Supported = KernelSelector.IsAvx2Supported
let isSse2Supported = KernelSelector.IsSse2Supported
let bestImplementation = KernelSelector.BestImplementation
let simdThreshold = KernelSelector.SimdThreshold

/// <summary>Print hardware SIMD detection information.</summary>let printHardwareInfo () : unit =
    KernelSelector.PrintSimdInfo()

// ============================================================================
// Element-wise Operations
// ============================================================================

/// <summary>Add two arrays using best available hardware instructions.</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>New array with element-wise sum.</returns>/// <remarks>
/// Uses AVX2 on modern CPUs (4 doubles at a time), SSE2 on older CPUs (2 doubles),
/// or scalar fallback for small arrays.
/// </remarks>
let add (a: double[]) (b: double[]) : double[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    let result = Array.zeroCreate a.Length
    KernelSelector.Add(a, b, result)
    result

/// <summary>Subtract two arrays using best available hardware instructions.</summary>/// <param name="a">First array (minuend).</param>/// <param name="b">Second array (subtrahend).</param>/// <returns>New array with element-wise difference.</returns>let sub (a: double[]) (b: double[]) : double[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    let result = Array.zeroCreate a.Length
    KernelSelector.Subtract(a, b, result)
    result

/// <summary>Multiply two arrays using best available hardware instructions.</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>New array with element-wise product.</returns>let mul (a: double[]) (b: double[]) : double[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    let result = Array.zeroCreate a.Length
    KernelSelector.Multiply(a, b, result)
    result

/// <summary>Divide two arrays using best available hardware instructions.</summary>/// <param name="a">First array (dividend).</param>/// <param name="b">Second array (divisor).</param>/// <returns>New array with element-wise quotient.</returns>let div (a: double[]) (b: double[]) : double[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    let result = Array.zeroCreate a.Length
    KernelSelector.Divide(a, b, result)
    result

/// <summary>Add scalar to each element using hardware instructions.</summary>/// <param name="a">Input array.</param>/// <param name="scalar">Value to add.</param>/// <returns>New array with scalar added.</returns>let addScalar (a: double[]) (scalar: double) : double[] =
    let result = Array.zeroCreate a.Length
    KernelSelector.AddScalar(a, scalar, result)
    result

/// <summary>Multiply each element by scalar using hardware instructions.</summary>/// <param name="a">Input array.</param>/// <param name="scalar">Value to multiply.</param>/// <returns>New array with scalar multiplied.</returns>let mulScalar (a: double[]) (scalar: double) : double[] =
    let result = Array.zeroCreate a.Length
    KernelSelector.MultiplyScalar(a, scalar, result)
    result

/// <summary>Negate all elements using hardware instructions.</summary>/// <param name="a">Input array.</param>/// <returns>New array with negated values.</returns>let negate (a: double[]) : double[] =
    let result = Array.zeroCreate a.Length
    KernelSelector.Negate(a, result)
    result

// ============================================================================
// Single-Precision Operations
// ============================================================================

/// <summary>Add two float arrays using best available hardware instructions.</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>New array with element-wise sum.</returns>/// <remarks>
/// AVX2: 8 floats at a time, SSE2: 4 floats at a time
/// </remarks>
let addSingle (a: single[]) (b: single[]) : single[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    let result = Array.zeroCreate a.Length
    KernelSelector.Add(a, b, result)
    result

/// <summary>Multiply two float arrays using best available hardware instructions.</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>New array with element-wise product.</returns>let mulSingle (a: single[]) (b: single[]) : single[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    let result = Array.zeroCreate a.Length
    KernelSelector.Multiply(a, b, result)
    result

// ============================================================================
// Reduction Operations
// ============================================================================

/// <summary>Sum all elements using best available hardware instructions.</summary>/// <param name="a">Input array.</param>/// <returns>Sum of all elements.</returns>/// <remarks>
/// Uses vectorized accumulation with horizontal reduction.
/// </remarks>let sum (a: double[]) : double =
    KernelSelector.Sum(a)

/// <summary>Calculate mean using hardware instructions.</summary>/// <param name="a">Input array.</param>/// <returns>Mean of all elements.</returns>/// <exception cref="System.ArgumentException">Thrown when array is empty.</exception>
let mean (a: double[]) : double =
    if a.Length = 0 then
        invalidArg "a" "Cannot compute mean of empty array"
    sum a / double a.Length

/// <summary>Dot product using best available hardware instructions.</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>Dot product of a and b.</returns>/// <example>
/// <code>
/// let a = [|1.0; 2.0; 3.0|]
/// let b = [|4.0; 5.0; 6.0|]
/// let result = dot a b  // 32.0
/// </code>
/// </example>
let dot (a: double[]) (b: double[]) : double =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    KernelSelector.Dot(a, b)

/// <summary>Find minimum element using hardware instructions.</summary>/// <param name="a">Input array.</param>/// <returns>Minimum value.</returns>/// <exception cref="System.ArgumentException">Thrown when array is empty.</exception>
let min (a: double[]) : double =
    KernelSelector.Min(a)

/// <summary>Find maximum element using hardware instructions.</summary>/// <param name="a">Input array.</param>/// <returns>Maximum value.</returns>/// <exception cref="System.ArgumentException">Thrown when array is empty.</exception>
let max (a: double[]) : double =
    KernelSelector.Max(a)

// ============================================================================
// In-Place Operations
// ============================================================================

/// <summary>Add arrays in-place using hardware instructions.</summary>/// <param name="result">Array to modify in place (also holds result).</param>/// <param name="b">Array to add.</param>let addInPlace (result: double[]) (b: double[]) : unit =
    if result.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    KernelSelector.AddInPlace(result, b)

/// <summary>Multiply array by scalar in-place using hardware instructions.</summary>/// <param name="result">Array to modify in place.</param>/// <param name="scalar">Value to multiply.</param>
let mulInPlace (result: double[]) (scalar: double) : unit =
    KernelSelector.MultiplyInPlace(result, scalar)

// ============================================================================
// Performance Comparison Utilities
// ============================================================================

/// <summary>Compare performance of different implementations.</summary>/// <param name="size">Array size for test.</param>let performanceComparison (size: int) : unit =
    printfn "\n=== Hardware SIMD Performance Comparison ==="
    printfn "Array size: %d elements" size
    printHardwareInfo()
    
    let rng = Random()
    let a = Array.init size (fun _ -> rng.NextDouble())
    let b = Array.init size (fun _ -> rng.NextDouble())
    let result = Array.zeroCreate size
    
    // Warmup
    KernelSelector.Add(a, b, result)
    
    let stopwatch = System.Diagnostics.Stopwatch()
    
    // Scalar
    stopwatch.Start()
    for _ = 1 to 100 do
        for i = 0 to size - 1 do
            result.[i] <- a.[i] + b.[i]
    stopwatch.Stop()
    let scalarTime = stopwatch.ElapsedMilliseconds
    
    // Vector<T> (portable)
    stopwatch.Restart()
    for _ = 1 to 100 do
        Fowl.SIMD.ElementWise.add a b |> ignore
    stopwatch.Stop()
    let vectorTTime = stopwatch.ElapsedMilliseconds
    
    // AVX2/SSE2 (hardware-specific)
    stopwatch.Restart()
    for _ = 1 to 100 do
        KernelSelector.Add(a, b, result)
    stopwatch.Stop()
    let hardwareTime = stopwatch.ElapsedMilliseconds
    
    printfn "\nAddition (100 iterations):"
    printfn "  Scalar:    %4d ms" scalarTime
    printfn "  Vector<T>: %4d ms (%.1fx speedup)" vectorTTime (float scalarTime / float vectorTTime)
    printfn "  Hardware:  %4d ms (%.1fx speedup)" hardwareTime (float scalarTime / float hardwareTime)
    printfn ""
