/// <summary>Fowl SIMD Element-wise Operations</summary>
/// <remarks>
/// Provides SIMD-accelerated element-wise arithmetic operations.
/// Falls back to scalar loops for small arrays or non-SIMD hardware.
/// </remarks>
module Fowl.SIMD.ElementWise

open System
open System.Numerics
open Fowl.SIMD.Core

// ============================================================================
// Double-precision Element-wise Operations
// ============================================================================

/// <summary>Add two double arrays element-wise using SIMD.</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>New array containing element-wise sum.</returns>/// <example>
/// <code>
/// let a = [|1.0; 2.0; 3.0; 4.0|]
/// let b = [|5.0; 6.0; 7.0; 8.0|]
/// let result = add a b  // [|6.0; 8.0; 10.0; 12.0|]
/// </code>
/// </example>
let add (a: double[]) (b: double[]) : double[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    
    let result = Array.zeroCreate a.Length
    
    // Use SIMD for large enough arrays on supported hardware
    if shouldUseSimd a.Length then
        let vecSize = vectorCountDouble
        let mutable i = 0
        
        // SIMD vectorized loop
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vb = Vector(b, i)
            let vr = va + vb
            vr.CopyTo(result, i)
            i <- i + vecSize
        
        // Scalar remainder
        while i < a.Length do
            result.[i] <- a.[i] + b.[i]
            i <- i + 1
    else
        // Pure scalar for small arrays
        addScalar a b result
    
    result

/// <summary>Subtract two double arrays element-wise using SIMD.</summary>/// <param name="a">First array (minuend).</param>/// <param name="b">Second array (subtrahend).</param>/// <returns>New array containing element-wise difference.</returns>/// <example>
/// <code>
/// let a = [|10.0; 20.0; 30.0|]
/// let b = [|1.0; 2.0; 3.0|]
/// let result = sub a b  // [|9.0; 18.0; 27.0|]
/// </code>
/// </example>
let sub (a: double[]) (b: double[]) : double[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    
    let result = Array.zeroCreate a.Length
    
    if shouldUseSimd a.Length then
        let vecSize = vectorCountDouble
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vb = Vector(b, i)
            let vr = va - vb
            vr.CopyTo(result, i)
            i <- i + vecSize
        
        while i < a.Length do
            result.[i] <- a.[i] - b.[i]
            i <- i + 1
    else
        subScalar a b result
    
    result

/// <summary>Multiply two double arrays element-wise using SIMD.</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>New array containing element-wise product.</returns>/// <example>
/// <code>
/// let a = [|2.0; 3.0; 4.0|]
/// let b = [|5.0; 6.0; 7.0|]
/// let result = mul a b  // [|10.0; 18.0; 28.0|]
/// </code>
/// </example>
let mul (a: double[]) (b: double[]) : double[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    
    let result = Array.zeroCreate a.Length
    
    if shouldUseSimd a.Length then
        let vecSize = vectorCountDouble
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vb = Vector(b, i)
            let vr = va * vb
            vr.CopyTo(result, i)
            i <- i + vecSize
        
        while i < a.Length do
            result.[i] <- a.[i] * b.[i]
            i <- i + 1
    else
        mulScalar a b result
    
    result

/// <summary>Divide two double arrays element-wise using SIMD.</summary>/// <param name="a">First array (dividend).</param>/// <param name="b">Second array (divisor).</param>/// <returns>New array containing element-wise quotient.</returns>/// <example>
/// <code>
/// let a = [|10.0; 20.0; 30.0|]
/// let b = [|2.0; 4.0; 5.0|]
/// let result = div a b  // [|5.0; 5.0; 6.0|]
/// </code>
/// </example>
let div (a: double[]) (b: double[]) : double[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    
    let result = Array.zeroCreate a.Length
    
    if shouldUseSimd a.Length then
        let vecSize = vectorCountDouble
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vb = Vector(b, i)
            let vr = va / vb
            vr.CopyTo(result, i)
            i <- i + vecSize
        
        while i < a.Length do
            result.[i] <- a.[i] / b.[i]
            i <- i + 1
    else
        divScalar a b result
    
    result

/// <summary>Negate all elements in array using SIMD.</summary>/// <param name="a">Input array.</param>/// <returns>New array with negated values.</returns>let negate (a: double[]) : double[] =
    let result = Array.zeroCreate a.Length
    
    if shouldUseSimd a.Length then
        let vecSize = vectorCountDouble
        let zero = Vector<double>(0.0)
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vr = zero - va
            vr.CopyTo(result, i)
            i <- i + vecSize
        
        while i < a.Length do
            result.[i] <- -a.[i]
            i <- i + 1
    else
        for i = 0 to a.Length - 1 do
            result.[i] <- -a.[i]
    
    result

/// <summary>Add scalar to each element using SIMD.</summary>/// <param name="a">Input array.</param>/// <param name="scalar">Value to add.</param>/// <returns>New array with scalar added.</returns>let addScalar (a: double[]) (scalar: double) : double[] =
    let result = Array.zeroCreate a.Length
    
    if shouldUseSimd a.Length then
        let vecSize = vectorCountDouble
        let vscalar = Vector<double>(scalar)
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vr = va + vscalar
            vr.CopyTo(result, i)
            i <- i + vecSize
        
        while i < a.Length do
            result.[i] <- a.[i] + scalar
            i <- i + 1
    else
        for i = 0 to a.Length - 1 do
            result.[i] <- a.[i] + scalar
    
    result

/// <summary>Multiply each element by scalar using SIMD.</summary>/// <param name="a">Input array.</param>/// <param name="scalar">Value to multiply.</param>/// <returns>New array with scalar multiplied.</returns>let mulScalar (a: double[]) (scalar: double) : double[] =
    let result = Array.zeroCreate a.Length
    
    if shouldUseSimd a.Length then
        let vecSize = vectorCountDouble
        let vscalar = Vector<double>(scalar)
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vr = va * vscalar
            vr.CopyTo(result, i)
            i <- i + vecSize
        
        while i < a.Length do
            result.[i] <- a.[i] * scalar
            i <- i + 1
    else
        for i = 0 to a.Length - 1 do
            result.[i] <- a.[i] * scalar
    
    result

// ============================================================================
// Single-precision Element-wise Operations
// ============================================================================

/// <summary>Add two single arrays element-wise using SIMD.</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>New array containing element-wise sum.</returns>let addSingle (a: single[]) (b: single[]) : single[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    
    let result = Array.zeroCreate a.Length
    
    // SIMD benefits are greater for single precision (more elements per vector)
    if isHardwareAccelerated && a.Length >= 32 then
        let vecSize = vectorCountSingle
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vb = Vector(b, i)
            let vr = va + vb
            vr.CopyTo(result, i)
            i <- i + vecSize
        
        while i < a.Length do
            result.[i] <- a.[i] + b.[i]
            i <- i + 1
    else
        addScalarSingle a b result
    
    result

/// <summary>Subtract two single arrays element-wise using SIMD.</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>New array containing element-wise difference.</returns>let subSingle (a: single[]) (b: single[]) : single[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    
    let result = Array.zeroCreate a.Length
    
    if isHardwareAccelerated && a.Length >= 32 then
        let vecSize = vectorCountSingle
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vb = Vector(b, i)
            let vr = va - vb
            vr.CopyTo(result, i)
            i <- i + vecSize
        
        while i < a.Length do
            result.[i] <- a.[i] - b.[i]
            i <- i + 1
    else
        subScalarSingle a b result
    
    result

/// <summary>Multiply two single arrays element-wise using SIMD.</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>New array containing element-wise product.</returns>let mulSingle (a: single[]) (b: single[]) : single[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    
    let result = Array.zeroCreate a.Length
    
    if isHardwareAccelerated && a.Length >= 32 then
        let vecSize = vectorCountSingle
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vb = Vector(b, i)
            let vr = va * vb
            vr.CopyTo(result, i)
            i <- i + vecSize
        
        while i < a.Length do
            result.[i] <- a.[i] * b.[i]
            i <- i + 1
    else
        mulScalarSingle a b result
    
    result

/// <summary>Divide two single arrays element-wise using SIMD.</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>New array containing element-wise quotient.</returns>let divSingle (a: single[]) (b: single[]) : single[] =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    
    let result = Array.zeroCreate a.Length
    
    if isHardwareAccelerated && a.Length >= 32 then
        let vecSize = vectorCountSingle
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vb = Vector(b, i)
            let vr = va / vb
            vr.CopyTo(result, i)
            i <- i + vecSize
        
        while i < a.Length do
            result.[i] <- a.[i] / b.[i]
            i <- i + 1
    else
        divScalarSingle a b result
    
    result

// ============================================================================
// In-place Operations (no allocation)
// ============================================================================

/// <summary>Add arrays in-place (no allocation).</summary>/// <param name="a">Source and destination array (modified in place).</param>/// <param name="b">Array to add.</param>let addInPlace (a: double[]) (b: double[]) : unit =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    
    if shouldUseSimd a.Length then
        let vecSize = vectorCountDouble
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vb = Vector(b, i)
            let vr = va + vb
            vr.CopyTo(a, i)
            i <- i + vecSize
        
        while i < a.Length do
            a.[i] <- a.[i] + b.[i]
            i <- i + 1
    else
        for i = 0 to a.Length - 1 do
            a.[i] <- a.[i] + b.[i]

/// <summary>Multiply array by scalar in-place.</summary>/// <param name="a">Array to modify in place.</param>/// <param name="scalar">Value to multiply.</param>let mulInPlace (a: double[]) (scalar: double) : unit =
    if shouldUseSimd a.Length then
        let vecSize = vectorCountDouble
        let vscalar = Vector<double>(scalar)
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vr = va * vscalar
            vr.CopyTo(a, i)
            i <- i + vecSize
        
        while i < a.Length do
            a.[i] <- a.[i] * scalar
            i <- i + 1
    else
        for i = 0 to a.Length - 1 do
            a.[i] <- a.[i] * scalar
