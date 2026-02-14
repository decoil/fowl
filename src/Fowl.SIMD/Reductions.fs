/// <summary>Fowl SIMD Reduction Operations</summary>
/// <remarks>
/// Provides SIMD-accelerated reduction operations like sum, dot product.
/// Reductions are harder to vectorize efficiently due to horizontal operations.
/// </remarks>
module Fowl.SIMD.Reductions

open System
open System.Numerics
open Fowl.SIMD.Core

// ============================================================================
// Sum Operations
// ============================================================================

/// <summary>Sum all elements in array using SIMD.</summary>/// <param name="a">Input array.</param>/// <returns>Sum of all elements.</returns>/// <remarks>
/// Uses vectorized accumulation followed by horizontal reduction.
/// SIMD benefit is less than element-wise ops due to horizontal add overhead.
/// </remarks>/// <example>
/// <code>
/// let a = [|1.0; 2.0; 3.0; 4.0|]
/// let result = sum a  // 10.0
/// </code>
/// </example>
let sum (a: double[]) : double =
    if a.Length < 32 || not isHardwareAccelerated then
        // Scalar for small arrays
        let mutable s = 0.0
        for i = 0 to a.Length - 1 do
            s <- s + a.[i]
        s
    else
        let vecSize = vectorCountDouble
        let mutable vecSum = Vector<double>(0.0)
        let mutable i = 0
        
        // Vectorized accumulation
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            vecSum <- vecSum + va
            i <- i + vecSize
        
        // Horizontal reduction of vector sum
        let mutable total = 0.0
        for j = 0 to vecSize - 1 do
            total <- total + vecSum.[j]
        
        // Add remainder
        while i < a.Length do
            total <- total + a.[i]
            i <- i + 1
        
        total

/// <summary>Sum all elements in single array using SIMD.</summary>/// <param name="a">Input array.</param>/// <returns>Sum of all elements.</returns>let sumSingle (a: single[]) : single =
    if a.Length < 64 || not isHardwareAccelerated then
        let mutable s = 0.0f
        for i = 0 to a.Length - 1 do
            s <- s + a.[i]
        s
    else
        let vecSize = vectorCountSingle
        let mutable vecSum = Vector<single>(0.0f)
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            vecSum <- vecSum + va
            i <- i + vecSize
        
        let mutable total = 0.0f
        for j = 0 to vecSize - 1 do
            total <- total + vecSum.[j]
        
        while i < a.Length do
            total <- total + a.[i]
            i <- i + 1
        
        total

/// <summary>Calculate mean using SIMD.</summary>/// <param name="a">Input array.</param>/// <returns>Mean of all elements.</returns>let mean (a: double[]) : double =
    if a.Length = 0 then
        invalidArg "a" "Cannot compute mean of empty array"
    sum a / double a.Length

/// <summary>Calculate mean using SIMD (single precision).</summary>/// <param name="a">Input array.</param>/// <returns>Mean of all elements.</returns>
let meanSingle (a: single[]) : single =
    if a.Length = 0 then
        invalidArg "a" "Cannot compute mean of empty array"
    sumSingle a / single a.Length

// ============================================================================
// Dot Product
// ============================================================================

/// <summary>Compute dot product using SIMD.</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>Dot product of a and b.</returns>/// <example>
/// <code>
/// let a = [|1.0; 2.0; 3.0|]
/// let b = [|4.0; 5.0; 6.0|]
/// let result = dot a b  // 32.0 (1*4 + 2*5 + 3*6)
/// </code>
/// </example>
let dot (a: double[]) (b: double[]) : double =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    
    if a.Length < 32 || not isHardwareAccelerated then
        // Scalar for small arrays
        let mutable s = 0.0
        for i = 0 to a.Length - 1 do
            s <- s + a.[i] * b.[i]
        s
    else
        let vecSize = vectorCountDouble
        let mutable vecSum = Vector<double>(0.0)
        let mutable i = 0
        
        // Vectorized: multiply then accumulate
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vb = Vector(b, i)
            vecSum <- vecSum + (va * vb)
            i <- i + vecSize
        
        // Horizontal reduction
        let mutable total = 0.0
        for j = 0 to vecSize - 1 do
            total <- total + vecSum.[j]
        
        // Remainder
        while i < a.Length do
            total <- total + a.[i] * b.[i]
            i <- i + 1
        
        total

/// <summary>Compute dot product using SIMD (single precision).</summary>/// <param name="a">First array.</param>/// <param name="b">Second array.</param>/// <returns>Dot product of a and b.</returns>let dotSingle (a: single[]) (b: single[]) : single =
    if a.Length <> b.Length then
        invalidArg "b" "Arrays must have same length"
    
    if a.Length < 64 || not isHardwareAccelerated then
        let mutable s = 0.0f
        for i = 0 to a.Length - 1 do
            s <- s + a.[i] * b.[i]
        s
    else
        let vecSize = vectorCountSingle
        let mutable vecSum = Vector<single>(0.0f)
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vb = Vector(b, i)
            vecSum <- vecSum + (va * vb)
            i <- i + vecSize
        
        let mutable total = 0.0f
        for j = 0 to vecSize - 1 do
            total <- total + vecSum.[j]
        
        while i < a.Length do
            total <- total + a.[i] * b.[i]
            i <- i + 1
        
        total

// ============================================================================
// Min/Max Operations
// ============================================================================

/// <summary>Find minimum element using SIMD.</summary>/// <param name="a">Input array.</param>/// <returns>Minimum value.</returns>/// <exception cref="System.ArgumentException">Thrown when array is empty.</exception>
let min (a: double[]) : double =
    if a.Length = 0 then
        invalidArg "a" "Cannot find min of empty array"
    
    if a.Length < 16 || not isHardwareAccelerated then
        Array.min a
    else
        let vecSize = vectorCountDouble
        let mutable vecMin = Vector(a, 0)
        let mutable i = vecSize
        
        // Vectorized min
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            vecMin <- Vector.Min(vecMin, va)
            i <- i + vecSize
        
        // Horizontal reduction
        let mutable minimum = vecMin.[0]
        for j = 1 to vecSize - 1 do
            if vecMin.[j] < minimum then
                minimum <- vecMin.[j]
        
        // Remainder
        while i < a.Length do
            if a.[i] < minimum then
                minimum <- a.[i]
            i <- i + 1
        
        minimum

/// <summary>Find maximum element using SIMD.</summary>/// <param name="a">Input array.</param>/// <returns>Maximum value.</returns>/// <exception cref="System.ArgumentException">Thrown when array is empty.</exception>
let max (a: double[]) : double =
    if a.Length = 0 then
        invalidArg "a" "Cannot find max of empty array"
    
    if a.Length < 16 || not isHardwareAccelerated then
        Array.max a
    else
        let vecSize = vectorCountDouble
        let mutable vecMax = Vector(a, 0)
        let mutable i = vecSize
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            vecMax <- Vector.Max(vecMax, va)
            i <- i + vecSize
        
        let mutable maximum = vecMax.[0]
        for j = 1 to vecSize - 1 do
            if vecMax.[j] > maximum then
                maximum <- vecMax.[j]
        
        while i < a.Length do
            if a.[i] > maximum then
                maximum <- a.[i]
            i <- i + 1
        
        maximum

// ============================================================================
// Absolute Sum
// ============================================================================

/// <summary>Sum of absolute values using SIMD.</summary>/// <param name="a">Input array.</param>/// <returns>Sum of absolute values.</returns>/// <remarks>
/// Useful for computing L1 norm.
/// </remarks>
let sumAbs (a: double[]) : double =
    if a.Length < 32 || not isHardwareAccelerated then
        let mutable s = 0.0
        for i = 0 to a.Length - 1 do
            s <- s + abs a.[i]
        s
    else
        let vecSize = vectorCountDouble
        let zero = Vector<double>(0.0)
        let mutable vecSum = Vector<double>(0.0)
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vabs = Vector.Max(va, zero - va)  // abs(x) = max(x, -x)
            vecSum <- vecSum + vabs
            i <- i + vecSize
        
        let mutable total = 0.0
        for j = 0 to vecSize - 1 do
            total <- total + vecSum.[j]
        
        while i < a.Length do
            total <- total + abs a.[i]
            i <- i + 1
        
        total

/// <summary>Calculate L2 norm (Euclidean length) using SIMD.</summary>/// <param name="a">Input array.</param>/// <returns>Square root of sum of squares.</returns>/// <example>
/// <code>
/// let a = [|3.0; 4.0|]
/// let result = norm a  // 5.0 (sqrt(3² + 4²))
/// </code>
/// </example>
let norm (a: double[]) : double =
    let mutable sumSquares = 0.0
    
    if a.Length >= 32 && isHardwareAccelerated then
        let vecSize = vectorCountDouble
        let mutable vecSum = Vector<double>(0.0)
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            vecSum <- vecSum + (va * va)
            i <- i + vecSize
        
        for j = 0 to vecSize - 1 do
            sumSquares <- sumSquares + vecSum.[j]
        
        while i < a.Length do
            sumSquares <- sumSquares + a.[i] * a.[i]
            i <- i + 1
    else
        for i = 0 to a.Length - 1 do
            sumSquares <- sumSquares + a.[i] * a.[i]
    
    Math.Sqrt sumSquares
