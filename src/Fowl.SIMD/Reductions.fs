/// <summary>
/// Fowl SIMD Reduction Operations
/// </summary>
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

/// <summary>
/// Sum all elements in array using SIMD.
/// </summary>
/// <param name="a">Input array.</param>
/// <returns>Sum of all elements.</returns>
/// <remarks>
/// Uses vectorized accumulation followed by horizontal reduction.
/// SIMD benefit is less than element-wise ops due to horizontal add overhead.
/// </remarks>
/// <example>
/// <code>
/// let a = [|1.0; 2.0; 3.0; 4.0|]
/// let result = sum a  // 10.0
/// </code>
/// </example>
let sum (a: double[]) : double =
    if a.Length < 32 || not isHardwareAccelerated then
        let mutable s = 0.0
        for i = 0 to a.Length - 1 do
            s <- s + a.[i]
        s
    else
        let vecSize = vectorCountDouble
        let mutable vecSum = Vector<double>(0.0)
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            vecSum <- vecSum + va
            i <- i + vecSize
        
        let mutable total = 0.0
        for j = 0 to vecSize - 1 do
            total <- total + vecSum.[j]
        
        while i < a.Length do
            total <- total + a.[i]
            i <- i + 1
        
        total

/// <summary>
/// Sum all elements in single array using SIMD.
/// </summary>
/// <param name="a">Input array.</param>
/// <returns>Sum of all elements.</returns>
let sumSingle (a: single[]) : single =
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

/// <summary>
/// Calculate mean using SIMD.
/// </summary>
/// <param name="a">Input array.</param>
/// <returns>Mean of all elements.</returns>
let mean (a: double[]) : double =
    if a.Length = 0 then
        invalidArg "a" "Cannot compute mean of empty array"
    sum a / double a.Length

/// <summary>
/// Calculate mean using SIMD (single precision).
/// </summary>
/// <param name="a">Input array.</param>
/// <returns>Mean of all elements.</returns>
let meanSingle (a: single[]) : single =
    if a.Length = 0 then
        invalidArg "a" "Cannot compute mean of empty array"
    sumSingle a / single a.Length

// ============================================================================
// Dot Product
// ============================================================================

/// <summary>
/// Compute dot product using SIMD.
/// </summary>
/// <param name="a">First array.</param>
/// <param name="b">Second array.</param>
/// <returns>Dot product of a and b.</returns>
/// <example>
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
        let mutable s = 0.0
        for i = 0 to a.Length - 1 do
            s <- s + a.[i] * b.[i]
        s
    else
        let vecSize = vectorCountDouble
        let mutable vecSum = Vector<double>(0.0)
        let mutable i = 0
        
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vb = Vector(b, i)
            vecSum <- vecSum + (va * vb)
            i <- i + vecSize
        
        let mutable total = 0.0
        for j = 0 to vecSize - 1 do
            total <- total + vecSum.[j]
        
        while i < a.Length do
            total <- total + a.[i] * b.[i]
            i <- i + 1
        
        total

/// <summary>
/// Compute dot product using SIMD (single precision).
/// </summary>
/// <param name="a">First array.</param>
/// <param name="b">Second array.</param>
/// <returns>Dot product of a and b.</returns>
let dotSingle (a: single[]) (b: single[]) : single =
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

/// <summary>
/// Find minimum value using SIMD.
/// </summary>
/// <param name="a">Input array.</param>
/// <returns>Minimum value.</returns>
let min (a: double[]) : double =
    if a.Length = 0 then
        invalidArg "a" "Cannot find min of empty array"
    
    Array.min a

/// <summary>
/// Find maximum value using SIMD.
/// </summary>
/// <param name="a">Input array.</param>
/// <returns>Maximum value.</returns>
let max (a: double[]) : double =
    if a.Length = 0 then
        invalidArg "a" "Cannot find max of empty array"
    
    Array.max a

// ============================================================================
// Norm and Distance
// ============================================================================

/// <summary>
/// Compute sum of absolute values (L1 norm).
/// </summary>
/// <param name="a">Input array.</param>
/// <returns>Sum of absolute values.</returns>
let sumAbs (a: double[]) : double =
    let mutable s = 0.0
    for i = 0 to a.Length - 1 do
        s <- s + abs a.[i]
    s

/// <summary>
/// Compute Euclidean norm (L2 norm).
/// </summary>
/// <param name="a">Input array.</param>
/// <returns>Euclidean norm.</returns>
let norm (a: double[]) : double =
    sqrt (dot a a)
