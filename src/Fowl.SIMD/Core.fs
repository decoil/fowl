/// <summary>
/// Fowl SIMD Core Module - Hardware Detection and Utilities
/// </summary>
/// <remarks>
/// Provides hardware capability detection and vector size information
/// for portable SIMD operations using System.Numerics.Vector.
/// </remarks>
module Fowl.SIMD.Core

open System
open System.Numerics

// ============================================================================
// Hardware Capability Detection
// ============================================================================

/// <summary>
/// Check if hardware-accelerated SIMD is available.
/// </summary>
/// <returns>true if Vector operations are hardware accelerated.</returns>
let isHardwareAccelerated : bool =
    Vector.IsHardwareAccelerated

/// <summary>
/// Get the number of elements that fit in a Vector<double>.
/// </summary>
/// <returns>The count of double-precision elements per vector.</returns>
/// <remarks>
/// Returns 2 on SSE2 (128-bit), 4 on AVX2 (256-bit), etc.
/// </remarks>
let vectorCountDouble : int =
    Vector<double>.Count

/// <summary>
/// Get the number of elements that fit in a Vector<single>.
/// </summary>
/// <returns>The count of single-precision elements per vector.</returns>
/// <remarks>
/// Returns 4 on SSE2 (128-bit), 8 on AVX2 (256-bit), etc.
/// </remarks>
let vectorCountSingle : int =
    Vector<single>.Count

/// <summary>
/// Get the size of a Vector in bits.
/// </summary>
/// <returns>Vector size in bits (typically 128, 256, or 512).</returns>
let vectorSizeBits : int =
    vectorCountDouble * 64  // Each double is 64 bits

/// <summary>
/// SIMD capability information.
/// </summary>
type SimdInfo = {
    IsHardwareAccelerated: bool
    VectorSizeBits: int
    VectorCountDouble: int
    VectorCountSingle: int
}

/// <summary>
/// Get current SIMD capability information.
/// </summary>
/// <returns>SimdInfo record with hardware details.</returns>
let getSimdInfo () : SimdInfo =
    {
        IsHardwareAccelerated = isHardwareAccelerated
        VectorSizeBits = vectorSizeBits
        VectorCountDouble = vectorCountDouble
        VectorCountSingle = vectorCountSingle
    }

/// <summary>
/// Format SIMD info for display.
/// </summary>
/// <param name="info">SIMD info record.</param>
/// <returns>Formatted string.</returns>
let formatSimdInfo (info: SimdInfo) : string =
    let accelStatus = if info.IsHardwareAccelerated then "Yes" else "No"
    $"SIMD Hardware Accelerated: {accelStatus}\n" +
    $"Vector Size: {info.VectorSizeBits} bits\n" +
    $"Doubles per Vector: {info.VectorCountDouble}\n" +
    $"Singles per Vector: {info.VectorCountSingle}"

// ============================================================================
// Thresholds and Configuration
// ============================================================================

/// <summary>
/// Minimum array size to benefit from SIMD overhead.
/// </summary>
/// <remarks>
/// Arrays smaller than this threshold use scalar operations
/// because SIMD setup overhead dominates for small arrays.
/// </remarks>
let simdThreshold : int = 16

/// <summary>
/// Check if array is large enough to benefit from SIMD.
/// </summary>
/// <param name="length">Array length.</param>
/// <returns>true if SIMD should be used.</returns>
let shouldUseSimd (length: int) : bool =
    isHardwareAccelerated && length >= simdThreshold

// ============================================================================
// Vector Creation Helpers
// ============================================================================

/// <summary>
/// Create a Vector<double> from array starting at index.
/// </summary>
/// <param name="source">Source array.</param>
/// <param name="index">Starting index.</param>
/// <returns>Vector containing elements from source.</returns>
let vectorFromArray (source: double[]) (index: int) : Vector<double> =
    Vector<double>(source, index)

/// <summary>
/// Create a Vector<single> from array starting at index.
/// </summary>
/// <param name="source">Source array.</param>
/// <param name="index">Starting index.</param>
/// <returns>Vector containing elements from source.</returns>
let vectorFromArraySingle (source: single[]) (index: int) : Vector<single> =
    Vector<single>(source, index)

/// <summary>
/// Create a Vector with all elements set to a value.
/// </summary>
/// <param name="value">Value to broadcast.</param>
/// <returns>Vector with all elements equal to value.</returns>
let vectorBroadcast (value: double) : Vector<double> =
    Vector<double>(value)

/// <summary>
/// Create a Vector with all elements set to a value (single precision).
/// </summary>
/// <param name="value">Value to broadcast.</param>
/// <returns>Vector with all elements equal to value.</returns>
let vectorBroadcastSingle (value: single) : Vector<single> =
    Vector<single>(value)

// ============================================================================
// Scalar Fallback Operations
// ============================================================================

/// <summary>
/// Scalar addition loop.
/// </summary>
/// <param name="a">First array.</param>
/// <param name="b">Second array.</param>
/// <param name="result">Result array (pre-allocated).</param>
let addScalar (a: double[]) (b: double[]) (result: double[]) : unit =
    for i = 0 to a.Length - 1 do
        result.[i] <- a.[i] + b.[i]

/// <summary>
/// Scalar subtraction loop.
/// </summary>
/// <param name="a">First array.</param>
/// <param name="b">Second array.</param>
/// <param name="result">Result array (pre-allocated).</param>
let subScalar (a: double[]) (b: double[]) (result: double[]) : unit =
    for i = 0 to a.Length - 1 do
        result.[i] <- a.[i] - b.[i]

/// <summary>
/// Scalar multiplication loop.
/// </summary>
/// <param name="a">First array.</param>
/// <param name="b">Second array.</param>
/// <param name="result">Result array (pre-allocated).</param>
let mulScalar (a: double[]) (b: double[]) (result: double[]) : unit =
    for i = 0 to a.Length - 1 do
        result.[i] <- a.[i] * b.[i]

/// <summary>
/// Scalar division loop.
/// </summary>
/// <param name="a">First array.</param>
/// <param name="b">Second array.</param>
/// <param name="result">Result array (pre-allocated).</param>
let divScalar (a: double[]) (b: double[]) (result: double[]) : unit =
    for i = 0 to a.Length - 1 do
        result.[i] <- a.[i] / b.[i]

/// <summary>
/// Scalar addition with constant.
/// </summary>
/// <param name="a">Input array.</param>
/// <param name="scalar">Scalar value.</param>
/// <param name="result">Result array (pre-allocated).</param>
let addScalarConst (a: double[]) (scalar: double) (result: double[]) : unit =
    for i = 0 to a.Length - 1 do
        result.[i] <- a.[i] + scalar

/// <summary>
/// Scalar multiplication with constant.
/// </summary>
/// <param name="a">Input array.</param>
/// <param name="scalar">Scalar value.</param>
/// <param name="result">Result array (pre-allocated).</param>
let mulScalarConst (a: double[]) (scalar: double) (result: double[]) : unit =
    for i = 0 to a.Length - 1 do
        result.[i] <- a.[i] * scalar

/// <summary>
/// Scalar negation.
/// </summary>
/// <param name="a">Input array.</param>
/// <param name="result">Result array (pre-allocated).</param>
let negateScalar (a: double[]) (result: double[]) : unit =
    for i = 0 to a.Length - 1 do
        result.[i] <- -a.[i]
