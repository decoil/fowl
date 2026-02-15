/// <summary>
/// Fowl SIMD Module - Hardware-Accelerated Numerical Operations
/// </summary>
/// <remarks>
/// Provides portable SIMD operations using System.Numerics.Vector.
/// Automatically uses hardware acceleration when available, falls back to scalar otherwise.
///
/// Supports both double (Float64) and single (Float32) precision operations.
///
/// Example usage:
/// <code>
/// open Fowl.SIMD
///
/// let a = [|1.0; 2.0; 3.0; 4.0|]
/// let b = [|5.0; 6.0; 7.0; 8.0|]
///
/// // SIMD-accelerated addition
/// let sum = ElementWise.add a b
///
/// // SIMD-accelerated dot product
/// let dot = Reductions.dot a b
///
/// // Check SIMD capabilities
/// printfn "%s" (Core.formatSimdInfo (Core.getSimdInfo()))
/// </code>
/// </remarks>
module Fowl.SIMD

// Re-export Core module values
let isHardwareAccelerated = Core.isHardwareAccelerated
let vectorCountDouble = Core.vectorCountDouble
let vectorCountSingle = Core.vectorCountSingle
let vectorSizeBits = Core.vectorSizeBits
let getSimdInfo = Core.getSimdInfo
let formatSimdInfo = Core.formatSimdInfo
let simdThreshold = Core.simdThreshold
let shouldUseSimd = Core.shouldUseSimd

// Re-export ElementWise module (double precision)
let add = ElementWise.add
let sub = ElementWise.sub
let mul = ElementWise.mul
let div = ElementWise.div
let negate = ElementWise.negate
let addScalar = ElementWise.addScalar
let mulScalar = ElementWise.mulScalar
let addInPlace = ElementWise.addInPlace
let mulInPlace = ElementWise.mulInPlace

// Re-export ElementWise module (single precision)
let addSingle = ElementWise.addSingle
let subSingle = ElementWise.subSingle
let mulSingle = ElementWise.mulSingle
let divSingle = ElementWise.divSingle

// Re-export Reductions module (double precision)
let sum = Reductions.sum
let mean = Reductions.mean
let dot = Reductions.dot
let min = Reductions.min
let max = Reductions.max
let sumAbs = Reductions.sumAbs
let norm = Reductions.norm

// Re-export Reductions module (single precision)
let sumSingle = Reductions.sumSingle
let meanSingle = Reductions.meanSingle
let dotSingle = Reductions.dotSingle

/// <summary>
/// SIMD information record.
/// </summary>
type SimdInfo = Core.SimdInfo

/// <summary>
/// Print SIMD capabilities to console.
/// </summary>
let printSimdInfo () : unit =
    Core.getSimdInfo ()
    |> Core.formatSimdInfo
    |> printfn "%s"

/// <summary>
/// Quick performance test comparing SIMD vs scalar.
/// </summary>
/// <param name="size">Array size for test.</param>
let performanceTest (size: int) : unit =
    printfn "\n=== SIMD Performance Test ==="
    printfn "Array size: %d elements" size
    printSimdInfo ()

    let rng = System.Random()
    let a = Array.init size (fun _ -> rng.NextDouble())
    let b = Array.init size (fun _ -> rng.NextDouble())

    // Warmup
    ElementWise.add a b |> ignore

    // SIMD version
    let stopwatch = System.Diagnostics.Stopwatch()
    stopwatch.Start()
    for _ = 1 to 100 do
        ElementWise.add a b |> ignore
    stopwatch.Stop()
    let simdTime = stopwatch.ElapsedMilliseconds

    // Scalar version (force by using small array threshold)
    let result = Array.zeroCreate size
    stopwatch.Restart()
    for _ = 1 to 100 do
        Core.addScalar a b result
    stopwatch.Stop()
    let scalarTime = stopwatch.ElapsedMilliseconds

    printfn "\nAddition:"
    printfn "  SIMD:   %d ms" simdTime
    printfn "  Scalar: %d ms" scalarTime
    printfn "  Speedup: %.2fx" (float scalarTime / float simdTime)

    // Dot product test
    stopwatch.Restart()
    for _ = 1 to 100 do
        Reductions.dot a b |> ignore
    stopwatch.Stop()
    let dotSimdTime = stopwatch.ElapsedMilliseconds

    let mutable s = 0.0
    stopwatch.Restart()
    for _ = 1 to 100 do
        for i = 0 to size - 1 do
            s <- s + a.[i] * b.[i]
    stopwatch.Stop()
    let dotScalarTime = stopwatch.ElapsedMilliseconds

    printfn "\nDot Product:"
    printfn "  SIMD:   %d ms" dotSimdTime
    printfn "  Scalar: %d ms" dotScalarTime
    if dotSimdTime > 0 then
        printfn "  Speedup: %.2fx" (float dotScalarTime / float dotSimdTime)
    printfn ""
