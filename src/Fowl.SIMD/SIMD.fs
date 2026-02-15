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

// Re-export Core module values with fully qualified names
let isHardwareAccelerated = Fowl.SIMD.Core.isHardwareAccelerated
let vectorCountDouble = Fowl.SIMD.Core.vectorCountDouble
let vectorCountSingle = Fowl.SIMD.Core.vectorCountSingle
let vectorSizeBits = Fowl.SIMD.Core.vectorSizeBits
let getSimdInfo = Fowl.SIMD.Core.getSimdInfo
let formatSimdInfo = Fowl.SIMD.Core.formatSimdInfo
let simdThreshold = Fowl.SIMD.Core.simdThreshold
let shouldUseSimd = Fowl.SIMD.Core.shouldUseSimd

// Re-export ElementWise module (double precision)
let add = Fowl.SIMD.ElementWise.add
let sub = Fowl.SIMD.ElementWise.sub
let mul = Fowl.SIMD.ElementWise.mul
let div = Fowl.SIMD.ElementWise.div
let negate = Fowl.SIMD.ElementWise.negate
let addScalar = Fowl.SIMD.ElementWise.addScalar
let mulScalar = Fowl.SIMD.ElementWise.mulScalar
let addInPlace = Fowl.SIMD.ElementWise.addInPlace
let mulInPlace = Fowl.SIMD.ElementWise.mulInPlace

// Re-export ElementWise module (single precision)
let addSingle = Fowl.SIMD.ElementWise.addSingle
let subSingle = Fowl.SIMD.ElementWise.subSingle
let mulSingle = Fowl.SIMD.ElementWise.mulSingle
let divSingle = Fowl.SIMD.ElementWise.divSingle

// Re-export Reductions module (double precision)
let sum = Fowl.SIMD.Reductions.sum
let mean = Fowl.SIMD.Reductions.mean
let dot = Fowl.SIMD.Reductions.dot
let min = Fowl.SIMD.Reductions.min
let max = Fowl.SIMD.Reductions.max
let sumAbs = Fowl.SIMD.Reductions.sumAbs
let norm = Fowl.SIMD.Reductions.norm

// Re-export Reductions module (single precision)
let sumSingle = Fowl.SIMD.Reductions.sumSingle
let meanSingle = Fowl.SIMD.Reductions.meanSingle
let dotSingle = Fowl.SIMD.Reductions.dotSingle

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
