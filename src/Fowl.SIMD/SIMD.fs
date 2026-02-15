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
/// open Fowl.SIMD.Core
/// open Fowl.SIMD.ElementWise
/// open Fowl.SIMD.Reductions
///
/// let a = [|1.0; 2.0; 3.0; 4.0|]
/// let b = [|5.0; 6.0; 7.0; 8.0|]
///
/// // SIMD-accelerated addition
/// let sum = ElementWise.add a b
///
/// // Check SIMD capabilities
/// printfn "%s" (formatSimdInfo (getSimdInfo()))
/// </code>
/// </remarks>
module Fowl.SIMD

// This module serves as a container for submodules.
// Access functionality through:
// - Fowl.SIMD.Core for hardware detection
// - Fowl.SIMD.ElementWise for element-wise operations
// - Fowl.SIMD.Reductions for reduction operations
// - Fowl.SIMD.Hardware for hardware-specific intrinsics
