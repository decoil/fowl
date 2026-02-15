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
namespace Fowl.SIMD

// This namespace contains submodules:
// - Core for hardware detection
// - ElementWise for element-wise operations  
// - Reductions for reduction operations
// - Hardware for hardware-specific intrinsics
