/// <summary>Fowl Algorithmic Differentiation Module</summary>
/// <remarks>
/// Provides automatic differentiation in forward and reverse modes.
/// 
/// Forward mode is efficient for functions R -> R^n (single input, multiple outputs).
/// Reverse mode is efficient for functions R^n -> R (multiple inputs, single output).
/// 
/// Example usage:
/// <code>
/// open Fowl.AD
/// 
/// // Define function
/// let f x = sin (x * x)
/// 
/// // Forward mode derivative
/// let df = diff f (F 2.0)  // f'(2.0)
/// 
/// // Reverse mode gradient
/// let g x = x * x + sin x
/// let grad_g = grad g (F 1.0)  // ∇g(1.0)
/// </code>
/// </remarks>
module Fowl.AD

open Fowl.AD.Types
open Fowl.AD.Core
open Fowl.AD.Ops
open Fowl.AD.API

// ============================================================================
// Re-export Types
// ============================================================================

/// <summary>AD type for dual numbers.</summary>
type t = Types.t

/// <summary>Create forward mode node.</summary>
let make_forward = Core.make_forward

/// <summary>Create reverse mode node.</summary>
let make_reverse = Core.make_reverse

/// <summary>Get primal value.</summary>
let primal = Core.primal

/// <summary>Get primal recursively.</summary>
let primal' = Core.primal'

/// <summary>Get tangent (forward mode).</summary>
let tangent = Core.tangent

/// <summary>Get adjoint (reverse mode).</summary>
let adjval = Core.adjval

/// <summary>Pack float into AD type.</summary>
let pack_flt = Core.pack_flt

/// <summary>Unpack AD type to float.</summary>
let unpack_flt = Core.unpack_flt

/// <summary>Pack array into AD type.</summary>
let pack_arr = Core.pack_arr

/// <summary>Unpack AD type to array.</summary>
let unpack_arr = Core.unpack_arr

// ============================================================================
// Re-export Math Operations
// ============================================================================

/// <summary>Sine function with AD.</summary>
let sin = Ops.sin

/// <summary>Cosine function with AD.</summary>
let cos = Ops.cos

/// <summary>Exponential function with AD.</summary>
let exp = Ops.exp

/// <summary>Natural logarithm with AD.</summary>
let log = Ops.log

/// <summary>Square root with AD.</summary>
let sqrt = Ops.sqrt

/// <summary>Absolute value with AD.</summary>
let abs = Ops.abs

/// <summary>Negation with AD.</summary>
let neg = Ops.neg

/// <summary>Addition with AD.</summary>
let add = Ops.add

/// <summary>Subtraction with AD.</summary>
let sub = Ops.sub

/// <summary>Multiplication with AD.</summary>
let mul = Ops.mul

/// <summary>Division with AD.</summary>
let div = Ops.div

/// <summary>Power with AD.</summary>
let pow = Ops.pow

// ============================================================================
// Re-export High-level APIs
// ============================================================================

/// <summary>Differentiate function (forward mode).</summary>
let diff' = API.diff'

/// <summary>Differentiate function (forward mode).</summary>
let diff = API.diff

/// <summary>Differentiate function at float.</summary>
let diffF' = API.diffF'

/// <summary>Differentiate function at float.</summary>
let diffF = API.diffF

/// <summary>Gradient of function (reverse mode).</summary>
let grad' = API.grad'

/// <summary>Gradient of function (reverse mode).</summary>
let grad = API.grad

/// <summary>Gradient at float.</summary>
let gradF' = API.gradF'

/// <summary>Gradient at float.</summary>
let gradF = API.gradF

/// <summary>Jacobian-vector product (forward mode).</summary>
let jacobianv' = API.jacobianv'

/// <summary>Jacobian-vector product.</summary>
let jacobianv = API.jacobianv

/// <summary>Vector-Jacobian product (reverse mode).</summary>
let vjacobian' = API.vjacobian'

/// <summary>Vector-Jacobian product.</summary>
let vjacobian = API.vjacobian

/// <summary>Hessian of function.</summary>
let hessian' = API.hessian'

/// <summary>Hessian of function.</summary>
let hessian = API.hessian

/// <summary>Hessian at float.</summary>
let hessianF' = API.hessianF'

/// <summary>Curvature (second derivative).</summary>
let curvature = API.curvature

/// <summary>Jerk (third derivative).</summary>
let jerk = API.jerk

/// <summary>Laplacian of function.</summary>
let laplacian = API.laplacian

/// <summary>Laplacian at float.</summary>
let laplacianF = API.laplacianF

/// <summary>Directional derivative.</summary>
let directional' = API.directional'

/// <summary>Directional derivative.</summary>
let directional = API.directional

// ============================================================================
// Re-export Utilities
// ============================================================================

/// <summary>Evaluate function at float.</summary>
let evalF = API.evalF

/// <summary>Evaluate function at array.</summary>
let evalArr = API.evalArr

/// <summary>Check if forward mode.</summary>
let isForward = API.isForward

/// <summary>Check if reverse mode.</summary>
let isReverse = API.isReverse

/// <summary>Check if constant.</summary>
let isConstant = API.isConstant
