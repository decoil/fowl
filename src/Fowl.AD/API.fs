/// Fowl Algorithmic Differentiation - High-level API
/// Provides diff, grad, jacobian, hessian functions
module Fowl.AD.API

open System
open Fowl
open Fowl.Core.Types
open Fowl.AD.Types
open Fowl.AD.Core
open Fowl.AD.Ops

// ============================================================================
// Forward Mode Differentiation
// ============================================================================

/// <summary>Differentiate a scalar function at a point (forward mode).</summary>
/// <returns>Tuple of (value, derivative).</returns>
/// <example>
/// <code>
/// let f x = sin x
/// let value, deriv = diff' f (F 0.0)
/// // value = 0.0 (sin 0)
/// // deriv = 1.0 (cos 0)
/// </code>
/// </example>
let diff' (f: t -> t) (x: t) : t * t =
    Tag.reset()
    let x = make_forward x (F 1.0) (Tag.make())
    let y = f x
    primal y, tangent y

/// <summary>Differentiate a scalar function at a point (forward mode).</summary>
/// <returns>Just the derivative.</returns>
let diff f x = diff' f x |> snd

/// <summary>Evaluate function and its derivative at a float.</summary>
let diffF' (f: t -> t) (x: float) : float * float =
    let y, dy = diff' f (F x)
    unpack_flt y, unpack_flt dy

/// <summary>Derivative at a float.</summary>
let diffF f x = diffF' f x |> snd

// ============================================================================
// Reverse Mode (Gradient)
// ============================================================================

/// <summary>Gradient of a scalar function (reverse mode).</summary>
/// <returns>Tuple of (value, gradient).</returns>
let grad' (f: t -> t) (x: t) : t * t =
    Tag.reset()
    let x = make_reverse x
    reverse_reset x
    let y = f x
    reverse_push (F 1.0) y
    primal y, adjval x

/// <summary>Gradient of a scalar function (reverse mode).</summary>
/// <returns>Just the gradient.</returns>
let grad f x = grad' f x |> snd

/// <summary>Gradient at a float.</summary>
let gradF' (f: t -> t) (x: float) : float * float =
    let y, gy = grad' f (F x)
    unpack_flt y, unpack_flt gy

/// <summary>Gradient at a float.</summary>
let gradF f x = gradF' f x |> snd

// ============================================================================
// Vector-Jacobian Products
// ============================================================================

/// <summary>Jacobian-vector product (forward mode).</summary>
/// <param name="f">Function to differentiate.</param>
/// <param name="x">Input point.</param>
/// <param name="v">Vector to multiply with Jacobian.</param>
/// <returns>Tuple of (f(x), J*v).</returns>
let jacobianv' (f: t -> t) (x: t) (v: t) : t * t =
    Tag.reset()
    let x = make_forward x v (Tag.make())
    let y = f x
    primal y, tangent y

/// <summary>Jacobian-vector product (forward mode).</summary>
let jacobianv f x v = jacobianv' f x v |> snd

/// <summary>Vector-Jacobian product (reverse mode).</summary>
/// <param name="f">Function to differentiate.</param>
/// <param name="x">Input point.</param>
/// <param name="v">Vector to left-multiply with Jacobian.</param>
/// <returns>Tuple of (f(x), v^T*J).</returns>
let vjacobian' (f: t -> t) (x: t) (v: t) : t * t =
    Tag.reset()
    let x = make_reverse x
    reverse_reset x
    let y = f x
    reverse_push v y
    primal y, adjval x

/// <summary>Vector-Jacobian product (reverse mode).</summary>
let vjacobian f x v = vjacobian' f x v |> snd

// ============================================================================
// Higher-Order Derivatives
// ============================================================================

/// <summary>Hessian of a scalar function.</summary>
/// <remarks>Hessian is the Jacobian of the gradient.</remarks>
let hessian' (f: t -> t) (x: t) : t * t =
    // H = J(∇f)
    let grad_f x = grad f x
    jacobianv' grad_f x (F 1.0)

/// <summary>Hessian of a scalar function.</summary>
let hessian f x = hessian' f x |> snd

/// <summary>Hessian at a float.</summary>
let hessianF' (f: t -> t) (x: float) : float * float =
    let y, hy = hessian' f (F x)
    unpack_flt y, unpack_flt hy

/// <summary>Second derivative (curvature).</summary>
let curvature f x = hessian f x

/// <summary>Third derivative.</summary>
let jerk f x =
    // f''' = d/dx(f'')
    let f'' x = curvature f x
    diff f'' x

// ============================================================================
// Laplacian
// ============================================================================

/// <summary>Laplacian of a scalar function (trace of Hessian).</summary>
let laplacian (f: t -> t) (x: t) : t =
    // For scalar input, Laplacian = f''
    hessian f x

/// <summary>Laplacian at a float.</summary>
let laplacianF (f: t -> t) (x: float) : float =
    laplacian f (F x) |> unpack_flt

// ============================================================================
// Directional Derivatives
// ============================================================================

/// <summary>Directional derivative in direction v.</summary>
let directional' (f: t -> t) (x: t) (v: t) : t * t =
    jacobianv' f x v

/// <summary>Directional derivative.</summary>
let directional f x v = directional' f x v |> snd

// ============================================================================
// Convenience Functions
// ============================================================================

/// <summary>Evaluate function at float.</summary>
let evalF (f: t -> t) (x: float) : float =
    f (F x) |> unpack_flt

/// <summary>Evaluate function at array.</summary>
let evalArr (f: t -> t) (x: Ndarray<Float64, float>) : Ndarray<Float64, float> =
    f (Arr x) |> unpack_arr

/// <summary>Check if value is forward mode.</summary>
let isForward = function
    | DF _ -> true
    | _ -> false

/// <summary>Check if value is reverse mode.</summary>
let isReverse = function
    | DR _ -> true
    | _ -> false

/// <summary>Check if value is constant.</summary>
let isConstant = function
    | F _ | Arr _ -> true
    | _ -> false
