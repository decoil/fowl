module Fowl.AD.API

open Fowl
open Fowl.Core.Types
open Fowl.AD.Types
open Fowl.AD.Core
open Fowl.AD.Ops

/// Differentiate a scalar function at a point (forward mode)
let diff' (f: t -> t) (x: t) : t * t =
    let x = make_forward x (F 1.0) (Tag.make())
    let y = f x
    primal y, tangent y

/// Differentiate a scalar function at a point (forward mode)
let diff f x = diff' f x |> snd

/// Gradient of a scalar function (reverse mode)
let grad' (f: t -> t) (x: t) : t * t =
    let x = make_reverse x
    let y = f x
    reverse_push (F 1.0) y
    primal y, adjval x

/// Gradient of a scalar function (reverse mode)
let grad f x = grad' f x |> snd

/// Jacobian-vector product (forward mode)
let jacobianv' (f: t -> t) (x: t) (v: t) : t * t =
    let x = make_forward x v (Tag.make())
    let y = f x
    primal y, tangent y

/// Jacobian-vector product (forward mode)
let jacobianv f x v = jacobianv' f x v |> snd

/// Hessian of a scalar function
let hessian (f: t -> t) (x: t) : t =
    // Hessian = Jacobian of gradient
    grad (fun x -> grad f x) x

/// Laplacian of a scalar function (trace of Hessian)
let laplacian (f: t -> t) (x: t) : t =
    let h = hessian f x
    // Sum diagonal elements
    failwith "laplacian not fully implemented"

/// Convenience function: evaluate function at float
let eval_flt (f: t -> t) (x: float) : float =
    f (F x) |> unpack_flt

/// Convenience function: evaluate function at array
let eval_arr (f: t -> t) (x: Ndarray<Float64, float>) : Ndarray<Float64, float> =
    f (Arr x) |> unpack_arr
