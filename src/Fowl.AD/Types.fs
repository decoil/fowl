/// Fowl Algorithmic Differentiation Module
/// Based on Owl's AD architecture (Wang & Zhao, 2023)
module Fowl.AD.Types

open Fowl
open Fowl.Core.Types

/// AD type - dual number representation
type t =
    | F of float                    // Scalar constant
    | Arr of Ndarray<Float64, float>  // Array constant  
    | DF of t * t * int             // Forward mode: (primal, tangent, tag)
    | DR of t * t ref * op * int ref * int * int ref  // Reverse mode

/// Operation record for reverse mode
and adjoint = t -> t ref -> (t * t) list -> (t * t) list
and register = t list -> t list
and label = string * t list
and op = adjoint * register * label

/// Module type for building operators
module type Operator =
    val label: string
    val ff_f: float -> t
    val ff_arr: Ndarray<Float64, float> -> t
    val df: t -> t -> t -> t  // (cp, ap, at) -> tangent result
    val dr: t -> t -> t ref -> t  // (a, cp, ca) -> adjoint result

/// Tag generator for perturbation confusion avoidance
module Tag =
    let mutable private counter = 0
    
    let make() =
        counter <- counter + 1
        counter
    
    let reset() = counter <- 0
