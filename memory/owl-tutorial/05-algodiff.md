# Owl Tutorial Chapter 5: Algorithmic Differentiation

## Three Differentiation Methods

### 1. Numerical Differentiation
```ocaml
let diff f x = (f (x +. eps) -. f x) /. eps
```
- **Pros:** Simple, treats f as black box
- **Cons:** Truncation error, round-off error, slow

### 2. Symbolic Differentiation
- **Pros:** Exact, no numerical error
- **Cons:** Expression explosion, memory intensive, slow

### 3. Algorithmic Differentiation (AD)
- **Pros:** Exact, efficient, no expression explosion
- **Cons:** More complex implementation

## How AD Works

### Forward Mode (Tangent Mode)
- Computes **primal** and **tangent** values simultaneously
- Notation: `ṽ = ∂vᵢ/∂x₀` (derivative of intermediate w.r.t. input)
- Best when: **outputs >> inputs**

### Reverse Mode (Adjoint Mode)
- Two passes: forward (compute primals), backward (compute adjoints)
- Notation: `v̄ᵢ = ∂y/∂vᵢ` (derivative of output w.r.t. intermediate)
- Best when: **inputs >> outputs** (e.g., neural networks)
- **Key insight:** Reverse mode gives ALL gradients in ONE backward pass!

## Simple Implementations

### Forward Mode Data Type
```ocaml
type df = {
  mutable p: float;  (* primal *)
  mutable t: float   (* tangent *)
}
```

### Reverse Mode Data Type
```ocaml
type dr = {
  mutable p: float;
  mutable a: float ref;
  mutable adj_fun: float ref -> (float * dr) list -> (float * dr) list
}
```

### Unified Type
```ocaml
type t =
  | DF of float * float
  | DR of float * float ref * adjoint

and adjoint = float ref -> (float * t) list -> (float * t) list
```

## Module-Based Design Pattern

```ocaml
module type Unary = sig
  val ff : float -> float              (* primal function *)
  val df : float -> float -> float     (* forward derivative *)
  val dr : float -> float ref -> float (* reverse derivative *)
end

module type Binary = sig
  val ff : float -> float -> float
  val df : float -> float -> float -> float -> float
  val dr : float -> float -> float ref -> float * float
end

(* Example: sin operation *)
module Sin = struct
  let ff = sin
  let df p t = (cos p) *. t
  let dr p a = !a *. (cos p)
end

(* Template to build operations *)
let unary_op (module U: Unary) = fun x ->
  match x with
  | DF (p, t) -> DF (U.ff p, U.df p t)
  | DR (p, _, _) ->
    let adjfun' a stack = (U.dr p a, x) :: stack in
    DR (U.ff p, ref 0., adjfun')
```

## Owl's AD API

### High-Level Functions
```ocaml
(* Derivative: scalar -> scalar *)
val diff : (t -> t) -> t -> t

(* Gradient: vector -> scalar *)
val grad : (t -> t) -> t -> t

(* Jacobian: vector -> vector *)
val jacobian : (t -> t) -> t -> t

(* Hessian: vector -> scalar (second derivatives) *)
val hessian : (t -> t) -> t -> t
```

### Wrapping/Unwrapping
```ocaml
val F     : float -> t              (* wrap float *)
val Arr   : ndarray -> t            (* wrap array *)
val unpack_flt : t -> float         (* unwrap float *)
val unpack_arr : t -> ndarray       (* unwrap array *)
```

### Example Usage
```ocaml
open Algodiff.D

let f x =
  let x0 = Mat.get x 0 0 in
  let x1 = Mat.get x 0 1 in
  Maths.((F 1.) / (F 1. + exp (sin x0 + x0 * x1)))

let input = Arr (Dense.Matrix.D.ones 1 2)
let gradient = grad f input |> unpack_arr
```

## F# Design for AD

### Type Definition
```fsharp
type AD =
    | DF of primal: float * tangent: float
    | DR of primal: float * adjoint: float ref * adjointFunc

and adjointFunc = float ref -> (float * AD) list -> (float * AD) list
```

### Computation Expression Builder
```fsharp
type ADBuilder() =
    member _.Bind(x, f) = ...
    member _.Return(x) = ...

let ad = ADBuilder()

// Usage
let f x = ad {
    let! y = sin x
    let! z = exp y
    return z
}
```

### Module Structure
```fsharp
namespace Fowl

module Algodiff =
    module D = ... // double precision AD
    module S = ... // single precision AD

    // High-level API
    val diff : (AD -> AD) -> float -> float
    val grad : (AD -> AD) -> Vector -> Vector
    val jacobian : (AD -> AD) -> Matrix -> Matrix
```

---
_Learned: 2026-02-13_
