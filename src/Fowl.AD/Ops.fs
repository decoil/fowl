/// Fowl Algorithmic Differentiation - Operations
/// Implements forward and reverse mode AD for mathematical operations
module Fowl.AD.Ops

open System
open Fowl
open Fowl.Core.Types
open Fowl.AD.Types
open Fowl.AD.Core

// ============================================================================
// Basic arithmetic operations on AD types
// ============================================================================

/// Add two AD values
let rec add a b =
    match a, b with
    | F x, F y -> F (x + y)
    | Arr x, Arr y -
003e
        match Ndarray.add x y with
        | Ok z -> Arr z
        | Error _ -> failwith "Array add failed"
    | DF (ap, at, ai), DF (bp, bt, bi) when ai = bi ->
        // Forward mode: (a+b)' = a' + b'
        let cp = add ap bp
        let ct = add at bt
        DF (cp, ct, ai)
    | DR (ap, _, _, _, ai, _), DR (bp, _, _, _, bi, _) when ai = bi ->
        // Reverse mode
        let cp = add ap bp
        let adjoint cp ca t =
            // Gradient of a+b w.r.t a is 1, w.r.t b is 1
            (ca, a) :: (ca, b) :: t
        let register t = a :: b :: t
        DR (cp, ref (zero cp), (adjoint, register, ("add", [a; b])), ref 0, ai, ref 0)
    | DF (ap, at, ai), bp ->
        // Mixed: a is forward, b is constant
        let cp = add ap bp
        DF (cp, at, ai)
    | ap, DF (bp, bt, bi) ->
        // Mixed: a is constant, b is forward
        let cp = add ap bp
        DF (cp, bt, bi)
    | DR (ap, _, _, _, ai, _), bp ->
        // Mixed: a is reverse, b is constant
        let cp = add ap bp
        let adjoint cp ca t = (ca, a) :: t
        let register t = a :: t
        DR (cp, ref (zero cp), (adjoint, register, ("add_c", [a])), ref 0, ai, ref 0)
    | ap, DR (bp, _, _, _, bi, _) ->
        // Mixed: a is constant, b is reverse
        let cp = add ap bp
        let adjoint cp ca t = (ca, b) :: t
        let register t = b :: t
        DR (cp, ref (zero cp), (adjoint, register, ("add_c", [b])), ref 0, bi, ref 0)
    | _ -> failwithf "Cannot add %A and %A" a b

/// Negate an AD value
let rec neg = function
    | F x -> F (-x)
    | Arr x -> Arr (Ndarray.map (fun v -> -v) x)
    | DF (ap, at, ai) -> DF (neg ap, neg at, ai)
    | DR (ap, aa, op, af, ai, tracker) ->
        let adjoint cp ca t = (neg !ca, DR (ap, aa, op, af, ai, tracker)) :: t
        DR (neg ap, aa, op, af, ai, tracker)

/// Subtract: a - b = a + (-b)
let sub a b = add a (neg b)

/// Multiply two AD values
let rec mul a b =
    match a, b with
    | F x, F y -> F (x * y)
    | Arr x, Arr y -
003e
        match Ndarray.mul x y with
        | Ok z -> Arr z
        | Error _ -> failwith "Array mul failed"
    | DF (ap, at, ai), DF (bp, bt, bi) when ai = bi ->
        // Forward mode: (a*b)' = a'*b + a*b'
        let cp = mul ap bp
        let term1 = mul at bp
        let term2 = mul ap bt
        let ct = add term1 term2
        DF (cp, ct, ai)
    | DR (ap, _, _, _, ai, _), DR (bp, _, _, _, bi, _) when ai = bi ->
        // Reverse mode: grad w.r.t a is b, w.r.t b is a
        let cp = mul ap bp
        let adjoint cp ca t =
            let abar = mul !ca bp
            let bbar = mul !ca ap
            (abar, a) :: (bbar, b) :: t
        let register t = a :: b :: t
        DR (cp, ref (zero cp), (adjoint, register, ("mul", [a; b])), ref 0, ai, ref 0)
    | DF (ap, at, ai), bp ->
        let cp = mul ap bp
        let ct = mul at bp
        DF (cp, ct, ai)
    | ap, DF (bp, bt, bi) ->
        let cp = mul ap bp
        let ct = mul ap bt
        DF (cp, ct, bi)
    | DR (ap, _, _, _, ai, _), bp ->
        let cp = mul ap bp
        let adjoint cp ca t = (mul !ca bp, a) :: t
        let register t = a :: t
        DR (cp, ref (zero cp), (adjoint, register, ("mul_c", [a])), ref 0, ai, ref 0)
    | ap, DR (bp, _, _, _, bi, _) ->
        let cp = mul ap bp
        let adjoint cp ca t = (mul !ca ap, b) :: t
        let register t = b :: t
        DR (cp, ref (zero cp), (adjoint, register, ("mul_c", [b])), ref 0, bi, ref 0)
    | _ -> failwithf "Cannot mul %A and %A" a b

/// Divide: a / b
let rec div a b =
    match a, b with
    | F x, F y -> F (x / y)
    | DF (ap, at, ai), DF (bp, bt, bi) when ai = bi ->
        // Forward mode: (a/b)' = (a'*b - a*b')/b^2
        let cp = div ap bp
        let num = sub (mul at bp) (mul ap bt)
        let den = mul bp bp
        let ct = div num den
        DF (cp, ct, ai)
    | DR (ap, _, _, _, ai, _), DR (bp, _, _, _, bi, _) when ai = bi ->
        let cp = div ap bp
        let adjoint cp ca t =
            // grad w.r.t a is 1/b
            let abar = div !ca bp
            // grad w.r.t b is -a/b^2
            let bbar = neg (mul !ca (div ap (mul bp bp)))
            (abar, a) :: (bbar, b) :: t
        let register t = a :: b :: t
        DR (cp, ref (zero cp), (adjoint, register, ("div", [a; b])), ref 0, ai, ref 0)
    | _ -> failwithf "div not fully implemented for %A / %A" a b

/// Power: a^b
let rec pow a b =
    match a, b with
    | F x, F y -> F (Math.Pow(x, y))
    | DF (ap, at, ai), DF (bp, bt, bi) when ai = bi ->
        // Forward mode: d(a^b) = a^b * (b/a * da + ln(a) * db)
        let cp = pow ap bp
        let term1 = mul bp (div at ap)
        let term2 = mul bt (log ap)
        let ct = mul cp (add term1 term2)
        DF (cp, ct, ai)
    | _ -> failwithf "pow not fully implemented for %A ^ %A" a b

// ============================================================================
// Elementary functions
// ============================================================================

/// Sine
let rec sin = function
    | F x -> F (Math.Sin x)
    | Arr x -> Arr (Ndarray.map Math.Sin x)
    | DF (ap, at, ai) ->
        // d(sin x) = cos x * dx
        let cp = sin ap
        let ct = mul at (cos ap)
        DF (cp, ct, ai)
    | DR (ap, aa, op, af, ai, tracker) ->
        let cp = sin ap
        let adjoint cp ca t = (mul !ca (cos (DR (ap, aa, op, af, ai, tracker))), DR (ap, aa, op, af, ai, tracker)) :: t
        let register t = DR (ap, aa, op, af, ai, tracker) :: t
        DR (cp, ref (zero cp), (adjoint, register, ("sin", [DR (ap, aa, op, af, ai, tracker)])), ref 0, ai, ref 0)

/// Cosine
and cos = function
    | F x -> F (Math.Cos x)
    | Arr x -> Arr (Ndarray.map Math.Cos x)
    | DF (ap, at, ai) ->
        // d(cos x) = -sin x * dx
        let cp = cos ap
        let ct = neg (mul at (sin ap))
        DF (cp, ct, ai)
    | DR (ap, aa, op, af, ai, tracker) ->
        let cp = cos ap
        let adjoint cp ca t = (neg (mul !ca (sin (DR (ap, aa, op, af, ai, tracker)))), DR (ap, aa, op, af, ai, tracker)) :: t
        let register t = DR (ap, aa, op, af, ai, tracker) :: t
        DR (cp, ref (zero cp), (adjoint, register, ("cos", [DR (ap, aa, op, af, ai, tracker)])), ref 0, ai, ref 0)

/// Exponential
and exp = function
    | F x -> F (Math.Exp x)
    | Arr x -> Arr (Ndarray.map Math.Exp x)
    | DF (ap, at, ai) ->
        // d(exp x) = exp x * dx
        let cp = exp ap
        let ct = mul at cp
        DF (cp, ct, ai)
    | DR (ap, aa, op, af, ai, tracker) ->
        let cp = exp ap
        let adjoint cp ca t = (mul !ca cp, DR (ap, aa, op, af, ai, tracker)) :: t
        let register t = DR (ap, aa, op, af, ai, tracker) :: t
        DR (cp, ref (zero cp), (adjoint, register, ("exp", [DR (ap, aa, op, af, ai, tracker)])), ref 0, ai, ref 0)

/// Natural logarithm
and log = function
    | F x -> F (Math.Log x)
    | Arr x -> Arr (Ndarray.map Math.Log x)
    | DF (ap, at, ai) ->
        // d(log x) = dx / x
        let cp = log ap
        let ct = div at ap
        DF (cp, ct, ai)
    | DR (ap, aa, op, af, ai, tracker) ->
        let cp = log ap
        let adjoint cp ca t = (div !ca (DR (ap, aa, op, af, ai, tracker)), DR (ap, aa, op, af, ai, tracker)) :: t
        let register t = DR (ap, aa, op, af, ai, tracker) :: t
        DR (cp, ref (zero cp), (adjoint, register, ("log", [DR (ap, aa, op, af, ai, tracker)])), ref 0, ai, ref 0)

/// Square root
let sqrt x = pow x (F 0.5)

/// Absolute value
let abs = function
    | F x -> F (Math.Abs x)
    | Arr x -> Arr (Ndarray.map Math.Abs x)
    | _ -> failwith "abs not implemented for AD modes"

// ============================================================================
// Trigonometric functions
// ============================================================================

let tan x = div (sin x) (cos x)
let cot x = div (cos x) (sin x)
let sec x = div (F 1.0) (cos x)
let csc x = div (F 1.0) (sin x)

// ============================================================================
// Hyperbolic functions
// ============================================================================

let sinh x = div (sub (exp x) (exp (neg x))) (F 2.0)
let cosh x = div (add (exp x) (exp (neg x))) (F 2.0)
let tanh x = div (sinh x) (cosh x)
