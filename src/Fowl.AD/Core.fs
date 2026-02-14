module Fowl.AD.Core

open Fowl
open Fowl.Core.Types
open Fowl.AD.Types

/// Get primal value (unwrap AD type)
let rec primal = function
    | DF (ap, _, _) -> ap
    | DR (ap, _, _, _, _, _) -> ap
    | ap -> ap

/// Get primal recursively
let rec primal' = function
    | DF (ap, _, _) -> primal' ap
    | DR (ap, _, _, _, _, _) -> primal' ap
    | ap -> ap

/// Get tangent (forward mode)
let tangent = function
    | DF (_, at, _) -> at
    | DR _ -> failwith "Cannot get tangent from reverse mode node"
    | ap -> F 0.0  // Constants have zero tangent

/// Get adjoint value (reverse mode)
let adjval = function
    | DF _ -> failwith "Cannot get adjoint from forward mode node"
    | DR (_, aa, _, _, _, _) -> !aa
    | ap -> F 0.0

/// Create zero value of same type
let rec zero = function
    | F _ -> F 0.0
    | Arr arr -> match Ndarray.zeros<Float64> (Ndarray.shape arr) with Ok z -> Arr z | Error _ -> F 0.0
    | DF (ap, _, _) -> ap |> primal' |> zero
    | DR (ap, _, _, _, _, _) -> ap |> primal' |> zero

/// Pack float into AD type
let pack_flt x = F x

/// Unpack AD type to float
let unpack_flt x =
    match primal x with
    | F x -> x
    | _ -> failwith "Expected scalar, got array"

/// Pack array into AD type
let pack_arr arr = Arr arr

/// Unpack AD type to array
let unpack_arr x =
    match primal x with
    | Arr arr -> arr
    | _ -> failwith "Expected array, got scalar"

/// Create forward mode node
let make_forward p t = DF (p, t, Tag.make())

/// Create reverse mode node
let make_reverse p =
    let noop _ ca t = t
    let noop_reg t = t
    DR (p, ref (zero p), (noop, noop_reg, ("Noop", [])), ref 0, Tag.make(), ref 0)

/// Add two adjoint values
let adjoint_add a b =
    match a, b with
    | F x, F y -> F (x + y)
    | Arr x, Arr y -> match Ndarray.add x y with Ok z -> Arr z | Error e -> failwithf "Add failed: %A" e
    | _ -> failwith "Cannot add different types"

/// Reset adjoint values in graph (before backward pass)
let reverse_reset x =
    let rec reset xs =
        match xs with
        | [] -> ()
        | x :: t -
            match x with
            | DR (_, aa, (_, register, _), af, _, tracker) -
003e
                aa := zero !aa
                af := !af + 1
                tracker := !tracker + 1
                if !af = 1 && !tracker = 1 then
                    reset (register t)
                else
                    reset t
            | _ -> reset t
    reset [x]

/// Push adjoint values backward through graph
let reverse_push v x =
    let rec push xs =
        match xs with
        | [] -> ()
        | (v, x) :: t -
            match x with
            | DR (_, aa, (adjoint, _, _), af, _, tracker) -
003e
                aa := adjoint_add !aa v
                af := !af - 1
                if !af = 0 && !tracker = 1 then
                    push (adjoint (primal x) aa t)
                else
                    tracker := !tracker - 1
                    push t
            | _ -> push t
    push [v, x]
