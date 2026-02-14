# Owl Tutorial Chapter 1: Introduction

## Summary
Owl is OCaml's scientific computing library. Uses functional programming for numerical computing - proving FP is viable for high-performance numerical work.

## Key Architecture Points

### Three Ndarray Types
1. **Standard Ndarray** - C-based, uses OpenBLAS, high performance
2. **Base Ndarray** - Pure OCaml implementation, slower but self-contained
3. **CGraph-Ndarray** - Computation graph support (like TensorFlow v1), symbolic style

### Dependencies
- **OpenBLAS** - BLAS + LAPACK implementation
- CBLAS/LAPACKE interfaces for linear algebra
- Performance-critical code in C, wrapped by OCaml

## OCaml Patterns Observed

### Pipeline Operator
```ocaml
let make_network input_shape =
  input input_shape
  |> lambda (fun x -> Maths.(x / F 256.))
  |> conv2d [|5;5;1;32|] [|1;1] ~act_typ:Activation.Relu
  |> max_pool2d [|2;2|] [|2;2]
  ...
```
**F# equivalent:** Same! `|>` works identically in F#.

### Labeled/Optional Arguments
```ocaml
conv2d [|5;5;1;32|] [|1;1] ~act_typ:Activation.Relu
fully_connected 1024 ~act_typ:Activation.Relu
```
**F# equivalent:** Named parameters with `?` for optional:
```fsharp
conv2d [|5;5;1;32|] [|1;1] ?actType=Activation.Relu
```

### Module Structure
```ocaml
open Owl
open Neural.S
open Neural.S.Graph
open Neural.S.Algodiff
```
**F# equivalent:** `open` works the same way. Submodules accessed with `.`

### Local Opens
```ocaml
Maths.(x / F 256.)
```
**F# equivalent:** No direct equivalent. Would use:
```fsharp
Maths.divide x (Maths.F 256.)
// or open Maths locally
```

## F# Design Notes

1. **Pipeline-first design** - Owl uses `|>` heavily, which fits F# perfectly
2. **Module organization** - Hierarchical modules translate well to F# namespaces
3. **Type inference** - Both OCaml and F# have strong inference
4. **Performance strategy** - Core in C/interop, API in functional language

## Questions for Later
- How does Owl handle memory layout for Ndarray?
- What's the interop story between OCaml and C?
- How to structure F# interop with native libraries?

---
_Learned: 2026-02-13_
