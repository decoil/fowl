# Owl Tutorial Chapter 4: Linear Algebra

## Module Structure
```ocaml
Linalg.S  (* single precision *)
Linalg.D  (* double precision *)
Linalg.C  (* complex single precision *)
Linalg.Z  (* complex double precision *)
Linalg.Generic (* generic, requires type info *)
```

## Core Operations

### Matrix Creation
```ocaml
Mat.empty 5 5        (* uninitialized *)
Mat.zeros 5 5        (* all zeros *)
Mat.ones 5 5         (* all ones *)
Mat.eye 5            (* identity *)
Mat.uniform 5 5      (* random uniform *)
Mat.gaussian 5 5     (* random Gaussian *)
Mat.sequential 5 5   (* 0,1,2,3... *)
Mat.magic 5          (* magic square *)
Mat.semidef 5        (* random semi-definite *)
```

### Gaussian Elimination & LU
```ocaml
(* PA = LU factorization *)
val lu : ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t * (int32, int32_elt) t
(* Returns (L, U, permutation_vector) *)
```

### Key Functions
```ocaml
val inv        : ('a, 'b) t -> ('a, 'b) t           (* inverse *)
val transpose  : ('a, 'b) t -> ('a, 'b) t           (* transpose *)
val rank       : ('a, 'b) t -> int                  (* matrix rank *)
val det        : ('a, 'b) t -> 'a                   (* determinant *)
val logdet     : ('a, 'b) t -> 'a                   (* log determinant *)
val null       : ('a, 'b) t -> ('a, 'b) t           (* null space basis *)
val linsolve   : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t  (* solve Ax=b *)

(* QR factorization *)
val qr : ?thin:bool -> ?pivot:bool -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t * (int32, int32_elt) t

(* Eigenvalues/eigenvectors *)
val eig     : ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t  (* returns (vectors, values) *)
val eigvals : ('a, 'b) t -> ('a, 'b) t              (* eigenvalues only *)

(* SVD *)
val svd     : ?thin:bool -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t * ('a, 'b) t
(* Returns (U, S, Vt) *)

(* Cholesky decomposition *)
val chol : ('a, 'b) t -> ('a, 'b) t

(* Condition number *)
val cond : ('a, 'b) t -> float

(* Norm *)
val norm : ('a, 'b) t -> float
```

## Internal: CBLAS/LAPACKE
Owl uses OpenBLAS which provides:
- **BLAS** (Basic Linear Algebra Subprograms) - low-level ops
- **LAPACK** (Linear Algebra Package) - higher-level factorizations

All performance-critical linear algebra goes through C interfaces.

## F# Design Notes

### Interop Strategy
F# options:
1. **MKL** via `System.Numerics` or P/Invoke
2. **OpenBLAS** via P/Invoke (same as Owl)
3. **Pure F#** for simpler operations, native for complex ones

### Module Structure
```fsharp
namespace Fowl

module Linalg =
    module S = ... // float32
    module D = ... // float64
    module C = ... // complex32
    module Z = ... // complex64
```

---
_Learned: 2026-02-13_
