# Owl Tutorial Chapter 2: N-Dimensional Arrays

## Core Architecture

### Type Definition
```ocaml
type ('a, 'b) t = ('a, 'b, c_layout) Genarray.t
```
- Built on OCaml's Bigarray.Genarray
- `('a, 'b)` = (element_type, precision_kind)
- Fixed C-layout (row-major) only - simpler than supporting both C and Fortran layouts
- Maximum 16 dimensions (Bigarray limitation)

### Why C-Layout Only?
1. Mixing layouts is bug-prone (C: 0-indexed, FORTRAN: 1-indexed)
2. Different layouts need different implementations = code bloat
3. Owl prioritizes simple design over FORTRAN compatibility

## Key Function Groups

### 1. Creation Functions
```ocaml
val empty : ('a, 'b) kind -> int array -> ('a, 'b) t
val zeros : ('a, 'b) kind -> int array -> ('a, 'b) t
val ones  : ('a, 'b) kind -> int array -> ('a, 'b) t
val create: ('a, 'b) kind -> int array -> 'a -> ('a, 'b) t

val uniform  : ('a, 'b) kind -> ?a:'a -> ?b:'a -> int array -> ('a, 'b) t
val gaussian : ('a, 'b) kind -> ?mu:'a -> ?sigma:'a -> int array -> ('a, 'b) t
val bernoulli: ('a, 'b) kind -> ?p:float -> int array -> ('a, 'b) t

val sequential: ('a, 'b) kind -> ?a:'a -> ?step:'a -> int array -> ('a, 'b) t
val linspace  : ('a, 'b) kind -> 'a -> 'a -> int -> ('a, 'b) t
val logspace  : ('a, 'b) kind -> ?base:float -> 'a -> 'a -> int -> ('a, 'b) t

val init   : ('a, 'b) kind -> int array -> (int -> 'a) -> ('a, 'b) t
val init_nd: ('a, 'b) kind -> int array -> (int array -> 'a) -> ('a, 'b) t
```
**F# equivalent signatures:**
```fsharp
val empty: kind:('a, 'b) Kind -> shape:int array -> ('a, 'b) t
val zeros: kind:('a, 'b) Kind -> shape:int array -> ('a, 'b) t
val init: kind:('a, 'b) Kind -> shape:int array -> f:(int -> 'a) -> ('a, 'b) t
```

### 2. Properties Functions
```ocaml
val shape     : ('a, 'b) t -> int array
val num_dims  : ('a, 'b) t -> int
val nth_dim   : ('a, 'b) t -> int -> int
val numel     : ('a, 'b) t -> int
val nnz       : ('a, 'b) t -> int
val density   : ('a, 'b) t -> float
val size_in_bytes: ('a, 'b) t -> int
val same_shape: ('a, 'b) t -> ('a, 'b) t -> bool
val kind      : ('a, 'b) t -> ('a, 'b) kind
```

### 3. Map Functions
```ocaml
val map : ('a -> 'a) -> ('a, 'b) t -> ('a, 'b) t
val mapi: (int -> 'a -> 'a) -> ('a, 'b) t -> ('a, 'b) t
```
**Key insight:** Map is pure - always creates new array, never modifies in place.

### 4. Fold Functions
```ocaml
val fold : ?axis:int -> ('a -> 'a -> 'a) -> 'a -> ('a, 'b) t -> ('a, 'b) t
val foldi: ?axis:int -> (int -> 'a -> 'a -> 'a) -> 'a -> ('a, 'b) t -> ('a, 'b) t
```
- Optional `axis` parameter - if not specified, flattens first
- Returns ndarray (not scalar) - use `sum'` vs `sum` pattern
- Example: `let sum' ?axis x = Arr.fold ?axis ( +. ) 0. x`

### 5. Scan Functions
```ocaml
val scan : ?axis:int -> ('a -> 'a -> 'a) -> ('a, 'b) t -> ('a, 'b) t
val scani: ?axis:int -> (int -> 'a -> 'a -> 'a) -> ('a, 'b) t -> ('a, 'b) t
```
- Like map + fold combined
- Accumulates but preserves shape (think CDF from PDF)
- Functions: cumsum, cumprod, cummin, cummax

### 6. Comparison Functions
Four groups:
1. `equal`, `less`, etc. → returns `bool`
2. `elt_equal`, `elt_less`, etc. → returns 0-1 ndarray
3. `equal_scalar`, `less_scalar`, etc. → ndarray vs scalar, returns `bool`
4. `elt_equal_scalar`, etc. → ndarray vs scalar, returns 0-1 ndarray

Example idiom:
```ocaml
(* Keep elements > 0.5, zero the rest *)
let z = Arr.((x >.$ 0.5) * x)
```

### 7. Manipulation Functions
```ocaml
val tile      : ('a, 'b) t -> int array -> ('a, 'b) t
val repeat    : ('a, 'b) t -> int array -> ('a, 'b) t
val expand    : ('a, 'b) t -> int -> ('a, 'b) t
val squeeze   : ?axis:int array -> ('a, 'b) t -> ('a, 'b) t
val pad       : ?v:'a -> int list list -> ('a, 'b) t -> ('a, 'b) t
val concatenate: ?axis:int -> ('a, 'b) t array -> ('a, 'b) t
val split     : ?axis:int -> int array -> ('a, 'b) t -> ('a, 'b) t array
val sort      : ('a, 'b) t -> unit  (* in-place! *)
```

### 8. Serialisation
```ocaml
val save: out:string -> ('a, 'b) t -> unit
val load: ('a, 'b) kind -> string -> ('a, 'b) t

(* NPY format for Python interop *)
val save_npy: out:string -> ('a, 'b) t -> unit
val load_npy: string -> ('a, 'b) t
```

## Tensors vs Ndarrays

**Key distinction:**
- **Ndarray**: Data structure for storing n-dimensional data
- **Tensor**: Mathematical object that transforms in specific ways under coordinate changes

Tensor contraction:
```ocaml
val contract1: (int * int) array -> ('a, 'b) t -> ('a, 'b) t
val contract2: (int * int) array -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
```

Matrix multiplication is a special case:
```ocaml
let z = Mat.dot x y
let z = Arr.contract2 [|(1,0)|] x y  (* equivalent *)
```

## F# Design Decisions for Fowl

### Type System
```fsharp
// Option 1: SRTP (Static Member Constraints)
type Ndarray<'T> = abstract Shape: int array

// Option 2: Discriminated Union with primitives
type Kind =
    | Float32
    | Float64
    | Complex32
    | Complex64
    | Int8
    | Int16
    | Int32
    | Int64

type Ndarray = internal | Ndarray of underlying: obj * kind: Kind * shape: int array
```

### Performance Strategy
- Core operations in native code (C/F# inline IL)
- Use `Span<'T>` and `Memory<'T>` for zero-allocation slicing
- P/Invoke to OpenBLAS for linear algebra
- Consider `System.Numerics.Tensors` as foundation?

### Questions
1. Should Fowl use `System.Numerics.Tensors.Tensor<'T>` as foundation?
2. How to handle the Kind GADT pattern in F#? (SRTP vs DU vs interfaces)
3. Native interop strategy: P/Invoke vs C++/CLI vs native AOT?

---
_Learned: 2026-02-13_
