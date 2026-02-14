# Architecture of Advanced Numerical Analysis Systems - Notes

**Book:** Architecture of Advanced Numerical Analysis Systems  
**Authors:** Liang Wang, Jianxin Zhao (Owl creators)  
**Date:** 2026-02-14

---

## Chapter 1: Introduction

### Why OCaml for Numerical Computing?
- Before Owl: Fragmented ecosystem (Lacaml, Oml, Pareto)
- Inconsistent data representation, boilerplate code required
- OCaml advantages: Concise functional code, type safety, performance

### System Scale
- **269K lines of OCaml**
- **142K lines of C**
- **6500+ functions**

### Architecture Overview
```
Numerical Subsystem:
├── Core modules (Ndarray - dense/sparse)
├── Classic analytics (Maths, Stats, Linalg, ODE, Signal)
├── Advanced analytics (AD, Optim, Regression, NN, NLP)
└── Service modules (Zoo, Actor)
```

### Key Design Decisions
1. **Bigarray foundation**: Max 16 dims, C-layout (row-major)
2. **Four types**: Float32, Float64, Complex32, Complex64
3. **Dense + Sparse**: CSR format for sparse
4. **Modular design**: Functors for flexibility

---

## Chapter 2: Core Optimizations

### Ndarray Implementation
```ocaml
type ('a, 'b) t = ('a, 'b, c_layout) Genarray.t

type ('a, 'b) kind =
  | Float32 : (float, float32_elt) kind
  | Float64 : (float, float64_elt) kind
  | Complex32 : (Complex.t, complex32_elt) kind
  | Complex64 : (Complex.t, complex64_elt) kind
```

### OCaml-to-C Interface
- Core computations in C for performance
- OCaml wrappers using `external` declarations
- C macros for code generation across types

Example macro template:
```c
#ifdef FUN4
CAMLprim value FUN4(value vN, value vX, value vY) {
  CAMLparam3(vN, vX, vY);
  // ... conversion and computation
  caml_release_runtime_system();  // Allow parallel threads
  // ... loop computation
  caml_acquire_runtime_system();
  CAMLreturn(Val_unit);
}
#endif
```

### CPU Parallelism

#### Instruction-Level Parallelism (ILP)
- Superscalar architecture executes multiple instructions/cycle
- Hardware automatically discovers parallelism

#### SIMD Vectorization
- **Intel**: MMX → SSE → AVX → AVX2 → AVX512
- **ARM**: NEON
- AVX2: Process 16 16-bit numbers per cycle

Example AVX2:
```c
#include <immintrin.h>
__m256i a = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
__m256i b = _mm256_set_epi32(10, 20, 30, 40, 50, 60, 70, 80);
__m256i c = _mm256_add_epi32(a, b);
```

#### Multicore with OpenMP
```c
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++) {
  NUMBER x = *(start_x + i);
  *(start_y + i) = (MAPFN(x));
}
```

### Memory Hierarchy

| Level | Size | Speed | Example (Intel i9-9900K) |
|-------|------|-------|-------------------------|
| L1 | 64KB/core | ~4 cycles | Per core |
| L2 | 256KB/core | ~10 cycles | Per core |
| L3 | 16MB | Slower | Shared |
| Main | GB+ | 100+ cycles | DRAM |

### Cache Optimization Techniques

#### Cache Size Detection
```c
void query_cache_sizes(int* l1p, int* l2p, int* l3p) {
  // Uses CPUID instruction (x86/x64)
  // Falls back to conservative defaults on other arches
}
```

#### Prefetching
- **Hardware**: CPU monitors patterns, predicts next accesses
- **Software**: `__builtin_prefetch(addr, rw, locality)`

```c
for (j=0; j < m; ++j) {
  __builtin_prefetch(&x[i+1][j]);
  c += x[i][j];
}
```

#### NUMA (Non-Uniform Memory Access)
- Each processor has local memory
- Reduces contention for shared memory
- Critical for memory-intensive applications

### Optimization Techniques

#### Loop Tiling (Matrix Multiplication)
Cut large matrices into L1-cache-sized blocks:
```c
for (i = 0; i < N; i += E)
  for (j = 0; j < N; j += E)
    for (k = 0; k < N; k += E)
      // Process E×E block
```

#### Loop Fusion
Merge consecutive loops to reuse cached data:
```c
// Before: two loops, x loaded twice
// After: one loop, x stays in cache
for (i = 0; i < N; i++) {
    x[i] = buf[i];
    y[i] = i * x[i] + bias;
}
```

#### Loop Unrolling
Reduce branch mispredictions:
```c
for(i=0; i<n; i+=8) {
    a[i] = b[i] + 1;
    a[i+1] = b[i+1] + 1;
    // ... up to i+7
}
```

#### Non-Temporal Writes
Bypass cache when data won't be reused:
```c
#include <ammintrin.h>
void _mm_stream_sd(double *p, __m128d a);
```

### Convolution Optimization
- Implemented via im2col algorithm
- Previously used Eigen (C++), now handcrafted C
- Three variants: Conv, ConvBackwardKernel, ConvBackwardInput

---

## Chapter 3: Algorithmic Differentiation

### Three Types of Differentiation
1. **Numerical**: Finite differences (error-prone)
2. **Symbolic**: Formula manipulation (limited expressiveness)
3. **Algorithmic**: Chain rule on computation graph (best of both)

### AD Types

#### Forward Mode (DF)
```ocaml
type t = 
  | DF of t * t * int  (* primal * tangent * tag *)
```
- Computes primal and derivative simultaneously
- Good for: f: ℝ → ℝⁿ (few inputs, many outputs)
- Used via: `make_forward p t i`

#### Reverse Mode (DR)
```ocaml
type t = 
  | DR of t * t ref * (adjoint * register * label) * 
          int ref * int * int ref
  (* primal * adjoint * callbacks * fanout * tag * tracker *)
```
- Forward pass builds graph, backward pass computes gradients
- Good for: f: ℝⁿ → ℝ (many inputs, single output)
- Used via: `make_reverse p i` + `reverse_prop v x`

### Operator Builder Pattern

Templates for code reuse:
- **SISO**: Single Input Single Output (sin, cos, exp)
- **SIPO**: Single Input Pair Output (qr decomposition)
- **SITO**: Single Input Three Output (svd)
- **SIAO**: Single Input Array Output (split)
- **PISO**: Pair Input Single Output (add, mul)
- **AISO**: Array Input Single Output (concatenate)

Example SISO builder:
```ocaml
let build_siso (module S : Siso) =
  let rec f a =
    let ff = function
      | F a -> S.ff_f a
      | Arr a -> S.ff_arr a
    in
    let r a = 
      let adjoint cp ca t = (S.dr (primal a) cp ca, a) :: t in
      let register t = a :: t in
      let label = S.label, [a] in
      adjoint, register, label
    in
    op_siso ~ff ~fd:(f a) ~df:S.df ~r a
  in
  f
```

Usage:
```ocaml
let sin = build_siso (
  module struct
    let label = "sin"
    let ff_f a = F A.Scalar.(sin a)
    let ff_arr a = Arr A.(sin a)
    let df _cp ap at = at * cos ap
    let dr a _cp ca = !ca * cos a
  end : Siso)
```

### API Hierarchy

#### Low-Level APIs
```ocaml
(* Forward mode *)
let make_forward p t i = DF (p, t, i)
let tangent = function DF (_, t, _) -> t | _ -> failwith ""

(* Reverse mode *)
let make_reverse p i = 
  DR (p, ref zero, (noop, noop, ("Noop", [])), ref 0, i, ref 0)
  
let reverse_prop v x =
  reverse_reset x;
  reverse_push v x
```

#### High-Level APIs
```ocaml
let diff f x = (* scalar derivative *)
  let x = make_forward x (pack_flt 1.) (tag ()) in
  tangent (f x)

let grad f x = (* vector gradient *)
  let x = make_reverse x (tag ()) in
  let y = f x in
  reverse_push (pack_flt 1.) y;
  adjval x

let jacobian f x = (* matrix of partials *)
  (* ... forward mode with basis vectors *)

let hessian f x = (* second derivatives *)
  jacobian (grad f) x
```

### Advanced Implementation

#### Perturbation Confusion
Problem: Nested derivatives confuse which tangent goes with which variable.

Example: f(x) = x + d(y)/dy where y = x
- Without tag: dx/dy incorrectly becomes 1
- With tag: Each derivative gets unique tag

Solution: `tag : int` field distinguishes nested AD calls.

#### Lazy Evaluation
Cache operator construction for large graphs:
```ocaml
let _sin_ad = lazy (Builder.build_siso (module Sin))
let sin = Lazy.force _sin_ad  (* Evaluated once *)
```

#### Graph Visualization
```ocaml
let _traverse_trace x =
  let nodes = Hashtbl.create 512 in
  let rec push tlist = 
    (* DFS traversal building node table *)
  in
  push [x];
  nodes  (* Convert to DOT format for Graphviz *)
```

### Functor Architecture

```ocaml
(* 1. Primal operations signature *)
module type Ndarray_Sig = sig
  type elt
  type arr
  val sin : arr -> arr
  (* ... all ndarray operations *)
end

(* 2. Core module with types *)
module Core_Make(A : Ndarray_Sig) = struct
  type t = 
    | F of A.elt
    | Arr of A.arr
    | DF of t * t * int
    | DR of t * t ref * ... * int
  (* ... core functions *)
end

(* 3. Builder for operators *)
module Builder_Make(Core : Core_Sig) = struct
  module type Siso = sig ... end
  let build_siso = ...
end

(* 4. Ops module using builder *)
module Ops_Make(Core : Core_Sig) = struct
  module Builder = Builder_Make(Core)
  let sin = Builder.build_siso (module struct ... end)
  (* ... all operators *)
end

(* 5. Final AD module *)
module Make(A : Ndarray_Sig) = struct
  module Core = Core_Make(A)
  include Core
  module Ops = Ops_Make(Core)
  include Ops
  
  let diff f x = ...
  let grad f x = ...
end
```

Usage:
```ocaml
module S = Owl_algodiff_generic.Make(Owl_algodiff_primal_ops.S)
module D = Owl_algodiff_generic.Make(Owl_algodiff_primal_ops.D)
```

### Backends
Can plug in different implementations:
- **Dense**: Owl_dense_ndarray (high performance)
- **Base**: Owl_base_dense_ndarray (pure OCaml)
- **Symbolic**: Computation graph module (optimization)

---

## Chapter 4: Mathematical Optimization (Partial)

### Problem Formulation
```
minimize f(x)
subject to gᵢ(x) ≤ bᵢ
```

### Types
- **Unconstrained**: No gᵢ constraints
- **Constrained**: With inequality constraints
- **Global**: Find absolute minimum
- **Local**: Find minimum in region

Owl focuses on: **Unconstrained local optimization**

### Gradient Descent
Core algorithm for neural network training:
```ocaml
xₙ₊₁ = xₙ - α∇f(xₙ)
```

Newton's method (uses second derivatives):
```ocaml
let rec newton ?(eta=F 0.01) ?(eps=1e-6) f x =
  let g = grad f x in
  let h = hessian f x in
  if (Maths.l2norm' g |> unpack_flt) < eps then x
  else newton ~eta ~eps f Maths.(x - eta * g *@ (inv h))
```

---

## Key Insights for Fowl

### 1. Functor-Based Architecture
- Separation of concerns via functors
- Pluggable backends (performance vs portability)
- Type-safe generic programming

### 2. C-Backend Strategy
- Critical paths in C with OCaml wrappers
- Macros for multi-type code generation
- Release runtime lock for true parallelism

### 3. AD Design
- Dual number representation (forward)
- Computation graph (reverse)
- Builder pattern eliminates boilerplate
- Tags solve perturbation confusion

### 4. Performance Engineering
- SIMD vectorization
- Cache-aware algorithms (tiling)
- NUMA awareness
- Prefetching
- OpenMP for multicore

### 5. API Design
- Low-level APIs for flexibility
- High-level APIs for usability
- Composable function pipeline

---

## F# Mapping Considerations

### Functors → F#
OCaml functors can become:
1. **Interfaces + Records**: Most idiomatic
2. **Computation expressions**: For builder patterns
3. **Statically resolved type parameters**: Inline functions
4. **Object expressions**: When needed

### Example mapping:
```fsharp
// OCaml functor
module Make(A : Ndarray_Sig) = struct ... end

// F# equivalent
module AdEngine<'Arr, 'Elt when 'Arr :> INdarray<'Elt>> =
    let diff f x = ...
    let grad f x = ...
```

Or using interfaces:
```fsharp
type INdarrayOps<'arr, 'elt> =
    abstract sin : 'arr -> 'arr
    abstract add : 'arr -> 'arr -> 'arr
    // ...

module AdEngine =
    let make (ops : INdarrayOps<'a, 'e>) = 
        let diff f x = ...
        let grad f x = ...
        { new IAdEngine with ... }
```

### Memory Considerations
- F# arrays vs .NET multidimensional arrays
- Memory alignment for SIMD
- Struct records for value types
- Span<T> for zero-copy slicing

---

_Next: Continue Chapter 4 (Optimization) and Chapter 5 (Neural Networks)_
