# Fowl Architecture

## Overview

Fowl is a high-performance numerical computing library for F#, modeled after OCaml's Owl library but designed with modern F# idioms and .NET performance characteristics in mind.

## Design Principles

1. **Type Safety First**: Leverage F#'s type system (phantom types, DUs) for correctness
2. **Composability**: Railway-oriented programming with Result types
3. **Performance Options**: Start correct, optimize hot paths
4. **Functional Core**: Immutable by default, mutable where needed
5. **Interop Friendly**: Easy integration with .NET ecosystem

## Module Structure

```
Fowl/
├── Core/          # Ndarray, Shape, Slice, Matrix primitives
├── Native/        # BLAS/LAPACK bindings (OpenBLAS)
├── Linalg/        # Linear algebra (solve, inverse, decompositions)
├── Stats/         # Statistics and distributions
└── AD/            # Algorithmic Differentiation
```

## Type System

### Phantom Types
```fsharp
type Ndarray<'K, 'T> = ...  // 'K is phantom (Float32, Float64, etc)
```

Used for type safety without runtime overhead. Could be extended to prevent mixing Float32/Float64 arrays at compile time.

### Error Handling
All public APIs return `FowlResult<'T>`:
```fsharp
type FowlResult<'T> = Result<'T, FowlError>

type FowlError =
    | InvalidShape of string
    | DimensionMismatch of string
    | NativeLibraryError of string
    | ...
```

## Performance Architecture

### Current State (Phase A: Correctness)
- Managed arrays with functional operations
- OpenBLAS for matrix operations
- No SIMD in managed code (5-10x slower than C for element-wise ops)

### Phase B: SIMD Optimization
Target: System.Runtime.Intrinsics for hardware acceleration

**Element-wise Operations:**
```fsharp
// Current
Array.map2 (+) a b  // Managed, scalar

// Optimized (Phase B)
use va = Vector.Load(a, i)
use vb = Vector.Load(b, i)
Vector.Add(va, vb)  // SIMD
```

**Hardware Support:**
- x64: SSE2, AVX2, AVX-512
- ARM: AdvSIMD (NEON)
- Fallback: Managed scalar code

### Phase C: Memory Optimization
Target: Span<T>, Memory<T>, ArrayPool

**Zero-Copy Slicing:**
```fsharp
// Current: Always copies
let slice arr specs = ... // returns new array

// Optimized (Phase C)
let sliceView arr specs =
    // Return Span<T> view into original memory
    Span(arr.Data, offset, length)
```

**Array Pooling:**
```fsharp
// For temporary allocations in hot loops
let pool = ArrayPool.Shared
let rented = pool.Rent(size)
try
    // use rented array
finally
    pool.Return(rented)
```

### Phase D: Parallelization
Target: Parallel.For + SIMD

**Independent Operations:**
```fsharp
Parallel.For(0, n, fun i ->
    // Each thread processes SIMD vectors
    let va = Vector.Load(a, i * Vector.Count)
    ...
)
```

**Thread Safety:**
- RandomState is immutable (safe to share)
- Ndarray operations are pure (safe to parallelize)
- Careful with mutable buffers

## Optimization Strategy

Per Owl Architecture Book (Chapter 2):

1. **Profile First**: Use BenchmarkDotNet to identify hot paths
2. **Optimize Iteratively**: SIMD → Parallel → Unsafe
3. **Maintain Correctness**: Property-based tests verify optimizations
4. **Platform Abstraction**: Abstract SIMD behind computation expressions

## Benchmarking Plan

### Baseline Measurements
- Element-wise: add, mul, sin, exp
- Matrix: matmul, solve, inverse
- Stats: mean, var, rvs
- AD: diff, grad

### Comparison Targets
- NumPy (Python)
- Owl (OCaml)
- Math.NET (C#)
- MKL (Intel, native)

### Metrics
- Throughput (elements/second)
- Memory allocations
- Cache misses
- Scaling with cores

## Future Directions

### Type Providers
```fsharp
type Data = CsvProvider<"data.csv">
let arr = Data.Load().AsNdarray()
```

### GPU Acceleration
- ComputeSharp (DirectX 12)
- ILGPU (CUDA/PTX)
- OpenCL via bindings

### Distributed Computing
- Actor-based (Akka.NET)
- Dataflow (TPL)
- MPI for clusters

## References

- Owl Architecture Book (Wang & Zhao, 2023)
- .NET Performance Docs (Microsoft)
- System.Runtime.Intrinsics API
- F# for Fun and Profit (Wlaschin)

---

*Architecture evolves with the codebase. Update as features are added.*
