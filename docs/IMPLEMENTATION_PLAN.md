# Fowl Optimization Implementation Plan

## Executive Summary

Based on research of .NET SIMD capabilities, Span<T>, benchmarking infrastructure, and reference libraries (Math.NET, Accord.NET), this document outlines a phased optimization strategy for Fowl.

**Key Findings:**
1. **SIMD can provide 4-8x speedup** for element-wise operations on AVX2
2. **Span<T> enables zero-copy slicing** - reduce memory usage by ~30%
3. **Parallel.For helps for large arrays** (> 10K elements)
4. **Reference libraries underutilize SIMD** - opportunity for Fowl

---

## Phase 1: Benchmarking Infrastructure (Week 1)

**Goal:** Establish baselines before optimization

### Tasks
- [ ] Fix BenchmarkDotNet project (F# compatibility)
- [ ] Run initial benchmarks on key operations
- [ ] Document hardware configuration
- [ ] Create performance regression tests

### Deliverables
- Baseline performance numbers
- Hot path identification
- Hardware capability matrix

### Success Criteria
- Benchmarks run successfully on target hardware
- Reproducible results (low variance)
- Clear identification of top 5 hot paths

---

## Phase 2: SIMD Foundation (Weeks 2-3)

**Goal:** Add portable SIMD with Vector<T>

### 2.1 Create Fowl.SIMD Module

```
src/
  Fowl.SIMD/
    Fowl.SIMD.fsproj
    Core.fs          # Vector<T> wrappers
    ElementWise.fs   # Add, mul, etc.
    Reductions.fs    # Sum, dot product
```

### 2.2 Implement Core Operations

**Priority 1: Element-wise (Highest Impact)**
- [ ] `add` - Array addition
- [ ] `mul` - Array multiplication
- [ ] `sub` - Array subtraction
- [ ] `div` - Array division
- [ ] `map` - Unary operations

**Priority 2: Reductions (Medium Impact)**
- [ ] `sum` - Horizontal add
- [ ] `dot` - Dot product
- [ ] `mean` - Average

**Priority 3: Math Functions (Lower Impact, Harder)**
- [ ] `sin` - Sine (polynomial approx)
- [ ] `exp` - Exponential (polynomial approx)
- [ ] `log` - Logarithm (polynomial approx)

### 2.3 Implementation Pattern

```fsharp
module Fowl.SIMD.ElementWise

open System.Numerics

let add (a: float[]) (b: float[]) : float[] =
    if not Vector.IsHardwareAccelerated then
        // Fallback to Array.map2
        Array.map2 (+) a b
    else
        let vecSize = Vector<double>.Count
        let result = Array.zeroCreate a.Length
        let mutable i = 0
        
        // SIMD loop
        while i <= a.Length - vecSize do
            let va = Vector(a, i)
            let vb = Vector(b, i)
            let vr = va + vb
            vr.CopyTo(result, i)
            i <- i + vecSize
        
        // Scalar remainder
        while i < a.Length do
            result.[i] <- a.[i] + b.[i]
            i <- i + 1
        
        result
```

### Deliverables
- Fowl.SIMD module with element-wise operations
- Benchmarks showing speedup vs Array.map2
- Hardware detection utilities

### Success Criteria
- >= 2x speedup on SSE2 (128-bit)
- >= 4x speedup on AVX2 (256-bit)
- Graceful fallback on unsupported hardware

---

## Phase 3: Hardware-Specific SIMD (Weeks 4-5)

**Goal:** Add AVX2/AVX-512 for x64 servers

### 3.1 Create Native SIMD Project

Create C# project for hardware-specific intrinsics:

```
src/
  Fowl.Native.SIMD/
    Fowl.Native.SIMD.csproj
    Avx2Kernels.cs    # AVX2 implementations
    Sse2Kernels.cs    # SSE2 fallback
    KernelSelector.cs # Auto-detection
```

### 3.2 Implement AVX2 Kernels

```csharp
// Fowl.Native.SIMD/Avx2Kernels.cs
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

public static class Avx2Kernels
{
    public static void Add(double[] a, double[] b, double[] result)
    {
        int vecSize = 4;  // 256 bits / 64 bits
        int i = 0;
        
        // Aligned loop (assume arrays are aligned or use unaligned loads)
        for (; i <= a.Length - vecSize; i += vecSize)
        {
            var va = Vector256.LoadUnsafe(ref a[i]);
            var vb = Vector256.LoadUnsafe(ref b[i]);
            var vr = Avx2.Add(va, vb);
            vr.StoreUnsafe(ref result[i]);
        }
        
        // Remainder
        for (; i < a.Length; i++)
        {
            result[i] = a[i] + b[i];
        }
    }
}
```

### 3.3 Auto-Detection

```csharp
public static class SimdKernelSelector
{
    public static void Add(double[] a, double[] b, double[] result)
    {
        if (Avx2.IsSupported)
        {
            Avx2Kernels.Add(a, b, result);
        }
        else if (Sse2.IsSupported)
        {
            Sse2Kernels.Add(a, b, result);
        }
        else
        {
            // Managed fallback
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] + b[i];
            }
        }
    }
}
```

### 3.4 F# Integration

```fsharp
// Fowl.SIMD/Avx2.fs
module Fowl.SIMD.Avx2

open Fowl.Native.SIMD

let add (a: float[]) (b: float[]) : float[] =
    let result = Array.zeroCreate a.Length
    SimdKernelSelector.Add(a, b, result)
    result
```

### Deliverables
- Fowl.Native.SIMD C# project
- AVX2 kernels for element-wise ops
- Auto-detection and fallback
- Benchmarks showing AVX2 vs Vector<T> vs scalar

### Success Criteria
- >= 6x speedup on AVX2 vs scalar
- No regression on non-AVX2 hardware
- Clean integration with F# codebase

---

## Phase 4: Memory Optimization (Weeks 6-7)

**Goal:** Zero-copy slicing and reduced allocations

### 4.1 Add Span<T> Support

**Step 1: Internal Usage**

```fsharp
// Fowl.Core.Ndarray
let mapInPlace (f: float -> float) (arr: Span<float>) : unit =
    for i = 0 to arr.Length - 1 do
        arr.[i] <- f arr.[i]
```

**Step 2: Public API**

```fsharp
type Ndarray<'K, 'T> with
    member this.AsSpan() : Span<'T> =
        match this with
        | Dense d -> Span(d.Data, d.Offset, d.Data.Length - d.Offset)
        | _ -> failwith "Not implemented for sparse"
```

**Step 3: Zero-Copy Views**

```fsharp
type NdarrayView<'K, 'T> = {
    Source: Ndarray<'K, 'T>
    Offset: int
    Shape: Shape
    Strides: int[]
}

module NdarrayView =
    let slice (view: NdarrayView) (specs: SliceSpec[]) : NdarrayView =
        // Compute new offset and shape, no copy
        { view with ... }
```

### 4.2 Array Pooling

```fsharp
open System.Buffers

module Fowl.Pooling

let rentDoubleArray (size: int) : double[] =
    ArrayPool.Shared.Rent(size)

let returnDoubleArray (arr: double[]) : unit =
    ArrayPool.Shared.Return(arr, clearArray = false)
```

Use in factorizations:

```fsharp
let lu (a: Ndarray) : FowlResult<...> =
    let workspace = rentDoubleArray (n * n)
    try
        // ... use workspace ...
        result
    finally
        returnDoubleArray workspace
```

### 4.3 Stack Allocation for Small Buffers

```fsharp
let smallOperation (x: float) : float =
    let buffer = stackalloc float 16
    // ... use buffer ...
    result
```

### Deliverables
- Span<T> integration in Ndarray
- NdarrayView for zero-copy slicing
- Array pooling for temporaries
- Reduced allocation benchmarks

### Success Criteria
- 30% reduction in allocations for typical workloads
- No API breaking changes
- Zero-copy slicing functional

---

## Phase 5: Parallelization (Weeks 8-9)

**Goal:** Multi-core utilization for large operations

### 5.1 Parallel Thresholds

```fsharp
module Fowl.Parallel

let parallelThreshold = 10000  // Configurable

let shouldParallelize (n: int) : bool =
    n >= parallelThreshold
```

### 5.2 Parallel Element-wise Operations

```fsharp
open System.Threading.Tasks

let addParallel (a: float[]) (b: float[]) : float[] =
    let n = a.Length
    let result = Array.zeroCreate n
    
    if shouldParallelize n then
        Parallel.For(0, n, fun i ->
            result.[i] <- a.[i] + b.[i]
        ) |> ignore
    else
        for i = 0 to n - 1 do
            result.[i] <- a.[i] + b.[i]
    
    result
```

### 5.3 Parallel Matrix Operations

```fsharp
let matmulParallel (a: Ndarray) (b: Ndarray) : Ndarray =
    // Outer loops parallel, inner loop sequential (for cache)
    Parallel.For(0, m, fun i ->
        for j = 0 to n - 1 do
            let mutable sum = 0.0
            for k = 0 to p - 1 do
                sum <- sum + a.[i,k] * b.[k,j]
            c.[i,j] <- sum
    ) |> ignore
```

### 5.4 SIMD + Parallel Combination

```fsharp
let addSimdParallel (a: float[]) (b: float[]) : float[] =
    let n = a.Length
    let result = Array.zeroCreate n
    
    if shouldParallelize n then
        // Chunk for parallel processing
        let chunkSize = n / Environment.ProcessorCount
        Parallel.For(0, Environment.ProcessorCount, fun threadId ->
            let start = threadId * chunkSize
            let end = min (start + chunkSize) n
            // SIMD within each chunk
            addSimdChunk a b result start end
        ) |> ignore
    else
        addSimd a b result
```

### Deliverables
- Parallel versions of element-wise ops
- Parallel matrix multiplication
- Configurable thresholds
- Scaling benchmarks (1, 2, 4, 8 cores)

### Success Criteria
- Near-linear scaling up to 4 cores
- No overhead for small arrays
- Thread-safe random state in Stats

---

## Phase 6: Cache Optimization (Weeks 10-11)

**Goal:** Cache-friendly algorithms for matrix operations

### 6.1 Tiled Matrix Multiplication

```fsharp
let BLOCK_SIZE = 64

let matmulTiled (a: Ndarray) (b: Ndarray) : Ndarray =
    for ii in 0..BLOCK_SIZE..m-1 do
        for jj in 0..BLOCK_SIZE..n-1 do
            for kk in 0..BLOCK_SIZE..p-1 do
                // Process BLOCK_SIZE x BLOCK_SIZE tile
                for i = ii to min(ii+BLOCK_SIZE, m)-1 do
                    for j = jj to min(jj+BLOCK_SIZE, n)-1 do
                        let mutable sum = c.[i,j]
                        for k = kk to min(kk+BLOCK_SIZE, p)-1 do
                            sum <- sum + a.[i,k] * b.[k,j]
                        c.[i,j] <- sum
```

### 6.2 Loop Reordering

```fsharp
// Bad: Strided access
for j in 0..n-1 do
    for i in 0..m-1 do
        a.[i,j]  // Column access (strided)

// Good: Sequential access
for i in 0..m-1 do
    for j in 0..n-1 do
        a.[i,j]  // Row access (sequential)
```

### Deliverables
- Tiled matrix multiplication
- Cache-friendly transpose
- Performance comparison vs naive

### Success Criteria
- 2-4x speedup for large matrix operations
- Better cache hit rates (measured)

---

## Phase 7: Integration & Polish (Week 12)

**Goal:** Integrate all optimizations into main codebase

### 7.1 Integration Strategy

**Replace existing implementations:**

```fsharp
// Before
let add a b = Array.map2 (+) a b

// After
let add a b =
    match detectOptimalStrategy a with
    | Scalar -> addScalar a b
    | VectorT -> addVector a b
    | Avx2 -> addAvx2 a b
    | ParallelAvx2 -> addParallelAvx2 a b
```

### 7.2 Configuration API

```fsharp
module Fowl.Config

type OptimizationConfig = {
    EnableSIMD: bool
    EnableParallel: bool
    ParallelThreshold: int
    EnableArrayPooling: bool
}

let defaultConfig = {
    EnableSIMD = true
    EnableParallel = true
    ParallelThreshold = 10000
    EnableArrayPooling = true
}

let mutable currentConfig = defaultConfig
```

### 7.3 Documentation

- [ ] Update ARCHITECTURE.md with optimization details
- [ ] Add performance tuning guide
- [ ] Document hardware requirements
- [ ] Migration guide for users

### 7.4 Testing

- [ ] Property-based tests for SIMD vs scalar
- [ ] Multi-hardware CI testing
- [ ] Performance regression tests
- [ ] Thread safety tests

### Deliverables
- Optimized Fowl with all improvements
- User documentation
- Performance report
- Test coverage

---

## Timeline Summary

| Phase | Weeks | Focus | Expected Speedup |
|-------|-------|-------|-----------------|
| 1 | 1 | Benchmarking | Baseline |
| 2 | 2-3 | Vector<T> SIMD | 2-4x element-wise |
| 3 | 4-5 | AVX2 SIMD | 4-8x element-wise |
| 4 | 6-7 | Span<T>, Pooling | 30% less memory |
| 5 | 8-9 | Parallelization | 2-4x on multi-core |
| 6 | 10-11 | Cache optimization | 2-4x matrix ops |
| 7 | 12 | Integration | Production ready |

**Total: 12 weeks (3 months)**

---

## Expected Final Performance

### Element-wise Operations (1M elements)

| Implementation | Time | Speedup |
|----------------|------|---------|
| Baseline (current) | 10 ms | 1x |
| Vector<T> | 3 ms | 3.3x |
| AVX2 | 1.5 ms | 6.7x |
| AVX2 + Parallel (8 cores) | 0.3 ms | 33x |

### Matrix Multiplication (1000x1000)

| Implementation | Time | Speedup |
|----------------|------|---------|
| Baseline (naive) | 2000 ms | 1x |
| Cache-tiled | 500 ms | 4x |
| Parallel (8 cores) | 100 ms | 20x |
| BLAS (reference) | 50 ms | 40x |

---

## Risk Mitigation

### Risk 1: SIMD Complexity
**Mitigation:** Start with Vector<T>, add hardware-specific later

### Risk 2: Hardware Variability
**Mitigation:** Extensive testing on different CPUs, graceful fallbacks

### Risk 3: Unsafe Code Safety
**Mitigation:** Property-based testing, fuzzing, code review

### Risk 4: API Breaking Changes
**Mitigation:** Maintain backward compatibility, add new APIs alongside old

### Risk 5: Maintenance Burden
**Mitigation:** Clear documentation, modular design, automated testing

---

## Success Metrics

1. **Performance:** >= 4x speedup on typical workloads
2. **Memory:** >= 30% reduction in allocations
3. **Compatibility:** Works on all .NET 8+ platforms
4. **Correctness:** 100% test pass rate, property-based verification
5. **Adoption:** User benchmarks show improvement

---

## Next Steps

1. **Immediate:** Start Phase 1 (Benchmarking)
2. **Week 1:** Establish baselines
3. **Week 2:** Begin Vector<T> implementation
4. **Monthly:** Review progress, adjust timeline

---

_Last updated: 2026-02-14_
_Plan version: 1.0_
