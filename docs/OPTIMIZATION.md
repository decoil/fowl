# Fowl Optimization Research Plan

## Overview

This document outlines the research and implementation plan for optimizing Fowl's performance using modern .NET capabilities.

## Research Goals

1. **Understand .NET SIMD landscape** - System.Runtime.Intrinsics, Vector<T>, hardware acceleration
2. **Memory optimization** - Span<T>, Memory<T>, ArrayPool for zero-copy operations
3. **Benchmarking** - Establish baselines, identify hot paths, measure improvements
4. **Reference implementations** - Study Math.NET Numerics, Accord.NET patterns
5. **Phased implementation plan** - Prioritized optimization roadmap

---

## Phase 1: SIMD Research (.NET 8+)

### System.Runtime.Intrinsics

.NET 8+ provides hardware-specific intrinsics via `System.Runtime.Intrinsics` namespace:

**Key APIs:**
- `Vector128<T>` / `Vector256<T>` / `Vector512<T>` - Fixed-size SIMD vectors
- `Vector<T>` - Variable-size (hardware-determined) vectors
- `Avx`, `Avx2`, `Sse`, `Sse2` - x86/x64 instruction sets
- `AdvSimd` - ARM64 NEON instructions

**Research Questions:**
1. Which operations benefit most from SIMD? (element-wise, matrix, reduction)
2. How to detect hardware capabilities at runtime?
3. How to structure fallback to scalar code?
4. What are the memory alignment requirements?

**Target Operations:**
- [ ] Element-wise: add, mul, sin, exp
- [ ] Reductions: sum, mean, dot product
- [ ] Matrix: matmul (when not using BLAS)

### Hardware Support Matrix

| Platform | Instructions | .NET Support |
|----------|-------------|--------------|
| x64 | SSE2, AVX, AVX2, AVX-512 | Full |
| ARM64 | NEON (AdvSimd) | Full |
| WASM | SIMD128 | Limited |

---

## Phase 2: Memory Optimization

### Span<T> and Memory<T>

**Use Cases:**
1. **Zero-copy slicing** - Create views without array copying
2. **Stack allocation** - `stackalloc` for small temporary buffers
3. **Unified API** - Work with arrays, strings, unmanaged memory uniformly

**Implementation Targets:**
- [ ] Slice operations return Span<T> instead of new arrays
- [ ] Matrix operations use Span<T> for row/column access
- [ ] Avoid allocations in hot loops with ArrayPool

### ArrayPool<T>

**Use Cases:**
1. Temporary buffers in factorizations
2. Work arrays in reduction operations
3. Avoid GC pressure in tight loops

---

## Phase 3: Benchmarking

### BenchmarkDotNet Setup

**Baseline Measurements:**
- Element-wise operations (add, mul, sin, exp)
- Matrix operations (matmul, solve, inverse)
- Statistics (mean, var, rvs)
- AD operations (diff, grad)

**Comparison Targets:**
- NumPy (via Python.NET or file export)
- Math.NET Numerics
- Accord.NET
- Native MKL (via existing bindings)

**Metrics:**
- Throughput (elements/second or GFLOPS)
- Memory allocations (bytes/op)
- Cache misses (hardware counters if available)
- Scaling with core count

---

## Phase 4: Reference Study

### Math.NET Numerics

**Optimization Patterns:**
- Provider pattern for native implementations
- Memory management strategies
- SIMD usage (if any)
- API design for performance

### Accord.NET

**Optimization Patterns:**
- Parallel.For usage
- Unsafe code blocks
- Cache-friendly algorithms
- Platform-specific optimizations

---

## Implementation Roadmap

### Priority 1: Element-wise Operations (High Impact, Easy)
- SIMD vectorization of add, mul, sub, div
- Benchmark against managed Array.map2
- Expected: 4-16x speedup on AVX2

### Priority 2: Reduction Operations (High Impact, Medium)
- SIMD sum, mean, dot product
- Horizontal add operations
- Expected: 4-8x speedup

### Priority 3: Memory Optimization (Medium Impact, Easy)
- Span<T> for slicing
- ArrayPool for temporaries
- Expected: Reduced GC pressure

### Priority 4: Matrix Operations (High Impact, Hard)
- Cache-tiled matrix multiplication
- Parallel.For for large matrices
- SIMD for small matrices
- Expected: 2-4x speedup over naive

### Priority 5: Advanced SIMD (Medium Impact, Hard)
- SIMD transcendental functions (sin, exp, log)
- AVX-512 for server workloads
- ARM64 NEON optimization

---

## Research Notes

### SIMD Detection Pattern

```csharp
if (Avx2.IsSupported) {
    // Use AVX2 (256-bit vectors)
} else if (Sse2.IsSupported) {
    // Use SSE2 (128-bit vectors)
} else {
    // Scalar fallback
}
```

### Span<T> Slicing Pattern

```csharp
// Instead of copying
let slice = Array.sub arr start length

// Use Span view
let span = ReadOnlySpan(arr, start, length)
```

### Vector<T> Pattern

```csharp
let vecCount = Vector<double>.Count
for i in 0..data.Length-1..vecCount do
    let va = Vector(data, i)
    let vb = Vector(data2, i)
    let vr = va + vb  // SIMD add
    vr.CopyTo(result, i)
```

---

## References

- [.NET SIMD Blog Series](https://devblogs.microsoft.com/dotnet/hardware-intrinsics-in-net-core/)
- [System.Runtime.Intrinsics Docs](https://docs.microsoft.com/en-us/dotnet/api/system.runtime.intrinsics)
- [Span<T> Guide](https://docs.microsoft.com/en-us/dotnet/api/system.span-1)
- [BenchmarkDotNet Docs](https://benchmarkdotnet.org/)
- Math.NET Numerics Source
- Accord.NET Source

---

## Progress Tracking

| Phase | Status | Notes |
|-------|--------|-------|
| 1. SIMD Research | 🔄 In Progress | Studying intrinsics APIs |
| 2. Memory Research | ⏳ Pending | Span<T>, ArrayPool |
| 3. Benchmarking | ⏳ Pending | Setup BenchmarkDotNet |
| 4. Reference Study | ⏳ Pending | Math.NET, Accord.NET |
| 5. Implementation Plan | ⏳ Pending | Prioritized roadmap |

---

_Last updated: 2026-02-14_
