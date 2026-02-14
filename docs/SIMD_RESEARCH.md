# SIMD Research Notes - System.Runtime.Intrinsics

## Overview

.NET Core 3.0+ and .NET 5+ provide hardware-specific SIMD intrinsics through `System.Runtime.Intrinsics` namespace. This enables writing vectorized code that maps directly to CPU instructions.

---

## Key Types

### Vector128<T>, Vector256<T>, Vector512<T>

Fixed-size SIMD vectors that map to hardware registers:

- **Vector128<float>**: 4 floats (128 bits) - SSE
- **Vector256<float>**: 8 floats (256 bits) - AVX/AVX2
- **Vector256<double>**: 4 doubles (256 bits) - AVX/AVX2
- **Vector512<float>**: 16 floats (512 bits) - AVX-512

### Vector<T>

Variable-size vector (hardware-determined):
- Uses largest available vector size
- Good for portable code
- Less control than fixed-size vectors

---

## Hardware Detection

```csharp
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// Check for AVX2 support
if (Avx2.IsSupported) {
    // Use 256-bit operations
}

// Check for SSE2 support (nearly universal on x64)
else if (Sse2.IsSupported) {
    // Use 128-bit operations
}

// Check for ARM64 NEON
else if (AdvSimd.IsSupported) {
    // Use ARM NEON instructions
}

else {
    // Scalar fallback
}
```

---

## Element-wise Addition Example

### Scalar (Baseline)

```csharp
void AddScalar(double[] a, double[] b, double[] result) {
    for (int i = 0; i < a.Length; i++) {
        result[i] = a[i] + b[i];
    }
}
```

### Vector<T> (Portable SIMD)

```csharp
using System.Numerics;

void AddVector(double[] a, double[] b, double[] result) {
    int vecSize = Vector<double>.Count;  // 2, 4, or 8 depending on hardware
    int i = 0;

    // Process vectors
    for (; i <= a.Length - vecSize; i += vecSize) {
        var va = new Vector<double>(a, i);
        var vb = new Vector<double>(b, i);
        var vr = va + vb;
        vr.CopyTo(result, i);
    }

    // Process remainder
    for (; i < a.Length; i++) {
        result[i] = a[i] + b[i];
    }
}
```

### AVX2 (256-bit, Hardware-Specific)

```csharp
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

void AddAvx2(double[] a, double[] b, double[] result) {
    if (!Avx2.IsSupported) {
        AddVector(a, b, result);  // Fallback
        return;
    }

    int vecSize = 256 / 64;  // 4 doubles = 256 bits
    int i = 0;

    // Process 256-bit vectors
    for (; i <= a.Length - vecSize; i += vecSize) {
        var va = Vector256.LoadUnsafe(ref a[i]);
        var vb = Vector256.LoadUnsafe(ref b[i]);
        var vr = Avx2.Add(va, vb);
        vr.StoreUnsafe(ref result[i]);
    }

    // Process remainder
    for (; i < a.Length; i++) {
        result[i] = a[i] + b[i];
    }
}
```

---

## Performance Expectations

### Element-wise Operations

| Operation | Scalar | SSE2 (128b) | AVX2 (256b) | AVX-512 (512b) |
|-----------|--------|-------------|-------------|----------------|
| Add | 1x | 2x | 4x | 8x |
| Multiply | 1x | 2x | 4x | 8x |
| FMA (a*b+c) | 1x | 2x | 4x | 8x |

### Reductions (Sum, Dot Product)

Reductions are harder to vectorize efficiently due to horizontal operations:

```csharp
// Horizontal add example (AVX2)
var sum = Vector256.Create(0.0);
for (...) {
    var v = Vector256.LoadUnsafe(...);
    sum = Avx2.Add(sum, v);
}
// Extract and sum 4 lanes
var partial = sum.GetLower() + sum.GetUpper();
// ... more extraction
```

Expected speedup: 2-4x (less than element-wise due to overhead)

---

## Memory Alignment

### Alignment Requirements

- **SSE**: 16-byte alignment preferred
- **AVX**: 32-byte alignment preferred
- **AVX-512**: 64-byte alignment preferred

### Alignment in .NET

.NET arrays are not guaranteed to be aligned for SIMD. Options:

1. **Use unaligned loads** (slower but safe):
```csharp
var v = Avx2.LoadVector256(ptr);  // Unaligned
```

2. **Pin arrays and align manually** (complex):
```csharp
fixed (double* ptr = array) {
    // Align pointer to 32 bytes
    var alignedPtr = (double*)(((ulong)ptr + 31) & ~31UL);
}
```

3. **Process unaligned prefix, then aligned bulk**:
```csharp
// Process first elements to reach alignment
// Then use aligned loads for bulk
```

---

## Cache Considerations

### Spatial Locality

SIMD works best with contiguous memory access. Strided access (e.g., matrix columns) is harder:

```csharp
// Good: contiguous access
for (int i = 0; i < n; i++) {
    var v = Vector256.LoadUnsafe(ref row[i]);
}

// Bad: strided access (columns)
for (int i = 0; i < n; i++) {
    // Loading single elements, no SIMD benefit
    double x = matrix[i * stride];
}
```

### Blocking/Tiling

For large arrays, process in cache-friendly blocks:

```csharp
const int BLOCK_SIZE = 4096;  // L1 cache size

for (int block = 0; block < n; block += BLOCK_SIZE) {
    int end = Math.Min(block + BLOCK_SIZE, n);
    for (int i = block; i < end; i += vecSize) {
        // SIMD operations on block
    }
}
```

---

## Transcendental Functions (sin, exp, log)

SIMD doesn't directly support transcendental functions. Options:

1. **Use System.Math** (scalar) - slowest
2. **Use Intel MKL** (via native calls) - fast but external dependency
3. **Polynomial approximations** - fast SIMD, less accurate

Example polynomial approximation for exp:

```csharp
// Minimax polynomial approximation
Vector256<double> ExpApprox(Vector256<double> x) {
    // Coefficients from approximation tables
    var c0 = Vector256.Create(1.0);
    var c1 = Vector256.Create(1.0);
    var c2 = Vector256.Create(0.5);
    // ... more coefficients

    var result = c0 + x * (c1 + x * (c2 + ...));
    return result;
}
```

---

## F# Integration

### Using C# SIMD from F#

Since System.Runtime.Intrinsics is primarily C#, we can:

1. **Write SIMD kernels in C#**, call from F#
2. **Use F# inline IL** (complex, not recommended)
3. **Wait for F# SIMD support** (not imminent)

### Recommended Approach

Create C# library `Fowl.Native.SIMD`:

```csharp
// Fowl.Native.SIMD/SimdKernels.cs
public static class SimdKernels {
    public static void Add(double[] a, double[] b, double[] result) {
        // SIMD implementation
    }
}
```

Reference from F#:

```fsharp
open Fowl.Native.SIMD

let addSimd a b result =
    SimdKernels.Add(a, b, result)
```

---

## Benchmarking SIMD

### What to Measure

1. **Throughput**: Elements processed per second
2. **Speedup**: SIMD time / Scalar time
3. **Scaling**: Performance vs array size
4. **Memory bandwidth**: Are we compute or memory bound?

### Expected Results

- Small arrays (< 1KB): SIMD overhead dominates, may be slower
- Medium arrays (1KB - 1MB): SIMD shines, 2-8x speedup
- Large arrays (> 1MB): Memory bandwidth bound, limited speedup

---

## Implementation Strategy for Fowl

### Phase 1: Vector<T> Wrapper

Create F# wrapper around Vector<T> for portable SIMD:

```fsharp
module Fowl.SIMD.Core

open System.Numerics

let simdAdd (a: float[]) (b: float[]) : float[] =
    let result = Array.zeroCreate a.Length
    let vecSize = Vector<double>.Count
    
    let mutable i = 0
    while i <= a.Length - vecSize do
        let va = Vector(a, i)
        let vb = Vector(b, i)
        let vr = va + vb
        vr.CopyTo(result, i)
        i <- i + vecSize
    
    // Remainder
    while i < a.Length do
        result.[i] <- a.[i] + b.[i]
        i <- i + 1
    
    result
```

### Phase 2: Hardware-Specific Paths

Add AVX2 path for x64 servers:

```fsharp
let addOptimized (a: float[]) (b: float[]) : float[] =
    if Avx2.IsSupported then
        addAvx2 a b
    elif Sse2.IsSupported then
        addSse2 a b
    else
        addScalar a b
```

### Phase 3: Integration

Replace Array.map2 in Ndarray operations:

```fsharp
// Before
let add a b =
    Array.map2 (+) a b

// After
let add a b =
    simdAdd a b
```

---

## Open Questions

1. **How to handle F# type inference with Vector<T>?**
2. **Should we use C# for SIMD kernels or pure F#?**
3. **How to benchmark on different hardware?**
4. **What's the crossover point where SIMD is beneficial?**

---

## References

- [.NET Hardware Intrinsics Documentation](https://docs.microsoft.com/en-us/dotnet/api/system.runtime.intrinsics)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [ARM NEON Documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- Owl Architecture Book, Chapter 2 (Performance Optimization)

---

_Research in progress. Next: Span<T> and Memory<T> study._
