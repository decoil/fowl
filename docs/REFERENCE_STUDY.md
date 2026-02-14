# Reference Library Study: Math.NET Numerics & Accord.NET

## Overview

Studying established .NET numerical libraries to understand optimization patterns, API design, and implementation strategies applicable to Fowl.

---

## Math.NET Numerics

### Architecture

**Provider Pattern:**
Math.NET uses a provider pattern to abstract implementation:

```csharp
// Linear algebra provider
LinearAlgebra.Provider = new ManagedLinearAlgebraProvider();
// or
LinearAlgebra.Provider = new NativeMKLProvider();
```

This allows:
- Switching between managed and native implementations
- Platform-specific optimizations
- Easy testing with different backends

**Implementation Strategy:**
1. Define interface (e.g., `ILinearAlgebraProvider`)
2. Implement managed fallback
3. Implement native optimized version
4. Auto-detect best provider at runtime

### Memory Management

**Array Pooling:**
```csharp
// Uses ArrayPool internally for temporary buffers
var result = new double[n];
using (var workspace = new ArrayPoolWorkspace<double>(n))
{
    // Use workspace.Array for temporaries
}
```

**In-place Operations:**
```csharp
// Avoid allocation
Vector.Add(a, b, result);  // result is pre-allocated
```

### SIMD Usage

Math.NET **does not** extensively use SIMD intrinsics. Instead:
- Relies on native BLAS/LAPACK (Intel MKL, OpenBLAS)
- For managed fallbacks, uses standard loops
- No System.Runtime.Intrinsics usage

**Lesson for Fowl:** SIMD may be less critical if we have good native bindings, but still valuable for:
- Element-wise operations not in BLAS
- Small arrays where native call overhead dominates
- Platforms without optimized BLAS

### API Design Patterns

**Fluent Interface:**
```csharp
var result = a.PointwiseMultiply(b).Add(c).Map(x => x * 2);
```

**Immutable by Default:**
```csharp
var c = a + b;  // Creates new vector
```

**Explicit In-place:**
```csharp
a.Add(b, result);  // Reuses result array
```

### Performance Characteristics

**Strengths:**
- Excellent native BLAS integration
- Consistent API across providers
- Good documentation

**Weaknesses:**
- No SIMD in managed code
- Allocation-heavy (immutable design)
- Complex provider setup

---

## Accord.NET

### Architecture

**Modular Design:**
Accord.NET is organized by domain:
- Accord.Math (linear algebra, optimization)
- Accord.Statistics (distributions, tests)
- Accord.MachineLearning (ML algorithms)

Each module can be used independently.

### Optimization Techniques

**1. Unsafe Code Blocks:**
Accord.NET uses `unsafe` extensively for performance:

```csharp
unsafe
{
    fixed (double* ptrA = a)
    fixed (double* ptrB = b)
    {
        for (int i = 0; i < n; i++)
        {
            ptrA[i] += ptrB[i];
        }
    }
}
```

**Advantages:**
- Eliminates bounds checking
- Direct pointer arithmetic
- Compatible with SIMD (when used with Vector<T>)

**Disadvantages:**
- Requires `<AllowUnsafeBlocks>true</AllowUnsafeBlocks>`
- Potential memory safety issues
- Harder to debug

**2. Parallel.For for Large Operations:**

```csharp
Parallel.For(0, n, i =>
{
    result[i] = a[i] + b[i];
});
```

Used for:
- Matrix operations
- Batch processing
- Independent element-wise ops

**3. Cache-Friendly Algorithms:**

```csharp
// Matrix multiplication with tiling
const int BLOCK_SIZE = 64;

for (int ii = 0; ii < n; ii += BLOCK_SIZE)
{
    for (int jj = 0; jj < n; jj += BLOCK_SIZE)
    {
        for (int kk = 0; kk < n; kk += BLOCK_SIZE)
        {
            // Process BLOCK_SIZE x BLOCK_SIZE tile
            for (int i = ii; i < Math.Min(ii + BLOCK_SIZE, n); i++)
            {
                for (int k = kk; k < Math.Min(kk + BLOCK_SIZE, n); k++)
                {
                    double aik = a[i, k];
                    for (int j = jj; j < Math.Min(jj + BLOCK_SIZE, n); j++)
                    {
                        c[i, j] += aik * b[k, j];
                    }
                }
            }
        }
    }
}
```

**4. Algorithm Selection:**

Accord.NET chooses algorithms based on size:

```csharp
if (n < 100)
{
    // Use simple algorithm (less overhead)
    SimpleMultiply(a, b, c);
}
else
{
    // Use optimized algorithm
    OptimizedMultiply(a, b, c);
}
```

### SIMD Usage

Accord.NET has **limited SIMD** usage:
- Some Vector<T> usage in recent versions
- Mostly relies on unsafe pointer loops
- No System.Runtime.Intrinsics

**Pattern:**
```csharp
// Vector<T> when available
if (Vector.IsHardwareAccelerated)
{
    // Use Vector<T>
}
else
{
    // Use unsafe loops
}
```

### Memory Management

**Jagged Arrays vs 2D Arrays:**
Accord.NET uses jagged arrays `double[][]` instead of 2D arrays `double[,]`:

**Advantages:**
- Better cache locality for row operations
- Compatible with native APIs (BLAS expects contiguous)
- Easier to work with in unsafe code

**Disadvantages:**
- More complex memory layout
- Extra indirection per row

**Pattern:**
```csharp
// Create jagged array
double[][] matrix = new double[rows][];
for (int i = 0; i < rows; i++)
{
    matrix[i] = new double[cols];
}

// Access
matrix[i][j] = value;  // vs matrix[i, j]
```

### Performance Characteristics

**Strengths:**
- Extensive unsafe code for speed
- Good parallelization
- Cache-friendly algorithms
- Rich algorithm selection

**Weaknesses:**
- Requires unsafe code
- Limited SIMD
- Sometimes over-engineered

---

## Comparison: Math.NET vs Accord.NET vs Fowl (Current)

| Feature | Math.NET | Accord.NET | Fowl (Current) |
|---------|----------|------------|----------------|
| SIMD | ❌ Native only | ⚠️ Limited Vector<T> | ❌ None |
| Unsafe code | ❌ No | ✅ Yes | ❌ No |
| Parallel | ⚠️ Limited | ✅ Yes | ❌ No |
| Native BLAS | ✅ Yes | ✅ Yes | ✅ Yes |
| Immutable API | ✅ Yes | ⚠️ Mixed | ✅ Yes (Result types) |
| Span<T> | ❌ No | ❌ No | ❌ No |
| Allocation | High | Medium | Medium |

---

## Lessons for Fowl

### 1. SIMD Strategy

**Decision:** Implement SIMD gradually

- **Phase 1:** Vector<T> for portable SIMD (50% of benefit, 20% of effort)
- **Phase 2:** System.Runtime.Intrinsics for AVX2 (80% of benefit, 50% of effort)
- **Phase 3:** Platform-specific optimizations (95% of benefit, 100% of effort)

### 2. Unsafe Code

**Decision:** Use unsafe code selectively

- **Use for:** Hot loops where bounds checking matters
- **Avoid for:** Public APIs (keep them safe)
- **Mitigation:** Extensive testing, property-based tests

### 3. Parallelization

**Decision:** Add Parallel.For for large operations

**Thresholds:**
- < 1000 elements: Sequential (overhead dominates)
- 1000 - 100000: Single parallel operation
- > 100000: Nested parallelism with care

### 4. Memory Management

**Decision:** Add Span<T> support

- **Zero-copy slicing:** Views instead of copies
- **Array pooling:** For temporary buffers
- **Stack allocation:** For small fixed-size buffers

### 5. Provider Pattern

**Decision:** Not needed initially

Math.NET's provider pattern adds complexity. Instead:
- Use function pointers for pluggable implementations
- Auto-detect best implementation at startup
- Keep it simple: managed → SIMD → native

### 6. Algorithm Selection

**Decision:** Add size-based dispatch

```fsharp
let add a b =
    let n = numel a
    if n < 100 then
        addScalar a b      // Simple loop
    elif n < 10000 then
        addSIMD a b        // Vector<T>
    else
        addParallelSIMD a b // Parallel + SIMD
```

---

## Optimization Patterns to Adopt

### Pattern 1: Unsafe Hot Loops

```fsharp
// F# with unsafe
#nowarn "9"  // Allow unsafe code

let addUnsafe (a: float[]) (b: float[]) (result: float[]) =
    let n = a.Length
    let mutable i = 0
    
    // Use Unsafe.Add to avoid bounds checking
    while i < n do
        let ptrA = &&a.[0]
        let ptrB = &&b.[0]
        let ptrR = &&result.[0]
        
        NativeInterop.NativePtr.write 
            (NativeInterop.NativePtr.add ptrR i)
            (NativeInterop.NativePtr.read (NativeInterop.NativePtr.add ptrA i) +
             NativeInterop.NativePtr.read (NativeInterop.NativePtr.add ptrB i))
        i <- i + 1
```

### Pattern 2: SIMD with Fallback

```fsharp
open System.Numerics

let addOptimized (a: float[]) (b: float[]) =
    if Vector.IsHardwareAccelerated && a.Length >= Vector<double>.Count then
        addVector a b
    else
        addScalar a b
```

### Pattern 3: Parallel Threshold

```fsharp
open System.Threading.Tasks

let addParallel (a: float[]) (b: float[]) =
    let n = a.Length
    let result = Array.zeroCreate n
    
    if n < 1000 then
        // Sequential
        for i = 0 to n - 1 do
            result.[i] <- a.[i] + b.[i]
    else
        // Parallel
        Parallel.For(0, n, fun i ->
            result.[i] <- a.[i] + b.[i]
        ) |> ignore
    
    result
```

### Pattern 4: Array Pooling

```fsharp
open System.Buffers

let someOperation (a: Ndarray) (b: Ndarray) =
    let pool = ArrayPool.Shared
    let temp = pool.Rent(10000)
    
    try
        // ... use temp ...
        result
    finally
        pool.Return(temp, clearArray = false)
```

---

## Open Questions

1. **Should we use unsafe code or stick to safe Span<T>?**
   - Unsafe: Faster, riskier
   - Safe Span<T>: Slower, safer
   - Compromise: Unsafe in internal kernels, safe public APIs

2. **How to structure SIMD code organization?**
   - Option A: Separate C# project for SIMD
   - Option B: F# with inline IL
   - Option C: Wait for F# SIMD support

3. **What's the threshold for parallelization?**
   - Depends on hardware
   - Should be tunable/configurable

4. **How to benchmark across different hardware?**
   - CI with multiple runners?
   - Community benchmarking?

---

## References

- [Math.NET Numerics Source](https://github.com/mathnet/mathnet-numerics)
- [Accord.NET Source](https://github.com/accord-net/framework)
- [Math.NET Documentation](https://numerics.mathdotnet.com/)
- [Accord.NET Documentation](http://accord-framework.net/)

---

_Research: Phase 4 of 5 complete. Next: Implementation plan._
