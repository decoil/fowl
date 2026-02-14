# Span<T> and Memory<T> Research Notes

## Overview

`Span<T>` and `Memory<T>` are .NET types for representing contiguous regions of arbitrary memory. They enable:
- Zero-copy slicing
- Stack-allocated buffers
- Unified API for arrays, strings, unmanaged memory
- Better performance through reduced allocations

---

## Span<T> vs Memory<T>

| Feature | Span<T> | Memory<T> |
|---------|---------|-----------|
| Stack-only | Yes (ref struct) | No |
| Can be stored in heap | No | Yes |
| Async/await friendly | No | Yes |
| Use case | Synchronous operations | Asynchronous operations |
| Performance | Faster (no heap alloc) | Slightly slower |

### Rule of Thumb

- **Use Span<T>** for synchronous, short-lived operations
- **Use Memory<T>** when data needs to outlive method scope or be used in async

---

## Current Fowl: Copy-heavy Operations

### Slicing (Current Implementation)

```fsharp
// Current: Always copies
let slice (arr: Ndarray) (specs: SliceSpec[]) : Ndarray =
    let outData = Array.zeroCreate newSize
    // ... copy elements one by one ...
    Ndarray.ofArray outData newShape
```

**Problem**: O(n) copy for every slice operation

### With Span<T> (Proposed)

```fsharp
// Proposed: Zero-copy view
let sliceView (arr: Ndarray) (specs: SliceSpec[]) : Span<float> =
    // Calculate offset and length
    let offset = computeOffset arr specs
    let length = computeLength specs
    // Return view into existing memory
    Span(arr.Data, offset, length)
```

---

## Use Cases for Fowl

### 1. Zero-Copy Slicing

**Before:**
```fsharp
let arr = Ndarray.zeros [|1000; 1000|]  // 8MB array
let row = arr.GetRow 5                   // Copies 1000 doubles = 8KB
```

**After:**
```fsharp
let arr = Ndarray.zeros [|1000; 1000|]
let rowView = arr.GetRowSpan 5           // No copy, just offset calculation
```

### 2. Matrix Row/Column Access

```fsharp
// Row-major layout: row is contiguous
type Ndarray with
    member this.GetRowSpan(row: int) : Span<float> =
        let cols = this.Shape.[1]
        let offset = row * cols
        Span(this.Data, offset, cols)
    
    // Column is strided (not contiguous)
    // Can still use Span but with manual indexing
    member this.GetColumnView(col: int) : ColumnView =
        { Data = this.Data; Col = col; Stride = this.Shape.[1] }
```

### 3. Element-wise Operations Without Allocation

```fsharp
// Before: allocates new array
let addInPlace (a: Ndarray) (b: Ndarray) : unit =
    for i = 0 to a.Length - 1 do
        a.Data.[i] <- a.Data.[i] + b.Data.[i]

// After: use Span for bounds checking elimination
let addInPlaceSpan (a: Span<float>) (b: ReadOnlySpan<float>) : unit =
    for i = 0 to a.Length - 1 do
        a.[i] <- a.[i] + b.[i]
```

### 4. Stack Allocation for Small Buffers

```fsharp
// Avoid heap allocation for small temporary arrays
let computeSomething (x: float) : float =
    let buffer = stackalloc float 16  // Stack allocation
    // Use buffer...
    result
```

### 5. String/Buffer Parsing

For data loading (CSV, binary formats):

```fsharp
open System.Buffers

let parseCsvLine (line: ReadOnlySpan<char>) : float[] =
    // Parse without allocating substrings
    let mutable start = 0
    let values = ResizeArray()
    
    for i = 0 to line.Length - 1 do
        if line.[i] = ',' then
            let field = line.Slice(start, i - start)
            values.Add(parseFloat field)
            start <- i + 1
    
    values.ToArray()
```

---

## Implementation Strategy

### Phase 1: Internal Span Usage

Modify internal operations to use Span<T>:

```fsharp
// Ndarray operations using Span
module NdarrayOps =
    let mapInPlace (f: float -> float) (arr: Span<float>) : unit =
        for i = 0 to arr.Length - 1 do
            arr.[i] <- f arr.[i]
    
    let addInPlace (a: Span<float>) (b: ReadOnlySpan<float>) : unit =
        for i = 0 to a.Length - 1 do
            a.[i] <- a.[i] + b.[i]
```

### Phase 2: Public Span API

Add Span-based methods to Ndarray type:

```fsharp
type Ndarray<'K, 'T> with
    /// Get Span view of entire array (if contiguous)
    member this.AsSpan() : Span<'T> =
        match this with
        | Dense d when d.Offset = 0 && d.Layout = CLayout ->
            Span(d.Data)
        | _ -> failwith "Cannot create Span for non-contiguous array"
    
    /// Get row as Span (only works for C-layout)
    member this.GetRowSpan(row: int) : Span<'T> =
        match this with
        | Dense d when d.Layout = CLayout ->
            let cols = d.Shape.[1]
            let offset = d.Offset + row * cols * d.Strides.[0]
            Span(d.Data, offset, cols)
        | _ -> failwith "GetRowSpan requires C-layout"
```

### Phase 3: Zero-Copy Views

Create view types that don't own memory:

```fsharp
/// View into existing Ndarray without copying
type NdarrayView<'K, 'T> = {
    Source: Ndarray<'K, 'T>
    Offset: int
    Shape: Shape
    Strides: int[]
}

module NdarrayView =
    let slice (view: NdarrayView) (specs: SliceSpec[]) : NdarrayView =
        // Just compute new offset and shape, no copy
        { view with
            Offset = computeNewOffset view specs
            Shape = computeNewShape view specs
            Strides = view.Strides }
    
    let toSpan (view: NdarrayView) : Span<'T> option =
        // Only if contiguous
        if isContiguous view then
            Some (Span(view.Source.Data, view.Offset, numel view.Shape))
        else
            None
```

---

## ArrayPool<T> for Temporary Buffers

### Problem

Operations that need temporary arrays create GC pressure:

```fsharp
let someOperation (a: Ndarray) (b: Ndarray) : Ndarray =
    let temp = Array.zeroCreate 10000  // Allocates every call!
    // ... use temp ...
    result
```

### Solution: ArrayPool

```fsharp
open System.Buffers

let someOperation (a: Ndarray) (b: Ndarray) : Ndarray =
    let pool = ArrayPool.Shared
    let temp = pool.Rent(10000)  // Rent from pool
    try
        // ... use temp ...
        result
    finally
        pool.Return(temp)  // Return to pool
```

### Use Cases in Fowl

1. **LU factorization workspace**
2. **Reduction operations** (sum, mean)
3. **Matrix multiplication temporaries**
4. **AD gradient accumulation**

---

## Performance Considerations

### Span Bounds Checking

Span has bounds checking by default. For hot loops, use unsafe:

```fsharp
// Safe but slower
let sumSpan (data: ReadOnlySpan<float>) : float =
    let mutable s = 0.0
    for i = 0 to data.Length - 1 do
        s <- s + data.[i]  // Bounds check
    s

// Unsafe but faster (requires <AllowUnsafeBlocks>true</AllowUnsafeBlocks>)
let sumSpanUnsafe (data: ReadOnlySpan<float>) : float =
    let mutable s = 0.0
    let ptr = &data.GetPinnableReference()
    for i = 0 to data.Length - 1 do
        s <- s + Unsafe.Add(ptr, i)  // No bounds check
    s
```

### Pinning Overhead

Span keeps arrays pinned (prevents GC from moving them). This has overhead:

```fsharp
// Short Span usage: OK
let process (arr: float[]) : float =
    let span = arr.AsSpan()
    sumSpan span  // Short-lived, low overhead

// Long Span usage: Consider Memory<T>
let processAsync (arr: float[]) : Task<float> =
    let memory = arr.AsMemory()  // Can be used across await points
    task {
        do! Task.Delay(100)
        return sumMemory memory
    }
```

---

## Migration Path for Fowl

### Step 1: Add Span Overloads

Keep existing API, add Span-based methods:

```fsharp
// Existing
let add (a: Ndarray) (b: Ndarray) : Ndarray = ...

// New Span-based
let addInPlace (a: Ndarray) (b: ReadOnlySpan<float>) : unit = ...
```

### Step 2: Optimize Internals

Use Span in internal implementations:

```fsharp
let map (f: float -> float) (arr: Ndarray) : Ndarray =
    let result = Ndarray.create arr.Shape
    let span = arr.AsSpan()
    let resultSpan = result.AsSpan()
    for i = 0 to span.Length - 1 do
        resultSpan.[i] <- f span.[i]
    result
```

### Step 3: Views API

Introduce zero-copy views as new feature:

```fsharp
module Ndarray.View =
    let slice arr specs = ...  // Returns view
    let row arr i = ...        // Returns view
    let column arr j = ...     // Returns view
```

---

## Comparison: Current vs Optimized

### Memory Usage Example

Processing 1000x1000 matrix:

**Current (Copy-heavy):**
```fsharp
let a = Ndarray.zeros [|1000; 1000|]      // 8MB
let b = Ndarray.zeros [|1000; 1000|]      // 8MB
let c = Ndarray.add a b                    // 8MB (new allocation)
let row = c.GetRow 5                       // 8KB (copy)
let col = c.GetColumn 3                    // 8KB (copy)
// Total: ~24MB allocated
```

**Optimized (Span + Views):**
```fsharp
let a = Ndarray.zeros [|1000; 1000|]      // 8MB
let b = Ndarray.zeros [|1000; 1000|]      // 8MB
let c = Ndarray.add a b                    // 8MB (may still allocate)
let row = c.GetRowSpan 5                   // 0 bytes (view)
let col = c.GetColumnView 3                // ~40 bytes (view struct)
// Total: ~16MB + small overhead
```

---

## Open Questions

1. **How to handle non-contiguous arrays with Span?**
   - Option A: Copy to contiguous buffer
   - Option B: Custom iterator (slower)
   - Option C: Don't use Span for non-contiguous

2. **Should views be mutable or read-only?**
   - Mutable: More flexible, risk of aliasing bugs
   - Read-only: Safer, may need copy-on-write

3. **How to integrate with existing slicing?**
   - Keep slice = copy (NumPy compatible)
   - Add view = no copy (new feature)

4. **ArrayPool size tuning?**
   - How to determine optimal pool sizes?
   - Per-operation pools vs shared pool?

---

## References

- [Span<T> Struct](https://docs.microsoft.com/en-us/dotnet/api/system.span-1)
- [Memory<T> Struct](https://docs.microsoft.com/en-us/dotnet/api/system.memory-1)
- [ArrayPool<T>](https://docs.microsoft.com/en-us/dotnet/api/system.buffers.arraypool-1)
- [System.Buffers Namespace](https://docs.microsoft.com/en-us/dotnet/api/system.buffers)
- [.NET Memory Management](https://docs.microsoft.com/en-us/dotnet/standard/garbage-collection/)

---

_Research: Phase 2 of 5 complete. Next: BenchmarkDotNet setup._
