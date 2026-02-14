# Fowl Code Audit & Tech Debt Report

**Date:** 2026-02-14
**Auditor:** Fowl (self-audit)
**Scope:** All F# modules (Core, Native, Linalg, Stats)

---

## Summary

**Lines of Code:** ~7,000 F#
**Modules:** 6
**Issues Found:** 15 (mix of critical, warnings, and improvements)
**Overall Quality:** Good foundation, needs refinement for production

---

## Critical Issues (Must Fix)

### 1. Error Handling - Using `failwith` Instead of Result Types

**Location:** Throughout codebase (Distributions.fs, Factorizations.fs, Slice.fs)

**Problem:** Using `failwith` for error cases instead of F#'s `Result` type makes error handling unpredictable and hard to compose.

```fsharp
// Current (bad)
let pdf (mu: float) (sigma: float) (x: float) : float =
    if sigma <= 0.0 then failwith "sigma must be positive"
    ...

// Should be (good)
let pdf (mu: float) (sigma: float) (x: float) : Result<float, string> =
    if sigma <= 0.0 then Error "sigma must be positive"
    else Ok ...
```

**Impact:** Production code will crash on invalid inputs instead of gracefully handling errors.

**Fix:** Replace all `failwith` in public APIs with `Result` types or create try-prefixed versions.

### 2. Random Number Generator - Creating New Instance Per Call

**Location:** Distributions.fs (rvs functions)

**Problem:** Each call to `rvs` creates a new `Random()` instance, which can produce correlated values if called rapidly.

```fsharp
// Current (bad)
let rvs ... =
    let rng = Random()  // New instance every call!
    ...

// Should be (good)
// Either pass rng as parameter or use thread-safe shared instance
```

**Impact:** Statistical properties compromised, especially in parallel scenarios.

**Fix:** Add optional RNG parameter or use functional random state passing.

### 3. Native Library Loading - No Fallback Strategy

**Location:** Native/Blas.fs, Linalg/Factorizations.fs

**Problem:** Direct P/Invoke to "libopenblas" with no error handling if library missing.

```fsharp
[<DllImport("libopenblas", ...)>]
extern ...  // Will crash if lib not found
```

**Impact:** Application crashes on startup if OpenBLAS not installed.

**Fix:** Add try-catch around native calls, provide managed fallback, or check library availability.

---

## Warnings & Code Smells

### 4. Mutable State in Pure Functions

**Location:** Various (Shape.stridesC, broadcastTo, etc.)

**Problem:** Using mutable variables in functions that should be pure.

```fsharp
// Current
let stridesC (shape: Shape) =
    let mutable stride = 1  // Mutable!
    for i = n - 1 downto 0 do
        s.[i] <- stride
        stride <- stride * shape.[i]
```

**Fix:** This is acceptable for performance, but document with comments. Consider `Array.scan` for more functional style.

### 5. Missing Input Validation

**Location:** Many functions

**Problem:** Functions don't validate inputs before processing.

**Examples:**
- `linspace` with `num < 2` only has `failwith`
- `percentile` validates but others don't
- `reshape` validates size match but could be clearer

**Fix:** Add consistent validation with helpful error messages.

### 6. Non-Idiomatic F# Patterns

#### 6a. Using ref cells instead of functional patterns

**Location:** Slice.fs

```fsharp
// Current
let outIdx = ref 0
iterate ... (fun ... ->
    outData.[!outIdx] <- ...
    outIdx := !outIdx + 1)
```

**Better:** Use `Array.mapi` or recursive function with accumulator.

#### 6b. Explicit type annotations where inference works

**Location:** Throughout

Many explicit type annotations are unnecessary and clutter code. F# has excellent type inference.

### 7. Inconsistent API Design

**Problem:** Some functions take curried args, others don't consistently.

```fsharp
// Curried - good
let add (a: Ndarray<...>) (b: Ndarray<...>) = ...

// But then some have optional params at wrong position
let var (?ddof: int) (arr: Ndarray<...>) = ...
// Should be: let var (arr: ...) (?ddof: int) for piping
```

**Fix:** Follow F# convention: data last for pipelining.

### 8. Missing Documentation Comments

**Location:** All modules

**Problem:** Most functions lack XML doc comments (`///`).

**Fix:** Add comprehensive documentation for public APIs.

### 9. Test Coverage Gaps

**Current:** 12 tests for Core only
**Missing:**
- Native/BLAS tests
- Linalg tests
- Stats tests
- Edge cases (empty arrays, single elements, large values)
- Property-based tests (FsCheck)

---

## Architecture & Design Issues

### 10. Dense/Sparse Split Not Fully Explored

**Problem:** Many functions just `failwith` for Sparse arrays.

**Options:**
- Remove Sparse for now (YAGNI)
- Implement proper sparse support
- Make Ndarray abstract with type provider

### 11. Phantom Types Not Fully Leveraged

**Current:** Phantom types defined but not used for type-safe operations.

**Opportunity:** Could prevent mixing Float32/Float64 arrays in operations.

```fsharp
// Could prevent this at compile time:
let a = Ndarray.zeros<Float32> [|3|]
let b = Ndarray.zeros<Float64> [|3|]
Ndarray.add a b  // Currently fails at runtime, could be compile-time
```

### 12. No Type Providers for Data Loading

**Missing:** F# type providers for CSV, HDF5, etc.

### 13. Slicing API Verbose

**Current:**
```fsharp
Fowl.Core.Slice.slice arr [|SliceSpec.Range(Some 1, Some 4, None); SliceSpec.All|]
```

**Could be:**
```fsharp
arr.[1..4, *]  // With custom indexing operators
```

---

## Performance Issues

### 14. Array Copying Excessive

**Location:** Slice.fs, Matrix operations

**Problem:** Many operations create copies when views would suffice.

**Impact:** Memory pressure on large arrays.

**Fix:** Consider using `Span<'T>` or memory slices for zero-copy operations.

### 15. No SIMD Vectorization in Managed Code

**Location:** Element-wise operations

**Problem:** `Array.map2`, `Array.map` don't use SIMD.

**Fix:** Either:
- Use System.Runtime.Intrinsics
- Offload everything to native BLAS
- Use existing SIMD-enabled libraries (Accord.NET, etc.)

---

## Recommendations by Priority

### Immediate (Block AD Implementation)
1. ✅ Add Result types for error handling
2. ✅ Fix Random instance sharing
3. ✅ Add native library availability checks
4. ✅ Add comprehensive tests for existing code

### Short-term (Before Release)
5. ✅ Standardize API conventions (data-last)
6. ✅ Add XML documentation
7. ✅ Implement property-based testing
8. ✅ Add benchmarks

### Medium-term
9. Consider Span<'T> for views
10. Add type providers
11. Optimize SIMD usage
12. Complete sparse array support

---

## Idiomatic F# Improvements

### From Essential F# Reading

1. **Discriminated Unions for Domain Modeling**
   - Use DUs instead of booleans for state
   - Make illegal states unrepresentable

2. **Single-Case DUs for Type Safety**
   ```fsharp
   type PositiveFloat = PositiveFloat of float
   type NonEmptyShape = NonEmptyShape of Shape
   ```

3. **Computation Expressions**
   - Could simplify error handling
   - Useful for AD chain rule composition

4. **Active Patterns**
   - Already using for Matrix/Vector
   - Could extend for more shape checking

5. **Railway-Oriented Programming**
   - Result types for error handling
   - Composition with `bind` and `map`

### From Stylish F# Reading

1. **Type-First Design**
   - Sketch signatures before implementation
   - Use types to make errors impossible

2. **Function Composition**
   - Use `|>` for readability
   - Curried functions for partial application

3. **Immutability**
   - Minimize mutable state
   - Use copy-and-update for records

---

## Action Plan

### Phase 1: Stability (Do First)
- [ ] Add Result-wrapped versions of all public APIs
- [ ] Fix Random state management
- [ ] Add native library detection
- [ ] Write comprehensive test suite

### Phase 2: Idiomatic Refinement
- [ ] Standardize API conventions
- [ ] Add XML documentation
- [ ] Implement property-based tests
- [ ] Refactor mutable code where possible

### Phase 3: Performance
- [ ] Add benchmarks
- [ ] Profile and optimize hot paths
- [ ] Consider Span<'T> adoption

### Phase 4: Features
- [ ] Complete sparse support
- [ ] Add type providers
- [ ] SIMD optimization

---

## Conclusion

The Fowl codebase has a solid foundation with good architectural decisions (phantom types, discriminated unions). However, it needs:

1. **Better error handling** before production use
2. **More comprehensive testing** for confidence
3. **Documentation** for usability
4. **API refinements** for F# idioms

**Recommendation:** Spend 1-2 days on Phase 1 fixes before implementing AD. This will make AD implementation more robust and the codebase production-ready.

---

*Audit completed. Ready to begin fixes.*
