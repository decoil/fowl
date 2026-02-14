# Fowl Performance Baseline Results

**Date:** 2026-02-14  
**Hardware:** Apple M3 Pro (to be filled after running)  
**OS:** macOS (to be filled after running)  
**.NET Version:** 8.0 (to be filled after running)  
**Fowl Commit:** a59393b

---

## Benchmark Environment

```
BenchmarkDotNet v0.13.12
CPU: (to be filled)
Cores: (to be filled)
Frequency: (to be filled)
Runtime: .NET 8.0.0
GC: (to be filled)
Jit: (to be filled)
```

---

## 1. Element-wise Operation Benchmarks

Measures raw array operations performance.

### 1.1 Addition Performance

| Method | Size | Mean | Error | StdDev | Gen0 | Gen1 | Gen2 | Allocated |
|--------|------|------|-------|--------|------|------|------|-----------|
| ScalarAdd | 100 | (to be filled) | | | | | | |
| ScalarAdd | 1,000 | (to be filled) | | | | | | |
| ScalarAdd | 10,000 | (to be filled) | | | | | | |
| ScalarAdd | 100,000 | (to be filled) | | | | | | |
| ScalarAdd | 1,000,000 | (to be filled) | | | | | | |
| ArrayMap2Add | 100 | (to be filled) | | | | | | |
| ArrayMap2Add | 1,000 | (to be filled) | | | | | | |
| ArrayMap2Add | 10,000 | (to be filled) | | | | | | |
| ArrayMap2Add | 100,000 | (to be filled) | | | | | | |
| ArrayMap2Add | 1,000,000 | (to be filled) | | | | | | |

**Key Metrics:**
- Baseline (ScalarAdd) throughput: ___ million elements/second
- Array.map2 overhead: ___%
- Allocation per operation: ___ bytes

### 1.2 Multiplication Performance

| Method | Size | Mean | Error | StdDev | Gen0 | Gen1 | Gen2 | Allocated |
|--------|------|------|-------|--------|------|------|------|-----------|
| ScalarMul | 1,000,000 | (to be filled) | | | | | | |
| ArrayMap2Mul | 1,000,000 | (to be filled) | | | | | | |

---

## 2. Reduction Operation Benchmarks

Measures sum, mean, and fold operations.

### 2.1 Sum Performance

| Method | Size | Mean | Error | StdDev | Gen0 | Gen1 | Gen2 | Allocated |
|--------|------|------|-------|--------|------|------|------|-----------|
| ScalarSum | 1,000 | (to be filled) | | | | | | |
| ScalarSum | 10,000 | (to be filled) | | | | | | |
| ScalarSum | 100,000 | (to be filled) | | | | | | |
| ScalarSum | 1,000,000 | (to be filled) | | | | | | |
| ArraySum | 1,000 | (to be filled) | | | | | | |
| ArraySum | 1,000,000 | (to be filled) | | | | | | |

**Key Metrics:**
- Sum throughput: ___ million elements/second
- Memory allocation: ___ bytes

### 2.2 Mean Performance

| Method | Size | Mean | Error | StdDev | Allocated |
|--------|------|------|-------|--------|-----------|
| ScalarMean | 1,000,000 | (to be filled) | | | |
| FoldMean | 1,000,000 | (to be filled) | | | |

---

## 3. Ndarray Operation Benchmarks

Measures Fowl.Core.Ndarray abstraction overhead.

### 3.1 Element-wise Operations

| Method | Size | Mean | Error | StdDev | Gen0 | Gen1 | Gen2 | Allocated |
|--------|------|------|-------|--------|------|------|------|-----------|
| NdarrayAdd | 1,000 | (to be filled) | | | | | | |
| NdarrayMul | 1,000 | (to be filled) | | | | | | |
| NdarrayMap | 1,000 | (to be filled) | | | | | | |
| NdarrayFold | 1,000 | (to be filled) | | | | | | |

**Key Metrics:**
- Abstraction overhead vs raw arrays: ___%
- Allocation per Ndarray operation: ___ bytes

---

## 4. Matrix Operation Benchmarks

Measures matrix algebra performance.

### 4.1 Transpose

| Size | Mean | Error | StdDev | Allocated |
|------|------|-------|--------|-----------|
| 10x10 | (to be filled) | | | |
| 50x50 | (to be filled) | | | |
| 100x100 | (to be filled) | | | |
| 200x200 | (to be filled) | | | |

### 4.2 Matrix Multiplication

| Size | Mean | Error | StdDev | Allocated |
|------|------|-------|--------|-----------|
| 10x10 | (to be filled) | | | |
| 50x50 | (to be filled) | | | |
| 100x100 | (to be filled) | | | |
| 200x200 | (to be filled) | | | |

**Key Metrics:**
- 100x100 matmul: ___ ms
- FLOPS: ___ GFLOPS

---

## 5. Statistics Benchmarks

Measures statistical function performance.

### 5.1 Descriptive Statistics

| Method | Size | Mean | Error | StdDev | Allocated |
|--------|------|------|-------|--------|-----------|
| Mean | 1,000 | (to be filled) | | | |
| Mean | 10,000 | (to be filled) | | | |
| Mean | 100,000 | (to be filled) | | | |
| Variance | 100,000 | (to be filled) | | | |
| StdDev | 100,000 | (to be filled) | | | |

---

## 6. Algorithmic Differentiation Benchmarks

Measures AD operation overhead.

### 6.1 Forward Mode

| Method | Mean | Error | StdDev | Allocated |
|--------|------|-------|--------|-----------|
| DiffSin | (to be filled) | | | |
| DiffPolynomial | (to be filled) | | | |

### 6.2 Reverse Mode

| Method | Mean | Error | StdDev | Allocated |
|--------|------|-------|--------|-----------|
| GradSin | (to be filled) | | | |

### 6.3 Higher Order

| Method | Mean | Error | StdDev | Allocated |
|--------|------|-------|--------|-----------|
| HessianPolynomial | (to be filled) | | | |

**Key Metrics:**
- Forward mode overhead vs manual derivative: ___%
- Reverse mode overhead: ___%

---

## Summary Statistics

### Top 5 Hot Paths (by time)

1. (to be filled after benchmarks)
2. (to be filled after benchmarks)
3. (to be filled after benchmarks)
4. (to be filled after benchmarks)
5. (to be filled after benchmarks)

### Memory Allocation Hotspots

1. (to be filled after benchmarks)
2. (to be filled after benchmarks)
3. (to be filled after benchmarks)

### Optimization Opportunities

Based on baseline measurements:

1. **Element-wise operations**: (to be filled)
2. **Matrix operations**: (to be filled)
3. **Reductions**: (to be filled)
4. **Memory allocations**: (to be filled)

---

## Next Steps

1. **Phase 2**: Implement Vector<T> SIMD for element-wise operations
2. **Phase 3**: Add AVX2 hardware-specific optimizations
3. **Phase 4**: Integrate Span<T> for zero-copy operations
4. **Phase 5**: Add parallelization for large arrays

---

## Raw Results

Full BenchmarkDotNet output:

```
(To be appended after running benchmarks)
```

---

*This is a template. Run benchmarks and fill in actual measurements.*
