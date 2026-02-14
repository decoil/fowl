# Fowl Benchmark Suite

Benchmarks for measuring Fowl performance using BenchmarkDotNet.

## Running Benchmarks

```bash
cd benchmarks/Fowl.Benchmarks
dotnet run -c Release
```

**Important**: Always run in `Release` mode for accurate measurements.

## Benchmark Categories

### 1. ElementWiseBenchmarks

Measures raw array operations vs F# Array functions:
- Scalar loops (baseline)
- Array.map2 operations
- Addition and multiplication

**What to measure**: SIMD opportunity - scalar loops vs vectorized

### 2. ReductionBenchmarks

Measures sum, mean, and other reduction operations:
- Scalar accumulation
- Array.sum
- Fold operations

**What to measure**: Horizontal operation efficiency

### 3. NdarrayBenchmarks

Measures Fowl.Core.Ndarray operations:
- Ndarray.add, Ndarray.mul
- Ndarray.map
- Ndarray.fold

**What to measure**: Abstraction overhead vs raw arrays

### 4. MatrixBenchmarks

Measures matrix operations:
- Matrix.transpose
- Matrix.matmul

**What to measure**: Cache efficiency, BLAS vs managed

### 5. StatsBenchmarks

Measures statistical operations:
- Descriptive.mean
- Descriptive.var
- Descriptive.std

**What to measure**: Algorithm efficiency

### 6. ADBenchmarks

Measures algorithmic differentiation:
- AD.diff (forward mode)
- AD.grad (reverse mode)
- AD.hessian

**What to measure**: Dual number overhead

## Interpreting Results

### Key Metrics

1. **Mean**: Average execution time
2. **Error**: 99.9% confidence interval
3. **StdDev**: Standard deviation
4. **Gen0/Gen1/Gen2**: Garbage collection counts
5. **Allocated**: Bytes allocated per operation

### What to Look For

**Good signs:**
- Low allocation (ideally 0 B for hot loops)
- Linear scaling with data size
- Consistent performance across runs

**Bad signs:**
- High allocation (indicates copying)
- Superlinear scaling (cache thrashing)
- High variance (GC pressure or cache issues)

## Baseline Measurements

Run these before any optimization work:

```bash
# Save baseline
dotnet run -c Release -- --artifacts ./baseline

# After optimization, compare
dotnet run -c Release -- --artifacts ./optimized
```

## Comparison with Other Libraries

Future work:
- [ ] Add Math.NET Numerics comparison
- [ ] Add Accord.NET comparison
- [ ] Add NumPy comparison (via Python.NET)

## Hardware Information

BenchmarkDotNet automatically detects:
- CPU model and frequency
- Cache sizes (L1, L2, L3)
- Memory size
- OS and .NET version

Include this information when reporting performance issues.

## Profiling Hot Paths

To identify optimization opportunities:

1. Run benchmarks to find slow operations
2. Use dotTrace or Visual Studio profiler
3. Look for:
   - Cache misses (memory-bound)
   - Branch mispredictions
   - GC pauses
   - Method call overhead

## Expected Performance

### Current (Managed, No SIMD)

| Operation | 1M elements | Expected |
|-----------|-------------|----------|
| Add | ~5-10 ms | Baseline |
| Multiply | ~5-10 ms | Baseline |
| Sum | ~2-5 ms | Baseline |
| MatMul (100x100) | ~10-20 ms | Baseline |

### After SIMD Optimization

| Operation | 1M elements | Expected Speedup |
|-----------|-------------|------------------|
| Add | ~1-2 ms | 4-8x |
| Multiply | ~1-2 ms | 4-8x |
| Sum | ~1-3 ms | 2-4x |
| MatMul (100x100) | ~5-10 ms | 2-4x |

**Note**: Actual results depend on hardware (AVX2 vs AVX-512).

## Continuous Benchmarking

Consider running benchmarks:
- Before/after optimization PRs
- Weekly to catch regressions
- On different hardware (x64, ARM64)

## References

- [BenchmarkDotNet Documentation](https://benchmarkdotnet.org/)
- [.NET Performance Guidelines](https://docs.microsoft.com/en-us/dotnet/framework/performance/)
- [Writing High-Performance .NET Code](https://www.writinghighperf.net/)
