# Benchmarking Guide

## Overview

This guide explains how to run benchmarks and establish baseline measurements for Fowl performance optimization.

## Quick Start

```bash
cd benchmarks/Fowl.Benchmarks
./run-benchmarks.sh --release
```

Or manually:

```bash
cd benchmarks/Fowl.Benchmarks
dotnet run -c Release
```

## Important: Release Mode

**Always run benchmarks in Release mode!**

Debug builds include:
- No JIT optimizations
- Extra bounds checking
- Debug symbols overhead

This can make code 10-100x slower than Release.

## Benchmark Categories

### 1. ElementWiseBenchmarks

**What it measures:** Raw array operation performance

**Methods:**
- `ScalarAdd` - Simple for-loop addition (baseline)
- `ScalarMul` - Simple for-loop multiplication
- `ArrayMap2Add` - F# Array.map2 addition
- `ArrayMap2Mul` - F# Array.map2 multiplication

**Sizes tested:** 100, 1K, 10K, 100K, 1M elements

**What to look for:**
- Throughput (elements/second)
- Array.map2 overhead vs scalar loops
- Memory allocations

### 2. ReductionBenchmarks

**What it measures:** Sum, mean, and fold operations

**Methods:**
- `ScalarSum` - Manual accumulation
- `ArraySum` - F# Array.sum
- `ScalarMean` - Manual mean calculation
- `FoldMean` - F# Array.fold

**What to look for:**
- Reduction throughput
- Fold vs explicit loop performance
- Memory efficiency

### 3. NdarrayBenchmarks

**What it measures:** Fowl.Core.Ndarray abstraction overhead

**Methods:**
- `NdarrayAdd` - Ndarray.add operation
- `NdarrayMul` - Ndarray.mul operation
- `NdarrayMap` - Ndarray.map operation
- `NdarrayFold` - Ndarray.fold operation

**What to look for:**
- Abstraction cost vs raw arrays
- Result type overhead
- Memory allocations per operation

### 4. MatrixBenchmarks

**What it measures:** Matrix algebra performance

**Methods:**
- `MatrixTranspose` - Matrix transpose
- `MatrixMul` - Matrix multiplication

**Sizes tested:** 10x10, 50x50, 100x100, 200x200

**What to look for:**
- FLOPS (floating point ops/second)
- Cache efficiency
- Scaling with matrix size

### 5. StatsBenchmarks

**What it measures:** Statistical function performance

**Methods:**
- `Mean` - Descriptive.mean
- `Variance` - Descriptive.var
- `StdDev` - Descriptive.std

**What to look for:**
- Algorithm efficiency
- Multiple passes vs single pass
- Memory usage

### 6. ADBenchmarks

**What it measures:** Algorithmic differentiation overhead

**Methods:**
- `DiffSin` - Forward mode derivative
- `DiffPolynomial` - Forward mode polynomial
- `GradSin` - Reverse mode gradient
- `HessianPolynomial` - Second derivative

**What to look for:**
- AD overhead vs manual derivatives
- Forward vs reverse mode cost
- Memory allocations in AD graph

## Interpreting Results

### Key Metrics

1. **Mean** - Average execution time (lower is better)
2. **Error** - 99.9% confidence interval
3. **StdDev** - Standard deviation (lower is more consistent)
4. **Gen0/Gen1/Gen2** - Garbage collections (lower is better)
5. **Allocated** - Bytes allocated per operation (0 B is ideal)

### Example Output

```
|    Method |    Size |      Mean |     Error |    StdDev |    Gen0 | Allocated |
|---------- |-------- |----------:|----------:|----------:|--------:|----------:|
| ScalarAdd |     100 |  20.45 ns |  0.412 ns |  0.385 ns |       - |       - |
| ScalarAdd |    1000 | 204.12 ns |  4.089 ns |  4.824 ns |       - |       - |
```

**Interpretation:**
- 1000 elements processed in ~204 ns
- ~4.9 elements/ns = ~4.9 billion elements/second
- 0 bytes allocated (working with existing arrays)

### Comparing Results

BenchmarkDotNet shows relative performance:

```
|      Method | Ratio | RatioSD |
|------------ |------:|--------:|
| ScalarAdd   |  1.00 |    0.00 |
| ArrayMap2Add|  1.35 |    0.05 |
```

ArrayMap2 is 1.35x slower than ScalarAdd (35% overhead).

## Hardware Considerations

### CPU Features Affecting Performance

1. **AVX2 Support** - 256-bit SIMD operations
2. **AVX-512 Support** - 512-bit SIMD operations
3. **Cache Size** - L1, L2, L3 cache sizes
4. **Memory Bandwidth** - Affects large array operations
5. **Core Count** - Affects parallel operations

### Getting CPU Info

```bash
# macOS
sysctl -a | grep machdep.cpu

# Linux
lscpu

# Windows (PowerShell)
Get-WmiObject -Class Win32_Processor
```

## Baseline Checklist

Before starting optimizations:

- [ ] Run benchmarks in Release mode
- [ ] Document hardware configuration
- [ ] Record .NET version
- [ ] Note Fowl commit hash
- [ ] Capture all 6 benchmark categories
- [ ] Fill in BASELINE_RESULTS.md template
- [ ] Identify top 5 hot paths
- [ ] Document memory allocation hotspots

## After Optimization

1. Re-run benchmarks with same parameters
2. Compare before/after results
3. Calculate speedup ratios
4. Document any regressions
5. Update BASELINE_RESULTS.md

## Troubleshooting

### Benchmarks won't compile

```bash
# Clean and rebuild
dotnet clean
dotnet build -c Release
```

### Benchmarks run too long

BenchmarkDotNet runs multiple iterations for statistical significance. Expected run time: 10-30 minutes.

To run specific benchmarks only:

```bash
dotnet run -c Release -- --filter "*ElementWise*"
```

### Out of memory

Reduce max size in benchmark parameters, or run categories separately.

### Inconsistent results

- Close other applications
- Disable CPU frequency scaling
- Run on plugged-in power (not battery)
- Run multiple times and average

## References

- [BenchmarkDotNet Documentation](https://benchmarkdotnet.org/)
- [.NET Performance Guidelines](https://docs.microsoft.com/en-us/dotnet/framework/performance/)
- [Fowl IMPLEMENTATION_PLAN.md](../docs/IMPLEMENTATION_PLAN.md)
