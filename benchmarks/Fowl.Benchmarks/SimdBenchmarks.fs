module Fowl.Benchmarks.SimdBenchmarks

open System
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Running
open BenchmarkDotNet.Jobs

// ============================================================================
// SIMD vs Scalar Benchmarks
// ============================================================================

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type Avx2Benchmarks() =
    let sizes = [| 1000; 10000; 100000; 1000000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val Size = 10000 with get, set
    
    member val A = Array.empty with get, set
    member val B = Array.empty with get, set
    member val Result = Array.empty with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.A <- Array.init this.Size (fun _ -> rng.NextDouble())
        this.B <- Array.init this.Size (fun _ -> rng.NextDouble())
        this.Result <- Array.zeroCreate this.Size
        
        // Print SIMD info once
        if this.Size = 1000 then
            printfn "\n=== AVX2 Benchmarks ==="
            printfn "AVX2 Supported: %b" Fowl.Native.SIMD.KernelSelector.IsAvx2Supported
            printfn "SSE2 Supported: %b" Fowl.Native.SIMD.KernelSelector.IsSse2Supported
            printfn "Best Implementation: %s" Fowl.Native.SIMD.KernelSelector.BestImplementation
    
    [<Benchmark(Baseline = true)>]
    member this.ScalarAdd() =
        for i = 0 to this.Size - 1 do
            this.Result.[i] <- this.A.[i] + this.B.[i]
        this.Result
    
    [<Benchmark>]
    member this.VectorTAdd() =
        Fowl.SIMD.ElementWise.add this.A this.B
    
    [<Benchmark>]
    member this.Avx2Add() =
        Fowl.Native.SIMD.KernelSelector.Add(this.A, this.B, this.Result)
        this.Result
    
    [<Benchmark>]
    member this.ScalarSum() =
        let mutable s = 0.0
        for i = 0 to this.Size - 1 do
            s <- s + this.A.[i]
        s
    
    [<Benchmark>]
    member this.VectorTSum() =
        Fowl.SIMD.Reductions.sum this.A
    
    [<Benchmark>]
    member this.Avx2Sum() =
        Fowl.Native.SIMD.KernelSelector.Sum(this.A)
    
    [<Benchmark>]
    member this.ScalarDot() =
        let mutable s = 0.0
        for i = 0 to this.Size - 1 do
            s <- s + this.A.[i] * this.B.[i]
        s
    
    [<Benchmark>]
    member this.VectorTDot() =
        Fowl.SIMD.Reductions.dot this.A this.B
    
    [<Benchmark>]
    member this.Avx2Dot() =
        Fowl.Native.SIMD.KernelSelector.Dot(this.A, this.B)

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type SimdElementWiseBenchmarks() =
    let sizes = [| 100; 1000; 10000; 100000; 1000000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val Size = 10000 with get, set
    
    member val A = Array.empty with get, set
    member val B = Array.empty with get, set
    member val Result = Array.empty with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.A <- Array.init this.Size (fun _ -> rng.NextDouble())
        this.B <- Array.init this.Size (fun _ -> rng.NextDouble())
        this.Result <- Array.zeroCreate this.Size
    
    [<Benchmark(Baseline = true)>]
    member this.ScalarAdd() =
        for i = 0 to this.Size - 1 do
            this.Result.[i] <- this.A.[i] + this.B.[i]
        this.Result
    
    [<Benchmark>]
    member this.ArrayMap2Add() =
        Array.map2 (+) this.A this.B
    
    [<Benchmark>]
    member this.SimdAdd() =
        Fowl.SIMD.ElementWise.add this.A this.B
    
    [<Benchmark>]
    member this.SimdAddInPlace() =
        Array.Copy(this.A, this.Result, this.Size)
        Fowl.SIMD.ElementWise.addInPlace this.Result this.B
    
    [<Benchmark>]
    member this.ScalarMul() =
        for i = 0 to this.Size - 1 do
            this.Result.[i] <- this.A.[i] * this.B.[i]
        this.Result
    
    [<Benchmark>]
    member this.SimdMul() =
        Fowl.SIMD.ElementWise.mul this.A this.B
    
    [<Benchmark>]
    member this.ScalarSub() =
        for i = 0 to this.Size - 1 do
            this.Result.[i] <- this.A.[i] - this.B.[i]
        this.Result
    
    [<Benchmark>]
    member this.SimdSub() =
        Fowl.SIMD.ElementWise.sub this.A this.B

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type SimdReductionBenchmarks() =
    let sizes = [| 1000; 10000; 100000; 1000000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val Size = 10000 with get, set
    
    member val A = Array.empty with get, set
    member val B = Array.empty with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.A <- Array.init this.Size (fun _ -> rng.NextDouble())
        this.B <- Array.init this.Size (fun _ -> rng.NextDouble())
    
    [<Benchmark(Baseline = true)>]
    member this.ScalarSum() =
        let mutable s = 0.0
        for i = 0 to this.Size - 1 do
            s <- s + this.A.[i]
        s
    
    [<Benchmark>]
    member this.ArraySum() =
        Array.sum this.A
    
    [<Benchmark>]
    member this.SimdSum() =
        Fowl.SIMD.Reductions.sum this.A
    
    [<Benchmark(Baseline = false)>]
    member this.ScalarDot() =
        let mutable s = 0.0
        for i = 0 to this.Size - 1 do
            s <- s + this.A.[i] * this.B.[i]
        s
    
    [<Benchmark>]
    member this.SimdDot() =
        Fowl.SIMD.Reductions.dot this.A this.B
    
    [<Benchmark>]
    member this.ScalarMean() =
        let mutable s = 0.0
        for i = 0 to this.Size - 1 do
            s <- s + this.A.[i]
        s / float this.Size
    
    [<Benchmark>]
    member this.SimdMean() =
        Fowl.SIMD.Reductions.mean this.A

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type SimdScalarBenchmarks() =
    let sizes = [| 1000; 10000; 100000; 1000000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val Size = 10000 with get, set
    
    member val A = Array.empty with get, set
    member val Result = Array.empty with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.A <- Array.init this.Size (fun _ -> rng.NextDouble())
        this.Result <- Array.zeroCreate this.Size
    
    [<Benchmark(Baseline = true)>]
    member this.ScalarAddScalar() =
        for i = 0 to this.Size - 1 do
            this.Result.[i] <- this.A.[i] + 5.0
        this.Result
    
    [<Benchmark>]
    member this.SimdAddScalar() =
        Fowl.SIMD.ElementWise.addScalar this.A 5.0
    
    [<Benchmark>]
    member this.ScalarMulScalar() =
        for i = 0 to this.Size - 1 do
            this.Result.[i] <- this.A.[i] * 2.5
        this.Result
    
    [<Benchmark>]
    member this.SimdMulScalar() =
        Fowl.SIMD.ElementWise.mulScalar this.A 2.5
    
    [<Benchmark>]
    member this.SimdMulInPlace() =
        Array.Copy(this.A, this.Result, this.Size)
        Fowl.SIMD.ElementWise.mulInPlace this.Result 2.5
