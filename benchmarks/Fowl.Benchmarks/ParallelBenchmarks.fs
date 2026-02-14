module Fowl.Benchmarks.ParallelBenchmarks

open System
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Running
open BenchmarkDotNet.Jobs

// ============================================================================
// Parallel vs Sequential Benchmarks
// ============================================================================

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type ParallelElementWiseBenchmarks() =
    // Test various sizes to show crossover point
    let sizes = [| 1000; 10000; 100000; 1000000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val Size = 100000 with get, set
    
    member val A = Array.empty with get, set
    member val B = Array.empty with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.A <- Array.init this.Size (fun _ -> rng.NextDouble())
        this.B <- Array.init this.Size (fun _ -> rng.NextDouble())
        
        // Print info once
        if this.Size = 1000 then
            printfn "\n=== Parallel Benchmarks ==="
            printfn "Processor Count: %d" Environment.ProcessorCount
            printfn "Parallel Threshold: %d" Fowl.Parallel.getParallelThreshold
    
    [<Benchmark(Baseline = true)>]
    member this.SequentialAdd() =
        let result = Array.zeroCreate this.Size
        for i = 0 to this.Size - 1 do
            result.[i] <- this.A.[i] + this.B.[i]
        result
    
    [<Benchmark>]
    member this.ParallelAdd() =
        Fowl.Parallel.ParallelOps.add this.A this.B
    
    [<Benchmark>]
    member this.ParallelSimdAdd() =
        Fowl.Parallel.ParallelSimdOps.add this.A this.B
    
    [<Benchmark>]
    member this.ParallelSimdAvx2Add() =
        Fowl.Parallel.ParallelSimdOps.addAvx2 this.A this.B
    
    [<Benchmark>]
    member this.SequentialMul() =
        let result = Array.zeroCreate this.Size
        for i = 0 to this.Size - 1 do
            result.[i] <- this.A.[i] * this.B.[i]
        result
    
    [<Benchmark>]
    member this.ParallelMul() =
        Fowl.Parallel.ParallelOps.mul this.A this.B

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type ParallelReductionBenchmarks() =
    let sizes = [| 10000; 100000; 1000000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val Size = 100000 with get, set
    
    member val A = Array.empty with get, set
    member val B = Array.empty with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.A <- Array.init this.Size (fun _ -> rng.NextDouble())
        this.B <- Array.init this.Size (fun _ -> rng.NextDouble())
    
    [<Benchmark(Baseline = true)>]
    member this.SequentialSum() =
        let mutable s = 0.0
        for i = 0 to this.Size - 1 do
            s <- s + this.A.[i]
        s
    
    [<Benchmark>]
    member this.ParallelSum() =
        Fowl.Parallel.ParallelReductions.sum this.A
    
    [<Benchmark>]
    member this.ArraySum() =
        Array.sum this.A
    
    [<Benchmark(Baseline = false)>]
    member this.SequentialDot() =
        let mutable s = 0.0
        for i = 0 to this.Size - 1 do
            s <- s + this.A.[i] * this.B.[i]
        s
    
    [<Benchmark>]
    member this.ParallelDot() =
        Fowl.Parallel.ParallelReductions.dot this.A this.B
    
    [<Benchmark>]
    member this.SequentialMin() =
        let mutable m = this.A.[0]
        for i = 1 to this.Size - 1 do
            if this.A.[i] < m then m <- this.A.[i]
        m
    
    [<Benchmark>]
    member this.ParallelMin() =
        Fowl.Parallel.ParallelReductions.min this.A

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type ParallelMapBenchmarks() =
    let sizes = [| 10000; 100000; 1000000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val Size = 100000 with get, set
    
    member val A = Array.empty with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.A <- Array.init this.Size (fun _ -> rng.NextDouble())
    
    [<Benchmark(Baseline = true)>]
    member this.SequentialMap() =
        Array.map (fun x -> x * x + 1.0) this.A
    
    [<Benchmark>]
    member this.ParallelMap() =
        Fowl.Parallel.ParallelOps.map (fun x -> x * x + 1.0) this.A

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type ThreadSafeRandomBenchmarks() =
    [<Params(1000, 10000, 100000)>]
    member val Count = 10000 with get, set
    
    [<Benchmark(Baseline = true)>]
    member this.SequentialRandom() =
        let rng = Random()
        Array.init this.Count (fun _ -> rng.NextDouble())
    
    [<Benchmark>]
    member this.ThreadSafeRandom() =
        Fowl.Parallel.ThreadSafeRandom.nextDoubleArray this.Count
    
    [<Benchmark>]
    member this.ThreadSafeNormal() =
        Fowl.Parallel.ThreadSafeRandom.nextNormalArray this.Count

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type ParallelMatrixBenchmarks() =
    let sizes = [| 50; 100; 200 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val N = 100 with get, set
    
    member val A = Unchecked.defaultof<float[,]> with get, set
    member val B = Unchecked.defaultof<float[,]> with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.A <- Array2D.init this.N this.N (fun _ _ -> rng.NextDouble())
        this.B <- Array2D.init this.N this.N (fun _ _ -> rng.NextDouble())
    
    [<Benchmark(Baseline = true)>]
    member this.SequentialMatMul() =
        let n = this.N
        let result = Array2D.zeroCreate n n
        for i = 0 to n - 1 do
            for j = 0 to n - 1 do
                let mutable sum = 0.0
                for k = 0 to n - 1 do
                    sum <- sum + this.A.[i, k] * this.B.[k, j]
                result.[i, j] <- sum
        result
    
    [<Benchmark>]
    member this.ParallelMatMul() =
        Fowl.Parallel.ParallelMatrixOps.matmul this.A this.B

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type ThresholdBenchmarks() =
    // Test to find optimal threshold
    let sizes = [| 1000; 5000; 10000; 20000; 50000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val Size = 10000 with get, set
    
    member val A = Array.empty with get, set
    member val B = Array.empty with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.A <- Array.init this.Size (fun _ -> rng.NextDouble())
        this.B <- Array.init this.Size (fun _ -> rng.NextDouble())
    
    [<Benchmark>]
    member this.ParallelWithThreshold1000() =
        Fowl.Parallel.ParallelOps.addWithThreshold 1000 this.A this.B
    
    [<Benchmark>]
    member this.ParallelWithThreshold5000() =
        Fowl.Parallel.ParallelOps.addWithThreshold 5000 this.A this.B
    
    [<Benchmark>]
    member this.ParallelWithThreshold10000() =
        Fowl.Parallel.ParallelOps.addWithThreshold 10000 this.A this.B
    
    [<Benchmark>]
    member this.ParallelWithThreshold20000() =
        Fowl.Parallel.ParallelOps.addWithThreshold 20000 this.A this.B
