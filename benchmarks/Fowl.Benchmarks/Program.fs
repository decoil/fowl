module Fowl.Benchmarks.Program

open System
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Running
open BenchmarkDotNet.Jobs
open Fowl
open Fowl.Core

// ============================================================================
// Element-wise Operation Benchmarks
// ============================================================================

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type ElementWiseBenchmarks() =
    let sizes = [| 100; 1000; 10000; 100000; 1000000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val Size = 1000 with get, set
    
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
    member this.ScalarMul() =
        for i = 0 to this.Size - 1 do
            this.Result.[i] <- this.A.[i] * this.B.[i]
        this.Result
    
    [<Benchmark>]
    member this.ArrayMap2Add() =
        Array.map2 (+) this.A this.B
    
    [<Benchmark>]
    member this.ArrayMap2Mul() =
        Array.map2 (*) this.A this.B

// ============================================================================
// Reduction Benchmarks
// ============================================================================

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type ReductionBenchmarks() =
    let sizes = [| 1000; 10000; 100000; 1000000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val Size = 10000 with get, set
    member val Data = Array.empty with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.Data <- Array.init this.Size (fun _ -> rng.NextDouble())
    
    [<Benchmark(Baseline = true)>]
    member this.ScalarSum() =
        let mutable sum = 0.0
        for i = 0 to this.Size - 1 do
            sum <- sum + this.Data.[i]
        sum
    
    [<Benchmark>]
    member this.ArraySum() =
        Array.sum this.Data
    
    [<Benchmark>]
    member this.ScalarMean() =
        let mutable sum = 0.0
        for i = 0 to this.Size - 1 do
            sum <- sum + this.Data.[i]
        sum / float this.Size
    
    [<Benchmark>]
    member this.FoldMean() =
        Array.fold (+) 0.0 this.Data / float this.Size

// ============================================================================
// Ndarray Operation Benchmarks
// ============================================================================

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type NdarrayBenchmarks() =
    let sizes = [| 100; 1000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val Size = 1000 with get, set
    
    member val A = Unchecked.defaultof<Ndarray<Float64, float>> with get, set
    member val B = Unchecked.defaultof<Ndarray<Float64, float>> with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        let shape = [||this.Size|]
        match Ndarray.ofArray (Array.init this.Size (fun _ -> rng.NextDouble())) shape with
        | Ok arr -> this.A <- arr
        | Error _ -> failwith "Setup failed"
        
        match Ndarray.ofArray (Array.init this.Size (fun _ -> rng.NextDouble())) shape with
        | Ok arr -> this.B <- arr
        | Error _ -> failwith "Setup failed"
    
    [<Benchmark>]
    member this.NdarrayAdd() =
        match Ndarray.add this.A this.B with
        | Ok result -> result
        | Error _ -> failwith "Add failed"
    
    [<Benchmark>]
    member this.NdarrayMul() =
        match Ndarray.mul this.A this.B with
        | Ok result -> result
        | Error _ -> failwith "Mul failed"
    
    [<Benchmark>]
    member this.NdarrayMap() =
        Ndarray.map (fun x -> x * 2.0) this.A
    
    [<Benchmark>]
    member this.NdarrayFold() =
        Ndarray.fold (fun s x -> s + x) 0.0 this.A

// ============================================================================
// Matrix Operation Benchmarks
// ============================================================================

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type MatrixBenchmarks() =
    let sizes = [| 10; 50; 100; 200 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val N = 100 with get, set
    
    member val A = Unchecked.defaultof<Ndarray<Float64, float>> with get, set
    member val B = Unchecked.defaultof<Ndarray<Float64, float>> with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        let shape = [||this.N; this.N|]
        
        let dataA = Array.init (this.N * this.N) (fun _ -> rng.NextDouble())
        match Ndarray.ofArray dataA shape with
        | Ok arr -> this.A <- arr
        | Error _ -> failwith "Setup failed"
        
        let dataB = Array.init (this.N * this.N) (fun _ -> rng.NextDouble())
        match Ndarray.ofArray dataB shape with
        | Ok arr -> this.B <- arr
        | Error _ -> failwith "Setup failed"
    
    [<Benchmark>]
    member this.MatrixTranspose() =
        match Matrix.transpose this.A with
        | Ok result -> result
        | Error _ -> failwith "Transpose failed"
    
    [<Benchmark>]
    member this.MatrixMul() =
        match Matrix.matmul this.A this.B with
        | Ok result -> result
        | Error _ -> failwith "Matmul failed"

// ============================================================================
// Statistics Benchmarks
// ============================================================================

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type StatsBenchmarks() =
    let sizes = [| 1000; 10000; 100000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val Size = 10000 with get, set
    
    member val Data = Unchecked.defaultof<Ndarray<Float64, float>> with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        let data = Array.init this.Size (fun _ -> rng.NextDouble())
        match Ndarray.ofArray data [||this.Size|] with
        | Ok arr -> this.Data <- arr
        | Error _ -> failwith "Setup failed"
    
    [<Benchmark>]
    member this.Mean() =
        Fowl.Stats.Descriptive.mean this.Data
    
    [<Benchmark>]
    member this.Variance() =
        Fowl.Stats.Descriptive.var this.Data
    
    [<Benchmark>]
    member this.StdDev() =
        Fowl.Stats.Descriptive.std this.Data

// ============================================================================
// AD Benchmarks
// ============================================================================

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type ADBenchmarks() =
    [<Benchmark>]
    member this.DiffSin() =
        let f x = AD.sin x
        AD.diffF f 1.0
    
    [<Benchmark>]
    member this.DiffPolynomial() =
        let f x = AD.pow x (AD.pack_flt 3.0)
        AD.diffF f 2.0
    
    [<Benchmark>]
    member this.GradSin() =
        let f x = AD.sin x
        AD.gradF f 1.0
    
    [<Benchmark>]
    member this.HessianPolynomial() =
        let f x = AD.pow x (AD.pack_flt 2.0)
        AD.hessianF' f 3.0

// ============================================================================
// Main Entry Point
// ============================================================================

[<EntryPoint>]
let main argv =
    printfn "Fowl Benchmark Suite"
    printfn "===================="
    printfn ""
    
    // Run all benchmarks
    let config = 
        DefaultConfig.Instance
            .WithOption(ConfigOptions.DisableOptimizationsValidator, true)
    
    let benchmarks = [|
        typeof<ElementWiseBenchmarks>
        typeof<ReductionBenchmarks>
        typeof<NdarrayBenchmarks>
        typeof<MatrixBenchmarks>
        typeof<StatsBenchmarks>
        typeof<ADBenchmarks>
        typeof<Avx2Benchmarks>
        typeof<SimdElementWiseBenchmarks>
        typeof<SimdReductionBenchmarks>
        typeof<SimdScalarBenchmarks>
        typeof<MemoryAllocationBenchmarks>
        typeof<ZeroCopyBenchmarks>
        typeof<NdarrayViewBenchmarks>
        typeof<InPlaceBenchmarks>
        typeof<PooledBufferBenchmarks>
        typeof<ParallelElementWiseBenchmarks>
        typeof<ParallelReductionBenchmarks>
        typeof<ParallelMapBenchmarks>
        typeof<ThreadSafeRandomBenchmarks>
        typeof<ParallelMatrixBenchmarks>
        typeof<ThresholdBenchmarks>
    |]
    
    BenchmarkRunner.Run(benchmarks, config) |> ignore
    
    0
