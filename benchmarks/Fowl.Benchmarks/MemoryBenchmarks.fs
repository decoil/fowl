module Fowl.Benchmarks.MemoryBenchmarks

open System
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Running
open BenchmarkDotNet.Jobs
open Fowl
open Fowl.Core

// ============================================================================
// Memory Allocation Benchmarks
// ============================================================================

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type MemoryAllocationBenchmarks() =
    let sizes = [| 100; 1000; 10000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val Size = 1000 with get, set
    
    member val A = Array.empty with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.A <- Array.init this.Size (fun _ -> rng.NextDouble())
    
    [<Benchmark(Baseline = true)>]
    member this.ArrayCopy() =
        Array.copy this.A
    
    [<Benchmark>]
    member this.ArrayZeroCreate() =
        Array.zeroCreate this.Size
    
    [<Benchmark>]
    member this.ArrayPoolRent() =
        let pool = System.Buffers.ArrayPool<double>.Shared
        let arr = pool.Rent(this.Size)
        pool.Return(arr, false)
        arr

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type ZeroCopyBenchmarks() =
    let sizes = [| 100; 1000; 10000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val Size = 1000 with get, set
    
    member val A = Array.empty with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.A <- Array.init this.Size (fun _ -> rng.NextDouble())
    
    [<Benchmark(Baseline = true)>]
    member this.ArraySubCopy() =
        // Traditional: copy slice
        let start = this.Size / 4
        let len = this.Size / 2
        Array.sub this.A start len
    
    [<Benchmark>]
    member this.SpanSlice() =
        // Zero-copy: span slice
        let start = this.Size / 4
        let len = this.Size / 2
        let span = System.Span(this.A, start, len)
        span.ToArray()  // Force realization
    
    [<Benchmark>]
    member this.SpanSliceNoCopy() =
        // Just the slice operation (no copy)
        let start = this.Size / 4
        let len = this.Size / 2
        System.Span(this.A, start, len)

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type NdarrayViewBenchmarks() =
    let sizes = [| 100; 500 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val N = 100 with get, set
    
    member val Matrix = Unchecked.defaultof<Ndarray<Float64, float>> with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        let data = Array.init (this.N * this.N) (fun _ -> rng.NextDouble())
        match Ndarray.ofArray data [||this.N; this.N|] with
        | Ok arr -> this.Matrix <- arr
        | Error _ -> failwith "Setup failed"
    
    [<Benchmark(Baseline = true)>]
    member this.SliceCopy() =
        // Traditional: copy slice
        Fowl.Core.Slice.slice this.Matrix [|SliceSpec.Range(Some 10, Some (this.N - 10), None); SliceSpec.All|]
    
    [<Benchmark>]
    member this.RowView() =
        // Zero-copy: row view
        Fowl.Memory.NdarrayView.row this.Matrix 50
    
    [<Benchmark>]
    member this.SubMatrixView() =
        // Zero-copy: submatrix view
        Fowl.Memory.NdarrayView.subMatrix this.Matrix 10 (this.N - 20) 10 (this.N - 20)

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type InPlaceBenchmarks() =
    let sizes = [| 1000; 10000; 100000 |]
    
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
    member this.AddNewAllocation() =
        // Creates new array
        Array.map2 (+) this.A this.B
    
    [<Benchmark>]
    member this.AddInPlace() =
        // No allocation
        Array.Copy(this.A, this.Result, this.Size)
        for i = 0 to this.Size - 1 do
            this.Result.[i] <- this.Result.[i] + this.B.[i]
        this.Result
    
    [<Benchmark>]
    member this.AddSimdInPlace() =
        // No allocation + SIMD
        Array.Copy(this.A, this.Result, this.Size)
        Fowl.Native.SIMD.KernelSelector.AddInPlace(this.Result, this.B)
        this.Result

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type PooledBufferBenchmarks() =
    let sizes = [| 1000; 10000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val Size = 1000 with get, set
    
    member val A = Array.empty with get, set
    member val B = Array.empty with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.A <- Array.init this.Size (fun _ -> rng.NextDouble())
        this.B <- Array.init this.Size (fun _ -> rng.NextDouble())
    
    [<Benchmark(Baseline = true)>]
    member this.NormalAllocation() =
        // Normal allocation every iteration
        let result = Array.zeroCreate this.Size
        for i = 0 to this.Size - 1 do
            result.[i] <- this.A.[i] + this.B.[i]
        result
    
    [<Benchmark>]
    member this.PooledBuffer() =
        // Reuse pooled buffer
        use pooled = Fowl.Memory.ArrayPoolOps.rentDouble this.Size
        let buffer = pooled.Array
        for i = 0 to this.Size - 1 do
            buffer.[i] <- this.A.[i] + this.B.[i]
        // Buffer automatically returned to pool
        buffer.[0]  // Return something
