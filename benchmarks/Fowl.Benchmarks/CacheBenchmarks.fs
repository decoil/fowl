module Fowl.Benchmarks.CacheBenchmarks

open System
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Running
open BenchmarkDotNet.Jobs

// ============================================================================
// Cache Performance Benchmarks
// ============================================================================

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type CacheMatrixBenchmarks() =
    let sizes = [| 100; 500; 1000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val N = 500 with get, set
    
    member val A = Unchecked.defaultof<float[,]> with get, set
    member val B = Unchecked.defaultof<float[,]> with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.A <- Array2D.init this.N this.N (fun _ _ -> rng.NextDouble())
        this.B <- Array2D.init this.N this.N (fun _ _ -> rng.NextDouble())
        
        if this.N = 100 then
            printfn "\n=== Cache Benchmarks ==="
            Fowl.Cache.printCacheInfo()
    
    [<Benchmark(Baseline = true)>]
    member this.NaiveMatMul() =
        // Naive i-j-k order (poor cache locality)
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
    member this.TiledMatMul() =
        Fowl.Cache.CacheMatrixOps.matmulTiled this.A this.B
    
    [<Benchmark>]
    member this.BlockedMatMul() =
        Fowl.Cache.CacheMatrixOps.matmulBlocked this.A this.B
    
    [<Benchmark>]
    member this.NaiveTranspose() =
        let m = this.N
        let n = this.N
        let result = Array2D.zeroCreate n m
        for i = 0 to m - 1 do
            for j = 0 to n - 1 do
                result.[j, i] <- this.A.[i, j]
        result
    
    [<Benchmark>]
    member this.BlockedTranspose() =
        Fowl.Cache.CacheMatrixOps.transposeBlocked this.A

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type CacheLocalityBenchmarks() =
    let sizes = [| 100; 500; 1000; 2000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val N = 1000 with get, set
    
    member val Matrix = Unchecked.defaultof<float[,]> with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.Matrix <- Array2D.init this.N this.N (fun _ _ -> rng.NextDouble())
    
    [<Benchmark(Baseline = true)>]
    member this.RowMajorSum() =
        Fowl.Cache.LoopReorderOps.sumRowMajor this.Matrix
    
    [<Benchmark>]
    member this.ColumnMajorSum() =
        Fowl.Cache.LoopReorderOps.sumColumnMajor this.Matrix
    
    [<Benchmark>]
    member this.NaiveCopy() =
        let rows = this.N
        let cols = this.N
        let result = Array2D.zeroCreate rows cols
        for i = 0 to rows - 1 do
            for j = 0 to cols - 1 do
                result.[i, j] <- this.Matrix.[i, j]
        result
    
    [<Benchmark>]
    member this.BlockedCopy() =
        Fowl.Cache.LoopReorderOps.copyBlocked this.Matrix

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type CacheVsParallelBenchmarks() =
    let sizes = [| 500; 1000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val N = 500 with get, set
    
    member val A = Unchecked.defaultof<float[,]> with get, set
    member val B = Unchecked.defaultof<float[,]> with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.A <- Array2D.init this.N this.N (fun _ _ -> rng.NextDouble())
        this.B <- Array2D.init this.N this.N (fun _ _ -> rng.NextDouble())
    
    [<Benchmark(Baseline = true)>]
    member this.Sequential() =
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
    member this.ParallelOnly() =
        Fowl.Parallel.ParallelMatrixOps.matmul this.A this.B
    
    [<Benchmark>]
    member this.CacheTiledOnly() =
        Fowl.Cache.CacheMatrixOps.matmulTiled this.A this.B
    
    [<Benchmark>]
    member this.CacheAndParallel() =
        // Best of both: cache blocking + parallel
        let m = this.N
        let n = this.N
        let p = this.N
        let result = Array2D.zeroCreate m p
        let blockSize = min 64 m
        
        System.Threading.Tasks.Parallel.For(0, (m + blockSize - 1) / blockSize, fun blockIdx ->
            let ii = blockIdx * blockSize
            let iEnd = min (ii + blockSize) m
            
            for jj in 0..blockSize..p-1 do
                let jEnd = min (jj + blockSize) p
                for kk in 0..blockSize..n-1 do
                    let kEnd = min (kk + blockSize) n
                    
                    for i = ii to iEnd - 1 do
                        for j = jj to jEnd - 1 do
                            let mutable sum = result.[i, j]
                            for k = kk to kEnd - 1 do
                                sum <- sum + this.A.[i, k] * this.B.[k, j]
                            result.[i, j] <- sum
        ) |> ignore
        
        result

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type SpatialLocalityBenchmarks() =
    let sizes = [| 100000; 1000000 |]
    
    [<ParamsSource(nameof(sizes))>]
    member val Size = 100000 with get, set
    
    member val Array = Array.empty with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.Array <- Array.init this.Size (fun _ -> rng.NextDouble())
    
    [<Benchmark(Baseline = true)>]
    member this.SequentialAccess() =
        let mutable sum = 0.0
        for i = 0 to this.Array.Length - 1 do
            sum <- sum + this.Array.[i]
        sum
    
    [<Benchmark>]
    member this.StridedAccess() =
        // Simulate strided access (poor locality)
        let stride = 16
        let mutable sum = 0.0
        for i in 0..stride..this.Array.Length-1 do
            sum <- sum + this.Array.[i]
        sum
    
    [<Benchmark>]
    member this.RandomAccess() =
        // Poor locality
        let rng = Random(42)
        let mutable sum = 0.0
        for _ = 1 to 10000 do
            let idx = rng.Next(this.Array.Length)
            sum <- sum + this.Array.[idx]
        sum

[<MemoryDiagnoser>]
[<SimpleJob(RuntimeMoniker.Net80)>]
type BlockSizeBenchmarks() =
    [<Params(128, 256, 512, 1024)>]
    member val N = 512 with get, set
    
    member val A = Unchecked.defaultof<float[,]> with get, set
    member val B = Unchecked.defaultof<float[,]> with get, set
    
    [<GlobalSetup>]
    member this.Setup() =
        let rng = Random()
        this.A <- Array2D.init this.N this.N (fun _ _ -> rng.NextDouble())
        this.B <- Array2D.init this.N this.N (fun _ _ -> rng.NextDouble())
    
    [<Benchmark>]
    member this.BlockSize32() =
        // 32x32 block (8KB)
        Fowl.Cache.CacheMatrixOps.matmulTiled this.A this.B
    
    [<Benchmark>]
    member this.BlockSize64() =
        // 64x64 block (32KB)
        Fowl.Cache.CacheMatrixOps.matmulTiled this.A this.B
    
    [<Benchmark>]
    member this.BlockSize128() =
        // 128x128 block (128KB)
        Fowl.Cache.CacheMatrixOps.matmulBlocked this.A this.B
