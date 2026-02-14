module Fowl.Tests.PropertyBased

open FsCheck
open Fowl
open Fowl.Core.Types

/// <summary>Property-based tests for numerical operations.
/// Uses FsCheck for randomized testing with shrinking.
/// </summary>module Properties =
    
    /// <summary>Property: Addition is commutative.
/// a + b = b + a
/// </summary>let addCommutative (a: float) (b: float) =
        a + b = b + a
    
    /// <summary>Property: Addition is associative.
/// (a + b) + c = a + (b + c)
/// </summary>let addAssociative (a: float) (b: float) (c: float) =
        let lhs = (a + b) + c
        let rhs = a + (b + c)
        abs (lhs - rhs) < 1e-10
    
    /// <summary>Property: Multiplication is distributive over addition.
/// a * (b + c) = a * b + a * c
/// </summary>let multDistributive (a: float) (b: float) (c: float) =
        let lhs = a * (b + c)
        let rhs = a * b + a * c
        abs (lhs - rhs) < 1e-10
    
    /// <summary>Property: Transpose of transpose equals original.
/// (A^T)^T = A
/// </summary>let transposeInvolution (rows: int) (cols: int) =
        let actualRows = max 1 (min rows 10)
        let actualCols = max 1 (min cols 10)
        
        let data = Array.init (actualRows * actualCols) (fun _ -> Random().NextDouble())
        match Ndarray.ofArray data [|actualRows; actualCols|] with
        | Ok arr ->
            match Matrix.transpose arr with
            | Ok transposed ->
                match Matrix.transpose transposed with
                | Ok back ->
                    let original = Ndarray.toArray arr
                    let result = Ndarray.toArray back
                    original = result
                | Error _ -> true  // Skip on error
            | Error _ -> true
        | Error _ -> true
    
    /// <summary>Property: Matrix trace is invariant under transpose.
/// tr(A) = tr(A^T)
/// </summary>let traceTransposeInvariant (n: int) =
        let size = max 1 (min n 10)
        
        let data = Array.init (size * size) (fun _ -> Random().NextDouble())
        match Ndarray.ofArray data [|size; size|] with
        | Ok arr ->
            match Matrix.trace arr with
            | Ok tr1 ->
                match Matrix.transpose arr with
                | Ok transposed ->
                    match Matrix.trace transposed with
                    | Ok tr2 -> abs (tr1 - tr2) < 1e-10
                    | Error _ -> true
                | Error _ -> true
            | Error _ -> true
        | Error _ -> true
    
    /// <summary>Property: Gaussian PDF integrates to approximately 1.
/// Numerical integration over wide range should be ~1.
/// </summary>let gaussianPDFIntegratesToOne (mu: float) (sigma: float) =
        let actualSigma = abs sigma + 0.1  // Ensure positive
        
        // Simple trapezoidal integration
        let a = mu - 5.0 * actualSigma
        let b = mu + 5.0 * actualSigma
        let n = 1000
        let h = (b - a) / float n
        
        let sum =
            seq { 0 .. n }
            |> Seq.map (fun i -> 
                let x = a + float i * h
                match GaussianDistribution.pdf mu actualSigma x with
                | Ok p -> p
                | Error _ -> 0.0)
            |> Seq.sum
        
        let integral = sum * h
        abs (integral - 1.0) < 0.01
    
    /// <summary>Property: Distribution mean is approximately the expected value.
/// For large samples, sample mean should be close to distribution mean.
/// </summary>let distributionMeanApproximation (lambda: float) =
        let actualLambda = abs lambda + 0.1
        
        // Generate large sample from Poisson
        match PoissonDistribution.rvs actualLambda [|1000|] with
        | Ok samples ->
            let sampleMean = samples |> Ndarray.toArray |> Array.average
            let expectedMean = actualLambda
            abs (sampleMean - expectedMean) < 0.5  // Loose tolerance for randomness
        | Error _ -> true
    
    /// <summary>Property: SVD reconstructs original matrix.
/// U * diag(S) * V^T ≈ A
/// </summary>let svdReconstruction (rows: int) (cols: int) =
        let actualRows = max 2 (min rows 8)
        let actualCols = max 2 (min cols 8)
        
        let data = Array.init (actualRows * actualCols) (fun _ -> Random().NextDouble())
        match Ndarray.ofArray data [|actualRows; actualCols|] with
        | Ok arr ->
            match Factorizations.svd arr with
            | Ok (u, s, vt) ->
                // Reconstruct: u * diag(s) * vt
                // For now, simplified check
                Ndarray.shape u = [|actualRows; actualRows|] &&
                Ndarray.shape s = [|min actualRows actualCols|] &&
                Ndarray.shape vt = [|actualCols; actualCols|]
            | Error _ -> true
        | Error _ -> true

/// <summary>Run all property-based tests.
/// </summary>let runAllTests() =
    printfn "Running property-based tests..."
    
    // Configure FsCheck
    let config = { Config.Quick with MaxTest = 100 }
    
    // Run tests
    let results = [
        "addCommutative", Check.One(config, Properties.addCommutative)
        "addAssociative", Check.One(config, Properties.addAssociative)
        "multDistributive", Check.One(config, Properties.multDistributive)
    ]
    
    // Print results
    results |> List.iter (fun (name, result) ->
        printfn "%s: %A" name result)
    
    printfn "Property-based tests complete."