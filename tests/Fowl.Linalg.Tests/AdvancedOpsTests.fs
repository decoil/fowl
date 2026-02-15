module Fowl.Tests.Linalg.AdvancedOps

open Expecto
open Fowl
open Fowl.Core.Types
open Fowl.Linalg.AdvancedOps

/// Helper for approximate float equality
let approxEqual (tol: float) (actual: float) (expected: float) =
    abs (actual - expected) < tol

let tests =
    testList "Advanced Linear Algebra" [
        // ========================================================================
        // Least Squares Tests
        // ========================================================================
        
        test "lstsq solves overdetermined system" {
            // Solve: find x that minimizes ||Ax - b||
            // A = [[1, 1], [1, 2], [1, 3], [1, 4]]
            // b = [6, 5, 7, 10]
            // Expected: approximately y = 1.5x + 3
            
            let A = Ndarray.ofArray [|1.0; 1.0; 1.0; 2.0; 1.0; 3.0; 1.0; 4.0|] [|4; 2|]
            let b = Ndarray.ofArray [|6.0; 5.0; 7.0; 10.0|] [|4|]
            
            match A, b with
            | Ok a, Ok bv ->
                match lstsq a bv with
                | Ok (x, residual, rank, s) ->
                    let xData = Ndarray.toArray x
                    Expect.equal xData.Length 2 "Solution has 2 parameters"
                    Expect.isTrue (rank >= 1) "Matrix has rank at least 1"
                    Expect.isTrue (residual >= 0.0) "Residual is non-negative"
                | Error e -> failtestf "lstsq failed: %A" e
            | Error e, _ | _, Error e -> failtestf "Array creation failed: %A" e
        }
        
        test "lstsq solves simple linear fit" {
            // y = 2x + 1
            // Points: (1, 3), (2, 5), (3, 7), (4, 9)
            
            let A = Ndarray.ofArray [|1.0; 1.0; 1.0; 2.0; 1.0; 3.0; 1.0; 4.0|] [|4; 2|]
            let b = Ndarray.ofArray [|3.0; 5.0; 7.0; 9.0|] [|4|]
            
            match A, b with
            | Ok a, Ok bv ->
                match lstsq a bv with
                | Ok (x, residual, _, _) ->
                    let xData = Ndarray.toArray x
                    // Solution should be close to [1, 2] (intercept, slope)
                    printfn "lstsq solution: intercept=%.4f, slope=%.4f, residual=%.4f" xData.[0] xData.[1] residual
                    
                    Expect.floatClose Accuracy.medium xData.[0] 1.0 "Intercept should be ~1"
                    Expect.floatClose Accuracy.medium xData.[1] 2.0 "Slope should be ~2"
                    Expect.floatClose Accuracy.medium residual 0.0 "Residual should be ~0 for exact fit"
                | Error e -> failtestf "lstsq failed: %A" e
            | Error e, _ | _, Error e -> failtestf "Array creation failed: %A" e
        }
        
        // ========================================================================
        // Pseudoinverse Tests
        // ========================================================================
        
        test "pinv computes pseudoinverse" {
            // A = [[1, 2], [3, 4], [5, 6]] (3x2)
            let A = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0|] [|3; 2|]
            
            match A with
            | Ok a ->
                match pinv a with
                | Ok pinvA ->
                    let shape = Ndarray.shape pinvA
                    // Pseudoinverse of 3x2 is 2x3
                    Expect.equal shape [|2; 3|] "Pseudoinverse has transposed shape"
                | Error e -> failtestf "pinv failed: %A" e
            | Error e -> failtestf "Array creation failed: %A" e
        }
        
        test "pinv satisfies Moore-Penrose conditions" {
            // For pseudoinverse A⁺: A * A⁺ * A = A
            let A = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0|] [|2; 2|]
            
            match A with
            | Ok a ->
                result {
                    let! pinvA = pinv a
                    let! aPinvA = Matrix.matmul a pinvA
                    let! aPinvA_A = Matrix.matmul aPinvA a
                    
                    // Check A * A⁺ * A ≈ A
                    let! original = Ndarray.toArray a
                    let! result = Ndarray.toArray aPinvA_A
                    
                    let tol = 1e-6
                    for i = 0 to 3 do
                        Expect.isTrue (abs (result.[i] - original.[i]) < tol)
                            (sprintf "Moore-Penrose condition 1 failed at index %d" i)
                } |> function
                    | Ok () -> ()
                    | Error e -> failtestf "Test failed: %A" e
            | Error e -> failtestf "Array creation failed: %A" e
        }
        
        // ========================================================================
        // Rank Tests
        // ========================================================================
        
        test "rank of full-rank matrix" {
            // Identity matrix should have rank = dimension
            let I = Ndarray.ofArray [|1.0; 0.0; 0.0; 1.0|] [|2; 2|]
            
            match I with
            | Ok m ->
                match rank m with
                | Ok r -> Expect.equal r 2 "Identity matrix has full rank"
                | Error e -> failtestf "rank failed: %A" e
            | Error e -> failtestf "Array creation failed: %A" e
        }
        
        test "rank of rank-deficient matrix" {
            // [[1, 2], [2, 4]] has rank 1 (rows are linearly dependent)
            let A = Ndarray.ofArray [|1.0; 2.0; 2.0; 4.0|] [|2; 2|]
            
            match A with
            | Ok m ->
                match rank m with
                | Ok r -> Expect.equal r 1 "Matrix has rank 1"
                | Error e -> failtestf "rank failed: %A" e
            | Error e -> failtestf "Array creation failed: %A" e
        }
        
        test "rank of zero matrix" {
            let Z = Ndarray.zeros<Float64> [|3; 3|]
            
            match Z with
            | Ok m ->
                match rank m with
                | Ok r -> Expect.equal r 0 "Zero matrix has rank 0"
                | Error e -> failtestf "rank failed: %A" e
            | Error e -> failtestf "zeros failed: %A" e
        }
        
        // ========================================================================
        // Condition Number Tests
        // ========================================================================
        
        test "cond of identity matrix" {
            let I = Ndarray.ofArray [|1.0; 0.0; 0.0; 1.0|] [|2; 2|]
            
            match I with
            | Ok m ->
                match cond m with
                | Ok c -> 
                    // Identity has condition number 1
                    Expect.floatClose Accuracy.medium c 1.0 "Identity has condition number 1"
                | Error e -> failtestf "cond failed: %A" e
            | Error e -> failtestf "Array creation failed: %A" e
        }
        
        test "cond of ill-conditioned matrix" {
            // Hilbert matrix is notoriously ill-conditioned
            // [[1, 1/2], [1/2, 1/3]]
            let H = Ndarray.ofArray [|1.0; 0.5; 0.5; 0.333333333|] [|2; 2|]
            
            match H with
            | Ok m ->
                match cond m with
                | Ok c -> 
                    // Should be large
                    Expect.isTrue (c > 10.0) "Hilbert matrix is ill-conditioned"
                | Error e -> failtestf "cond failed: %A" e
            | Error e -> failtestf "Array creation failed: %A" e
        }
        
        // ========================================================================
        // Matrix Exponential Tests
        // ========================================================================
        
        test "expm of zero matrix is identity" {
            let Z = Ndarray.zeros<Float64> [|2; 2|]
            
            match Z with
            | Ok m ->
                match expm m with
                | Ok result ->
                    let! data = Ndarray.toArray result
                    Expect.floatClose Accuracy.medium data.[0] 1.0 "e^0 = 1"
                    Expect.floatClose Accuracy.medium data.[3] 1.0 "e^0 = 1"
                    Expect.floatClose Accuracy.medium data.[1] 0.0 "Off-diagonal is 0"
                    Expect.floatClose Accuracy.medium data.[2] 0.0 "Off-diagonal is 0"
                | Error e -> failtestf "expm failed: %A" e
            | Error e -> failtestf "zeros failed: %A" e
        }
        
        test "expm of diagonal matrix" {
            // expm(diag([1, 2])) = diag([e^1, e^2])
            let D = Ndarray.ofArray [|1.0; 0.0; 0.0; 2.0|] [|2; 2|]
            
            match D with
            | Ok m ->
                match expm m with
                | Ok result ->
                    let! data = Ndarray.toArray result
                    let e1 = exp 1.0
                    let e2 = exp 2.0
                    
                    Expect.floatClose Accuracy.medium data.[0] e1 "e^1"
                    Expect.floatClose Accuracy.medium data.[3] e2 "e^2"
                    Expect.floatClose Accuracy.medium data.[1] 0.0 "Off-diagonal is 0"
                | Error e -> failtestf "expm failed: %A" e
            | Error e -> failtestf "Array creation failed: %A" e
        }
        
        // ========================================================================
        // Frobenius Norm Tests
        // ========================================================================
        
        test "normFrobenius of identity" {
            let I = Ndarray.ofArray [|1.0; 0.0; 0.0; 1.0|] [|2; 2|]
            
            match I with
            | Ok m ->
                match normFrobenius m with
                | Ok n -> 
                    // ||I||_F = sqrt(1^2 + 0^2 + 0^2 + 1^2) = sqrt(2)
                    Expect.floatClose Accuracy.medium n (sqrt 2.0) "Frobenius norm of identity"
                | Error e -> failtestf "normFrobenius failed: %A" e
            | Error e -> failtestf "Array creation failed: %A" e
        }
        
        test "normFrobenius of ones matrix" {
            let O = Ndarray.ones<Float64> [|3; 3|]
            
            match O with
            | Ok m ->
                match normFrobenius m with
                | Ok n -> 
                    // ||ones(3,3)||_F = sqrt(9) = 3
                    Expect.floatClose Accuracy.medium n 3.0 "Frobenius norm of 3x3 ones"
                | Error e -> failtestf "normFrobenius failed: %A" e
            | Error e -> failtestf "ones failed: %A" e
        }
    ]
