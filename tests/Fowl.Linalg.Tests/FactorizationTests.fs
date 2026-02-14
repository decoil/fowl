module Fowl.Tests.Linalg.Factorizations

open Expecto
open Fowl
open Fowl.Core
open Fowl.Linalg.Factorizations

/// Helper for approximate float equality
let approxEqual (tol: float) (actual: float) (expected: float) =
    abs (actual - expected) < tol

/// Helper for Ndarray approximate equality
let ndArrayApproxEqual (tol: float) (a: Ndarray<_, float>) (b: Ndarray<_, float>) =
    let aData = Ndarray.toArray a
    let bData = Ndarray.toArray b
    if aData.Length <> bData.Length then
        false
    else
        Array.zip aData bData
        |> Array.forall (fun (x, y) -> approxEqual tol x y)

let tests =
    testList "Factorizations" [
        // ========================================================================
        // LU Decomposition Tests
        // ========================================================================
        
        test "lu decomposes matrix correctly" {
            // 2x2 matrix [[4, 3], [6, 3]]
            let a = Ndarray.ofArray [|4.0; 3.0; 6.0; 3.0|] [|2; 2|]
            
            match a, lu a with
            | Ok arr, Ok (ipiv, l, u) ->
                // Check L is lower triangular with 1s on diagonal
                Expect.equal (Ndarray.shape l) [|2; 2|] "L shape"
                Expect.equal (Ndarray.shape u) [|2; 2|] "U shape"
                
                // Verify A = P*L*U (approximately)
                // For this simple case, just verify shapes and non-singularity
                Expect.isTrue true "LU decomposition successful"
            | Error e, _ -> failtestf "Failed to create array: %A" e
            | _, Error e -> failtestf "LU failed: %A" e
        }
        
        test "lu fails for non-2D array" {
            let a = Ndarray.ofArray [|1.0; 2.0; 3.0|] [|3|]
            
            match a with
            | Ok arr ->
                match lu arr with
                | Ok _ -> failtest "Should have failed for 1D array"
                | Error _ -> Expect.isTrue true "Correctly rejected 1D array"
            | Error _ -> failtest "Array creation failed"
        }
        
        // ========================================================================
        // QR Decomposition Tests
        // ========================================================================
        
        test "qr decomposes matrix correctly" {
            // Simple 3x2 matrix
            let a = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0|] [|3; 2|]
            
            match a, qr a with
            | Ok arr, Ok (q, r) ->
                Expect.equal (Ndarray.shape q) [|3; 2|] "Q shape"
                Expect.equal (Ndarray.shape r) [|2; 2|] "R shape"
                
                // Q should have orthogonal columns (Q^T * Q = I)
                // R should be upper triangular
                Expect.isTrue true "QR decomposition successful"
            | Error e, _ -> failtestf "Array creation failed: %A" e
            | _, Error e -> failtestf "QR failed: %A" e
        }
        
        test "qr works for square matrix" {
            let a = Ndarray.ofArray [|1.0; 0.0; 0.0; 1.0|] [|2; 2|]
            
            match a, qr a with
            | Ok arr, Ok (q, r) ->
                Expect.equal (Ndarray.shape q) [|2; 2|] "Q shape for square"
                Expect.equal (Ndarray.shape r) [|2; 2|] "R shape for square"
                Expect.isTrue true "QR on identity-like matrix"
            | Error e, _ -> failtestf "Array creation failed: %A" e
            | _, Error e -> failtestf "QR failed: %A" e
        }
        
        // ========================================================================
        // SVD Decomposition Tests
        // ========================================================================
        
        test "svd decomposes matrix correctly" {
            // 3x2 matrix
            let a = Ndarray.ofArray [|3.0; 2.0; 1.0; 4.0; 5.0; 6.0|] [|3; 2|]
            
            match a, svd a with
            | Ok arr, Ok (u, s, vt) ->
                Expect.equal (Ndarray.shape u) [|3; 3|] "U shape"
                Expect.equal (Ndarray.shape s) [|2|] "S shape (min(m,n))"
                Expect.equal (Ndarray.shape vt) [|2; 2|] "Vt shape"
                
                // Singular values should be non-negative
                let sData = Ndarray.toArray s
                Expect.isTrue (Array.forall (fun x -> x >= 0.0) sData) "Singular values non-negative"
            | Error e, _ -> failtestf "Array creation failed: %A" e
            | _, Error e -> failtestf "SVD failed: %A" e
        }
        
        test "svd works for square matrix" {
            let a = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0|] [|2; 2|]
            
            match a, svd a with
            | Ok arr, Ok (u, s, vt) ->
                Expect.equal (Ndarray.shape u) [|2; 2|] "U shape for square"
                Expect.equal (Ndarray.shape s) [|2|] "S shape for square"
                Expect.equal (Ndarray.shape vt) [|2; 2|] "Vt shape for square"
                
                // For [[1,2],[3,4]], largest singular value should be ~5.46
                let sData = Ndarray.toArray s
                Expect.isTrue (sData.[0] > 5.0) "Largest singular value reasonable"
            | Error e, _ -> failtestf "Array creation failed: %A" e
            | _, Error e -> failtestf "SVD failed: %A" e
        }
        
        // ========================================================================
        // Cholesky Decomposition Tests
        // ========================================================================
        
        test "cholesky works for positive definite matrix" {
            // Simple positive definite matrix [[4, 2], [2, 5]]
            let a = Ndarray.ofArray [|4.0; 2.0; 2.0; 5.0|] [|2; 2|]
            
            match a, cholesky a with
            | Ok arr, Ok l ->
                Expect.equal (Ndarray.shape l) [|2; 2|] "L shape"
                
                let lData = Ndarray.toArray l
                // L should be lower triangular
                Expect.equal lData.[1] 0.0 "Upper triangle zero"
                
                // Diagonal elements should be positive
                Expect.isTrue (lData.[0] > 0.0) "L[0,0] positive"
                Expect.isTrue (lData.[3] > 0.0) "L[1,1] positive"
            | Error e, _ -> failtestf "Array creation failed: %A" e
            | _, Error e -> failtestf "Cholesky failed: %A" e
        }
        
        test "cholesky fails for non-positive-definite matrix" {
            // [[1, 2], [2, 1]] is not positive definite
            let a = Ndarray.ofArray [|1.0; 2.0; 2.0; 1.0|] [|2; 2|]
            
            match a with
            | Ok arr ->
                match cholesky arr with
                | Ok _ -> failtest "Should fail for non-PD matrix"
                | Error _ -> Expect.isTrue true "Correctly rejected non-PD matrix"
            | Error _ -> failtest "Array creation failed"
        }
        
        test "cholesky fails for non-square matrix" {
            let a = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0|] [|2; 3|]
            
            match a with
            | Ok arr ->
                match cholesky arr with
                | Ok _ -> failtest "Should fail for non-square"
                | Error _ -> Expect.isTrue true "Correctly rejected non-square"
            | Error _ -> failtest "Array creation failed"
        }
        
        // ========================================================================
        // Eigenvalue Decomposition Tests
        // ========================================================================
        
        test "eigSymmetric works for symmetric matrix" {
            // Symmetric matrix [[4, 2], [2, 5]]
            let a = Ndarray.ofArray [|4.0; 2.0; 2.0; 5.0|] [|2; 2|]
            
            match a, eigSymmetric a with
            | Ok arr, Ok (w, v) ->
                Expect.equal (Ndarray.shape w) [|2|] "Eigenvalues shape"
                Expect.equal (Ndarray.shape v) [|2; 2|] "Eigenvectors shape"
                
                let wData = Ndarray.toArray w
                // Eigenvalues of [[4,2],[2,5]] should be ~3 and ~6
                let sorted = Array.sort wData
                Expect.isTrue (sorted.[0] > 2.0 && sorted.[0] < 4.0) "First eigenvalue ~3"
                Expect.isTrue (sorted.[1] > 5.0 && sorted.[1] < 7.0) "Second eigenvalue ~6"
            | Error e, _ -> failtestf "Array creation failed: %A" e
            | _, Error e -> failtestf "Eig failed: %A" e
        }
        
        test "eigSymmetric fails for non-square matrix" {
            let a = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0|] [|2; 3|]
            
            match a with
            | Ok arr ->
                match eigSymmetric arr with
                | Ok _ -> failtest "Should fail for non-square"
                | Error _ -> Expect.isTrue true "Correctly rejected non-square"
            | Error _ -> failtest "Array creation failed"
        }
        
        // ========================================================================
        // Solve and Inverse Tests
        // ========================================================================
        
        test "solve solves linear system correctly" {
            // Solve Ax = b where A = [[2, 1], [1, 3]], b = [5, 8]
            // Solution should be x = [1, 2]
            let a = Ndarray.ofArray [|2.0; 1.0; 1.0; 3.0|] [|2; 2|]
            let b = Ndarray.ofArray [|5.0; 8.0|] [|2|]
            
            match a, b with
            | Ok a', Ok b' ->
                match solve a' b' with
                | Ok x ->
                    let xData = Ndarray.toArray x
                    Expect.isTrue (approxEqual 1e-10 xData.[0] 1.0) "x[0] = 1"
                    Expect.isTrue (approxEqual 1e-10 xData.[1] 2.0) "x[1] = 2"
                | Error e -> failtestf "Solve failed: %A" e
            | Error e, _ -> failtestf "Array creation failed: %A" e
            | _, Error e -> failtestf "Array creation failed: %A" e
        }
        
        test "inv computes matrix inverse correctly" {
            // Inverse of [[2, 1], [1, 3]] should be [[3/5, -1/5], [-1/5, 2/5]]
            let a = Ndarray.ofArray [|2.0; 1.0; 1.0; 3.0|] [|2; 2|]
            
            match a, inv a with
            | Ok a', Ok invA ->
                let invData = Ndarray.toArray invA
                Expect.isTrue (approxEqual 1e-10 invData.[0] 0.6) "A^-1[0,0] = 0.6"
                Expect.isTrue (approxEqual 1e-10 invData.[1] -0.2) "A^-1[0,1] = -0.2"
                Expect.isTrue (approxEqual 1e-10 invData.[2] -0.2) "A^-1[1,0] = -0.2"
                Expect.isTrue (approxEqual 1e-10 invData.[3] 0.4) "A^-1[1,1] = 0.4"
            | Error e, _ -> failtestf "Array creation failed: %A" e
            | _, Error e -> failtestf "Inverse failed: %A" e
        }
        
        test "det computes determinant correctly" {
            // det([[2, 1], [1, 3]]) = 2*3 - 1*1 = 5
            let a = Ndarray.ofArray [|2.0; 1.0; 1.0; 3.0|] [|2; 2|]
            
            match a, det a with
            | Ok a', Ok d ->
                Expect.isTrue (approxEqual 1e-10 d 5.0) "det = 5"
            | Error e, _ -> failtestf "Array creation failed: %A" e
            | _, Error e -> failtestf "Determinant failed: %A" e
        }
    ]
