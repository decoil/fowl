module Fowl.Linalg.Factorizations

open System
open System.Runtime.InteropServices
open Fowl
open Fowl.Core.Types

/// Native LAPACK bindings with error handling
try
    [<DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl)>]
    extern int LAPACKE_dgetrf(
        int matrix_layout,
        int m,
        int n,
        double[] a,
        int lda,
        int[] ipiv
    )
    
    [<DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl)>]
    extern int LAPACKE_dgetri(
        int matrix_layout,
        int n,
        double[] a,
        int lda,
        int[] ipiv
    )
    
    [<DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl)>]
    extern int LAPACKE_dgesv(
        int matrix_layout,
        int n,
        int nrhs,
        double[] a,
        int lda,
        int[] ipiv,
        double[] b,
        int ldb
    )
    
    // QR decomposition
    [<DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl)>]
    extern int LAPACKE_dgeqrf(
        int matrix_layout,
        int m,
        int n,
        double[] a,
        int lda,
        double[] tau
    )
    
    [<DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl)>]
    extern int LAPACKE_dorgqr(
        int matrix_layout,
        int m,
        int n,
        int k,
        double[] a,
        int lda,
        double[] tau
    )
    
    // SVD decomposition
    [<DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl)>]
    extern int LAPACKE_dgesvd(
        int matrix_layout,
        char jobu,
        char jobvt,
        int m,
        int n,
        double[] a,
        int lda,
        double[] s,
        double[] u,
        int ldu,
        double[] vt,
        int ldvt,
        double[] superb
    )
    
    // Cholesky decomposition
    [<DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl)>]
    extern int LAPACKE_dpotrf(
        int matrix_layout,
        char uplo,
        int n,
        double[] a,
        int lda
    )
    
    // Eigenvalue decomposition for symmetric matrices
    [<DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl)>]
    extern int LAPACKE_dsyev(
        int matrix_layout,
        char jobz,
        char uplo,
        int n,
        double[] a,
        int lda,
        double[] w
    )
    
    let lapackAvailable = true
with
| _ -> 
    printfn "Warning: LAPACK not available. Linear algebra functions will not work."
    let lapackAvailable = false
    
    // Dummy implementations
    let LAPACKE_dgetrf(_,_,_,_,_,_) = -1
    let LAPACKE_dgetri(_,_,_,_,_) = -1
    let LAPACKE_dgesv(_,_,_,_,_,_,_,_) = -1
    let LAPACKE_dgeqrf(_,_,_,_,_,_) = -1
    let LAPACKE_dorgqr(_,_,_,_,_,_,_) = -1
    let LAPACKE_dgesvd(_,_,_,_,_,_,_,_,_,_,_,_,_) = -1
    let LAPACKE_dpotrf(_,_,_,_,_) = -1
    let LAPACKE_dsyev(_,_,_,_,_,_,_) = -1

/// Verify LAPACK is available
let ensureLapackAvailable() : FowlResult<unit> =
    if lapackAvailable then
        Ok ()
    else
        Error.nativeLibraryError """
LAPACK not available. Please install OpenBLAS:

macOS:    brew install openblas
Ubuntu:   sudo apt-get install libopenblas-dev
Fedora:   sudo dnf install openblas-devel
Windows:  https://github.com/xianyi/OpenBLAS/releases
"""

/// LU decomposition with partial pivoting
/// Returns (permutation_indices, L_matrix, U_matrix)
let lu (a: Ndarray<'K, float>) : FowlResult<int array * Ndarray<'K, float> * Ndarray<'K, float>> =
    ensureLapackAvailable()
    |> Result.bind (fun () ->
        match a with
        | Dense da ->
            match da.Shape with
            | [|m; n|] ->
                let minMN = min m n
                let ipiv = Array.zeroCreate minMN
                let aCopy = Array.copy da.Data
                
                let info = LAPACKE_dgetrf(101, m, n, aCopy, n, ipiv)
                if info < 0 then
                    Error.invalidArgument (sprintf "LAPACK dgetrf error: argument %d invalid" (-info))
                elif info > 0 then
                    Error.invalidState (sprintf "LAPACK dgetrf error: U is singular (zero at diagonal %d)" info)
                else
                    // Extract L and U from the factorized matrix
                    let lData = Array.zeroCreate (m * minMN)
                    let uData = Array.zeroCreate (minMN * n)
                    
                    // L has 1s on diagonal and subdiagonal elements from aCopy
                    for i = 0 to m - 1 do
                        for j = 0 to min minMN (i+1) - 1 do
                            if i = j then
                                lData.[i * minMN + j] <- 1.0
                            else
                                lData.[i * minMN + j] <- aCopy.[i * n + j]
                    
                    // U has diagonal and superdiagonal elements from aCopy
                    for i = 0 to minMN - 1 do
                        for j = i to n - 1 do
                            uData.[i * n + j] <- aCopy.[i * n + j]
                    
                    let lResult = Ndarray.ofArray lData [|m; minMN|]
                    let uResult = Ndarray.ofArray uData [|minMN; n|]
                    
                    match lResult, uResult with
                    | Ok l, Ok u -> Ok (ipiv, l, u)
                    | Error e, _ -> Error e
                    | _, Error e -> Error e
            | _ -> Error.invalidArgument "lu requires 2D matrix"
        | _ -> Error.notImplemented "lu not implemented for sparse arrays")

/// Solve linear system A*X = B using LU factorization
let solve (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
    ensureLapackAvailable()
    |> Result.bind (fun () ->
        match a, b with
        | Dense da, Dense db ->
            match da.Shape, db.Shape with
            | [|n; n2|], [|n3; nrhs|] when n = n2 && n = n3 ->
                let ipiv = Array.zeroCreate n
                let aCopy = Array.copy da.Data
                let bCopy = Array.copy db.Data
                
                let info = LAPACKE_dgesv(101, n, nrhs, aCopy, n, ipiv, bCopy, nrhs)
                if info < 0 then
                    Error.invalidArgument (sprintf "LAPACK dgesv error: argument %d invalid" (-info))
                elif info > 0 then
                    Error.invalidState (sprintf "LAPACK dgesv error: singular matrix (diagonal %d is zero)" info)
                else
                    Ndarray.ofArray bCopy db.Shape
            | [|n; n2|], [|n3|] when n = n2 && n = n3 ->
                // Single right-hand side - convert to 2D
                match Ndarray.reshape [|n; 1|] b with
                | Ok b2d ->
                    match solve a b2d with
                    | Ok result -> Ndarray.reshape [|n|] result
                    | Error e -> Error e
                | Error e -> Error e
            | _ -> Error.dimensionMismatch (sprintf "solve: incompatible shapes %A and %A" da.Shape db.Shape)
        | _ -> Error.notImplemented "solve not implemented for sparse arrays")

/// Matrix inverse using LU factorization
let inv (a: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
    ensureLapackAvailable()
    |> Result.bind (fun () ->
        match a with
        | Dense da ->
            match da.Shape with
            | [|n; n2|] when n = n2 ->
                let ipiv = Array.zeroCreate n
                let aCopy = Array.copy da.Data
                
                // Factorize
                let info1 = LAPACKE_dgetrf(101, n, n, aCopy, n, ipiv)
                if info1 <> 0 then
                    Error.invalidState "inv: matrix is singular"
                else
                    // Invert
                    let info2 = LAPACKE_dgetri(101, n, aCopy, n, ipiv)
                    if info2 < 0 then
                        Error.invalidArgument (sprintf "LAPACK dgetri error: argument %d invalid" (-info2))
                    elif info2 > 0 then
                        Error.invalidState "inv: matrix is singular"
                    else
                        Ndarray.ofArray aCopy da.Shape
            | _ -> Error.invalidArgument "inv requires square matrix"
        | _ -> Error.notImplemented "inv not implemented for sparse arrays")

/// Matrix determinant using LU factorization
let det (a: Ndarray<'K, float>) : FowlResult<float> =
    match a with
    | Dense da ->
        match da.Shape with
        | [|n; n2|] when n = n2 ->
            match lu a with
            | Ok (ipiv, l, u) ->
                // Determinant is product of diagonal of U, with sign changes from permutations
                let mutable d = 1.0
                for i = 0 to n - 1 do
                    match Ndarray.get u [|i; i|] with
                    | Ok v -> d <- d * v
                    | Error _ -> ()
                
                // Count row swaps
                let swaps = 
                    ipiv 
                    |> Array.mapi (fun i piv -> if piv <> i + 1 then 1 else 0)
                    |> Array.sum
                
                Ok (if swaps % 2 = 1 then -d else d)
            | Error e -> Error e
        | _ -> Error.invalidArgument "det requires square matrix"
    | _ -> Error.notImplemented "det not implemented for sparse arrays"

/// QR decomposition
/// Returns (Q, R) where A = QR, Q is orthogonal, R is upper triangular
let qr (a: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float> * Ndarray<'K, float>> =
    ensureLapackAvailable()
    |> Result.bind (fun () ->
        match a with
        | Dense da ->
            match da.Shape with
            | [|m; n|] ->
                let aCopy = Array.copy da.Data
                let k = min m n
                let tau = Array.zeroCreate k
                
                // Compute QR factorization
                let info = LAPACKE_dgeqrf(101, m, n, aCopy, n, tau)
                if info < 0 then
                    Error.invalidArgument (sprintf "LAPACK dgeqrf error: argument %d invalid" (-info))
                elif info > 0 then
                    Error.invalidState "qr: matrix is singular"
                else
                    // Extract R (upper triangular part of aCopy)
                    let rData = Array.zeroCreate (k * n)
                    for i = 0 to k - 1 do
                        for j = i to n - 1 do
                            rData.[i * n + j] <- aCopy.[i * n + j]
                    
                    // Generate Q
                    let info2 = LAPACKE_dorgqr(101, m, k, k, aCopy, n, tau)
                    if info2 <> 0 then
                        Error.invalidState (sprintf "LAPACK dorgqr error: %d" info2)
                    else
                        // Extract Q (first k columns)
                        let qData = Array.zeroCreate (m * k)
                        for i = 0 to m - 1 do
                            for j = 0 to k - 1 do
                                qData.[i * k + j] <- aCopy.[i * n + j]
                        
                        let qResult = Ndarray.ofArray qData [|m; k|]
                        let rResult = Ndarray.ofArray rData [|k; n|]
                        
                        match qResult, rResult with
                        | Ok q, Ok r -> Ok (q, r)
                        | Error e, _ -> Error e
                        | _, Error e -> Error e
            | _ -> Error.invalidArgument "qr requires 2D matrix"
        | _ -> Error.notImplemented "qr not implemented for sparse arrays")

/// SVD decomposition
/// Returns (U, S, Vt) where A = U * diag(S) * Vt
let svd (a: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float> * Ndarray<'K, float> * Ndarray<'K, float>> =
    ensureLapackAvailable()
    |> Result.bind (fun () ->
        match a with
        | Dense da ->
            match da.Shape with
            | [|m; n|] ->
                let aCopy = Array.copy da.Data
                let k = min m n
                let s = Array.zeroCreate k
                let u = Array.zeroCreate (m * m)
                let vt = Array.zeroCreate (n * n)
                let superb = Array.zeroCreate (k - 1)
                
                // Compute SVD
                let info = LAPACKE_dgesvd(101, 'A', 'A', m, n, aCopy, n, s, u, m, vt, n, superb)
                if info < 0 then
                    Error.invalidArgument (sprintf "LAPACK dgesvd error: argument %d invalid" (-info))
                elif info > 0 then
                    Error.invalidState (sprintf "SVD did not converge, superdiagonal element %d did not converge" info)
                else
                    let uResult = Ndarray.ofArray u [|m; m|]
                    let sResult = Ndarray.ofArray s [|k|]
                    let vtResult = Ndarray.ofArray vt [|n; n|]
                    
                    match uResult, sResult, vtResult with
                    | Ok u', Ok s', Ok vt' -> Ok (u', s', vt')
                    | Error e, _, _ -> Error e
                    | _, Error e, _ -> Error e
                    | _, _, Error e -> Error e
            | _ -> Error.invalidArgument "svd requires 2D matrix"
        | _ -> Error.notImplemented "svd not implemented for sparse arrays")

/// Cholesky decomposition for positive definite matrices
/// Returns L where A = L * L^T (lower triangular)
let cholesky (a: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
    ensureLapackAvailable()
    |> Result.bind (fun () ->
        match a with
        | Dense da ->
            match da.Shape with
            | [|n; n2|] when n = n2 ->
                let aCopy = Array.copy da.Data
                
                let info = LAPACKE_dpotrf(101, 'L', n, aCopy, n)
                if info < 0 then
                    Error.invalidArgument (sprintf "LAPACK dpotrf error: argument %d invalid" (-info))
                elif info > 0 then
                    Error.invalidState (sprintf "cholesky: matrix is not positive definite (leading minor %d is not positive)" info)
                else
                    // Zero out upper triangle to get clean L
                    for i = 0 to n - 1 do
                        for j = i + 1 to n - 1 do
                            aCopy.[i * n + j] <- 0.0
                    Ndarray.ofArray aCopy da.Shape
            | _ -> Error.invalidArgument "cholesky requires square matrix"
        | _ -> Error.notImplemented "cholesky not implemented for sparse arrays")

/// Eigenvalue decomposition for symmetric matrices
/// Returns (eigenvalues, eigenvectors) where A * V = V * diag(D)
let eigSymmetric (a: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float> * Ndarray<'K, float>> =
    ensureLapackAvailable()
    |> Result.bind (fun () ->
        match a with
        | Dense da ->
            match da.Shape with
            | [|n; n2|] when n = n2 ->
                let aCopy = Array.copy da.Data
                let w = Array.zeroCreate n
                
                let info = LAPACKE_dsyev(101, 'V', 'U', n, aCopy, n, w)
                if info < 0 then
                    Error.invalidArgument (sprintf "LAPACK dsyev error: argument %d invalid" (-info))
                elif info > 0 then
                    Error.invalidState (sprintf "Eigenvalue decomposition did not converge, off-diagonal element %d did not converge" info)
                else
                    let eigenvaluesResult = Ndarray.ofArray w [|n|]
                    let eigenvectorsResult = Ndarray.ofArray aCopy da.Shape
                    
                    match eigenvaluesResult, eigenvectorsResult with
                    | Ok w', Ok v' -> Ok (w', v')
                    | Error e, _ -> Error e
                    | _, Error e -> Error e
            | _ -> Error.invalidArgument "eigSymmetric requires square matrix"
        | _ -> Error.notImplemented "eigSymmetric not implemented for sparse arrays")
