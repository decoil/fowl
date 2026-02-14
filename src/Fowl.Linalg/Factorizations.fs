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
    
    let lapackAvailable = true
with
| _ -> 
    printfn "Warning: LAPACK not available. Linear algebra functions will not work."
    let lapackAvailable = false
    
    // Dummy implementations
    let LAPACKE_dgetrf(_,_,_,_,_,_) = -1
    let LAPACKE_dgetri(_,_,_,_,_) = -1
    let LAPACKE_dgesv(_,_,_,_,_,_,_,_) = -1

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
