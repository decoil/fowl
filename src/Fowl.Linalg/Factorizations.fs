module Fowl.Linalg.Factorizations

open System
open System.Runtime.InteropServices
open Fowl

/// Native LAPACK bindings
module Native =
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

/// LU decomposition with partial pivoting
/// Returns (permutation_indices, L_matrix, U_matrix)
let lu (a: Ndarray<'K, float>) : int array * Ndarray<'K, float> * Ndarray<'K, float> =
    match a with
    | Dense da ->
        match da.Shape with
        | [|m; n|] ->
            let minMN = min m n
            let ipiv = Array.zeroCreate minMN
            let aCopy = Array.copy da.Data
            
            let info = Native.LAPACKE_dgetrf(101, m, n, aCopy, n, ipiv)
            if info < 0 then
                failwithf "LAPACK dgetrf error: argument %d invalid" (-info)
            elif info > 0 then
                failwithf "LAPACK dgetrf error: U is singular (zero at diagonal %d)" info
            
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
            
            let l = Ndarray.ofArray lData [|m; minMN|]
            let u = Ndarray.ofArray uData [|minMN; n|]
            
            (ipiv, l, u)
        | _ -> failwith "lu requires 2D matrix"
    | _ -> failwith "lu not implemented for sparse arrays"

/// Solve linear system A*X = B using LU factorization
let solve (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : Ndarray<'K, float> =
    match a, b with
    | Dense da, Dense db ->
        match da.Shape, db.Shape with
        | [|n; n2|], [|n3; nrhs|] when n = n2 && n = n3 ->
            let ipiv = Array.zeroCreate n
            let aCopy = Array.copy da.Data
            let bCopy = Array.copy db.Data
            
            let info = Native.LAPACKE_dgesv(101, n, nrhs, aCopy, n, ipiv, bCopy, nrhs)
            if info < 0 then
                failwithf "LAPACK dgesv error: argument %d invalid" (-info)
            elif info > 0 then
                failwithf "LAPACK dgesv error: singular matrix (diagonal %d is zero)" info
            
            Ndarray.ofArray bCopy db.Shape
        | [|n; n2|], [|n3|] when n = n2 && n = n3 ->
            // Single right-hand side
            let b2d = Ndarray.reshape [|n; 1|] b
            solve a b2d |> Ndarray.reshape [|n|]
        | _ -> failwithf "solve: incompatible shapes %A and %A" da.Shape db.Shape
    | _ -> failwith "solve not implemented for sparse arrays"

/// Matrix inverse using LU factorization
let inv (a: Ndarray<'K, float>) : Ndarray<'K, float> =
    match a with
    | Dense da ->
        match da.Shape with
        | [|n; n2|] when n = n2 ->
            let ipiv = Array.zeroCreate n
            let aCopy = Array.copy da.Data
            
            // Factorize
            let info1 = Native.LAPACKE_dgetrf(101, n, n, aCopy, n, ipiv)
            if info1 <> 0 then
                failwith "inv: matrix is singular"
            
            // Invert
            let info2 = Native.LAPACKE_dgetri(101, n, aCopy, n, ipiv)
            if info2 < 0 then
                failwithf "LAPACK dgetri error: argument %d invalid" (-info2)
            elif info2 > 0 then
                failwith "inv: matrix is singular"
            
            Ndarray.ofArray aCopy da.Shape
        | _ -> failwith "inv requires square matrix"
    | _ -> failwith "inv not implemented for sparse arrays"

/// Matrix determinant using LU factorization
let det (a: Ndarray<'K, float>) : float =
    match a with
    | Dense da ->
        match da.Shape with
        | [|n; n2|] when n = n2 ->
            let ipiv, l, u = lu a
            
            // Determinant is product of diagonal of U, with sign changes from permutations
            let mutable d = 1.0
            for i = 0 to n - 1 do
                d <- d * (Ndarray.get u [|i; i|])
            
            // Count row swaps
            let swaps = 
                ipiv 
                |> Array.mapi (fun i piv -> if piv <> i + 1 then 1 else 0)
                |> Array.sum
            
            if swaps % 2 = 1 then -d else d
        | _ -> failwith "det requires square matrix"
    | _ -> failwith "det not implemented for sparse arrays"
