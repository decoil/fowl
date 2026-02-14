module Fowl.Native.Blas

open System
open System.Runtime.InteropServices
open Fowl

/// BLAS order constants
module Order =
    let RowMajor = 101
    let ColMajor = 102

/// BLAS transpose constants
module Transpose =
    let NoTrans = 111
    let Trans = 112
    let ConjTrans = 113

/// Native BLAS bindings via OpenBLAS
module Native =
    [<DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl)>]
    extern void cblas_dgemm(
        int Order,
        int TransA,
        int TransB,
        int M,
        int N,
        int K,
        double alpha,
        double[] A,
        int lda,
        double[] B,
        int ldb,
        double beta,
        double[] C,
        int ldc
    )
    
    [<DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl)>]
    extern void cblas_sgemm(
        int Order,
        int TransA,
        int TransB,
        int M,
        int N,
        int K,
        float alpha,
        float[] A,
        int lda,
        float[] B,
        int ldb,
        float beta,
        float[] C,
        int ldc
    )
    
    [<DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl)>]
    extern double cblas_ddot(
        int N,
        double[] X,
        int incX,
        double[] Y,
        int incY
    )
    
    [<DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl)>]
    extern void cblas_daxpy(
        int N,
        double alpha,
        double[] X,
        int incX,
        double[] Y,
        int incY
    )
    
    [<DllImport("libopenblas", CallingConvention = CallingConvention.Cdecl)>]
    extern void cblas_dscal(
        int N,
        double alpha,
        double[] X,
        int incX
    )

/// High-level BLAS operations
module Operations =
    /// Matrix-matrix multiply: C = alpha*A*B + beta*C
    let gemm (alpha: float) (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) (beta: float) (c: Ndarray<'K, float>) : Ndarray<'K, float> =
        match a, b, c with
        | Dense da, Dense db, Dense dc ->
            match da.Shape, db.Shape with
            | [|m; k1|], [|k2; n|] when k1 = k2 ->
                let cOut = if beta = 0.0 then Ndarray.zeros dc.Shape else Ndarray.create dc.Shape beta
                
                Native.cblas_dgemm(
                    Order.RowMajor,
                    Transpose.NoTrans,
                    Transpose.NoTrans,
                    m, n, k1,
                    alpha,
                    da.Data, k1,
                    db.Data, n,
                    beta,
                    cOut |> function Dense d -> d.Data | _ -> failwith "Expected dense", n
                )
                cOut
            | _ -> failwithf "Incompatible shapes for gemm: %A and %A" da.Shape db.Shape
        | _ -> failwith "gemm only implemented for dense arrays"
    
    /// Simplified matrix multiply: C = A*B
    let matmulBlas (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : Ndarray<'K, float> =
        match a, b with
        | Dense da, Dense db ->
            match da.Shape, db.Shape with
            | [|m; k1|], [|k2; n|] when k1 = k2 ->
                let c = Ndarray.zeros<'K> [|m; n|]
                gemm 1.0 a b 0.0 c
            | _ -> failwithf "Incompatible shapes: %A and %A" da.Shape db.Shape
        | _ -> failwith "matmulBlas only for dense arrays"
    
    /// Dot product of two vectors
    let dotBlas (x: Ndarray<'K, float>) (y: Ndarray<'K, float>) : float =
        match x, y with
        | Dense dx, Dense dy ->
            if dx.Shape.Length <> 1 || dy.Shape.Length <> 1 then
                failwith "dotBlas requires 1D arrays"
            if dx.Shape.[0] <> dy.Shape.[0] then
                failwith "dotBlas: length mismatch"
            
            Native.cblas_ddot(dx.Shape.[0], dx.Data, 1, dy.Data, 1)
        | _ -> failwith "dotBlas only for dense arrays"
    
    /// Y = alpha*X + Y (AXPY)
    let axpy (alpha: float) (x: Ndarray<'K, float>) (y: Ndarray<'K, float>) : unit =
        match x, y with
        | Dense dx, Dense dy ->
            if dx.Shape <> dy.Shape then
                failwith "axpy: shape mismatch"
            Native.cblas_daxpy(Shape.numel dx.Shape, alpha, dx.Data, 1, dy.Data, 1)
        | _ -> failwith "axpy only for dense arrays"
    
    /// X = alpha*X (scale in place)
    let scal (alpha: float) (x: Ndarray<'K, float>) : unit =
        match x with
        | Dense dx ->
            Native.cblas_dscal(Shape.numel dx.Shape, alpha, dx.Data, 1)
        | _ -> failwith "scal only for dense arrays"
