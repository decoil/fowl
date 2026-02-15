/// <summary>
/// Fowl - Functional Numerical Computing for F#
/// 
/// Core module providing the foundation for all numerical operations.
/// Design based on:
/// - Architecture of Advanced Numerical Analysis Systems (Liang Wang)
/// - Owl (OCaml numerical library)
/// - F# for Fun and Profit (Scott Wlaschin)
/// </summary>
namespace Fowl

open System

// ============================================================================
// Phantom Types for Type Safety
// ============================================================================

/// <summary>Phantom type for 32-bit floating point.</summary>
type Float32 = class end

/// <summary>Phantom type for 64-bit floating point (double precision).</summary>
type Float64 = class end

/// <summary>Phantom type for 32-bit complex numbers.</summary>
type Complex32 = class end

/// <summary>Phantom type for 64-bit complex numbers.</summary>
type Complex64 = class end

// ============================================================================
// Core Type Definitions
// ============================================================================

/// <summary>Shape is an array of dimension sizes.</summary>
/// <remarks>For example, a 3x4 matrix has shape [|3; 4|].</remarks>
type Shape = int array

/// <summary>Memory layout for array storage.</summary>
type Layout =
    | CLayout       /// Row-major (C-style)
    | FortranLayout /// Column-major (Fortran-style)

/// <summary>Dense array storage.</summary>
type DenseArray<'T> = {
    Data: 'T array
    Shape: Shape
    Strides: int array
    Offset: int
    Layout: Layout
}

/// <summary>Sparse array formats.</summary>
type SparseFormat =
    | CSR /// Compressed Sparse Row
    | CSC /// Compressed Sparse Column
    | COO /// Coordinate format

/// <summary>Sparse array storage.</summary>
type SparseArray<'T> = {
    Indices: int array array
    Values: 'T array
    Shape: Shape
    Format: SparseFormat
}

/// <summary>N-dimensional array with phantom type for kind.</summary>
/// <typeparam name="'K">Phantom type for element kind (Float32, Float64, etc.).</typeparam>
/// <typeparam name="'T">Actual element type (float32, float, Complex, etc.).</typeparam>
type Ndarray<'K, 'T> =
    | Dense of DenseArray<'T>
    | Sparse of SparseArray<'T>

/// <summary>Type alias for common float64 arrays.</summary>
type Float64Ndarray = Ndarray<Float64, float>

// ============================================================================
// Error Handling
// ============================================================================

/// <summary>Error types for Fowl operations.
/// All operations return Result types with these error variants.
/// </summary>
type FowlError =
    | InvalidShape of string
    | DimensionMismatch of string
    | IndexOutOfRange of string
    | InvalidArgument of string
    | NotImplemented of string
    | InvalidState of string
    | NativeLibraryError of string

/// <summary>Result type alias for Fowl operations.
/// </summary>
type FowlResult<'T> = Result<'T, FowlError>

/// <summary>Error handling utilities.
/// </summary>
module Error =
    let inline invalidShape msg = Error (InvalidShape msg)
    let inline dimensionMismatch msg = Error (DimensionMismatch msg)
    let inline indexOutOfRange msg = Error (IndexOutOfRange msg)
    let inline invalidArgument msg = Error (InvalidArgument msg)
    let inline notImplemented msg = Error (NotImplemented msg)
    let inline invalidState msg = Error (InvalidState msg)
    let inline nativeLibraryError msg = Error (NativeLibraryError msg)

/// <summary>Result computation expression builder.
/// </summary>
type ResultBuilder() =
    member _.Bind(m, f) = Result.bind f m
    member _.Return(x) = Ok x
    member _.ReturnFrom(m) = m
    member _.Zero() = Ok ()
    member _.Combine(a, b) = Result.bind (fun _ -> b) a
    member _.Delay(f) = f()
    member _.For(sequence, body) =
        sequence |> Seq.fold (fun acc item -> Result.bind (fun _ -> body item) acc) (Ok ())
    member _.While(guard, body) =
        let rec loop () =
            if guard() then Result.bind (fun _ -> loop()) (body())
            else Ok ()
        loop()

/// <summary>Global result computation expression.
/// </summary>
let result = ResultBuilder()

// ============================================================================
// Shape Operations
// ============================================================================

/// <summary>Operations on array shapes.
/// </summary>
module Shape =
    /// <summary>Calculate total number of elements.
    /// </summary>
    let numel (shape: Shape) : int =
        if shape.Length = 0 then 0
        else shape |> Array.fold (*) 1
    
    /// <summary>Calculate strides for C-layout (row-major).
    /// </summary>
    let stridesC (shape: Shape) : int array =
        let n = shape.Length
        let s = Array.zeroCreate n
        let mutable stride = 1
        for i = n - 1 downto 0 do
            s.[i] <- stride
            stride <- stride * shape.[i]
        s
    
    /// <summary>Calculate strides for Fortran-layout (column-major).
    /// </summary>
    let stridesF (shape: Shape) : int array =
        let n = shape.Length
        let s = Array.zeroCreate n
        let mutable stride = 1
        for i = 0 to n - 1 do
            s.[i] <- stride
            stride <- stride * shape.[i]
        s
    
    /// <summary>Validate that shape is valid (all dimensions positive).
    /// </summary>
    let validate (shape: Shape) : FowlResult<Shape> =
        if shape |> Array.exists (fun d -> d <= 0) then
            Error.invalidShape "Shape dimensions must be positive"
        else
            Ok shape

// ============================================================================
// Ndarray Operations
// ============================================================================

/// <summary>Core operations on N-dimensional arrays.
/// </summary>
module Ndarray =
    open Shape
    
    /// <summary>Get the shape of an array.
    /// </summary>
    let shape (arr: Ndarray<'K, 'T>) : Shape =
        match arr with
        | Dense d -> d.Shape
        | Sparse s -> s.Shape
    
    /// <summary>Get number of dimensions.
    /// </summary>
    let ndim (arr: Ndarray<'K, 'T>) : int =
        (shape arr).Length
    
    /// <summary>Get total number of elements.
    /// </summary>
    let numel (arr: Ndarray<'K, 'T>) : int =
        shape arr |> Shape.numel
    
    /// <summary>Create empty array with given shape.
    /// </summary>
    let empty<'K, 'T when 'T: (new: unit -> 'T) and 'T: struct> 
            (shape: Shape) : FowlResult<Ndarray<'K, 'T>> =
        result {
            let! validShape = Shape.validate shape
            let n = Shape.numel validShape
            let data = Array.zeroCreate n
            return Dense {
                Data = data
                Shape = validShape
                Strides = stridesC validShape
                Offset = 0
                Layout = CLayout
            }
        }
    
    /// <summary>Create array filled with zeros.
    /// </summary>
    let zeros (shape: Shape) : FowlResult<Float64Ndarray> =
        empty<Float64, float> shape
    
    /// <summary>Create array filled with ones.
    /// </summary>
    let ones (shape: Shape) : FowlResult<Float64Ndarray> =
        result {
            let! arr = empty<Float64, float> shape
            match arr with
            | Dense d ->
                Array.fill d.Data 0 d.Data.Length 1.0
                return Dense d
            | Sparse _ ->
                return! Error.notImplemented "ones for sparse arrays"
        }
    
    /// <summary>Create array from flat data.
    /// </summary>
    let ofArray<'K, 'T> (data: 'T array) (shape: Shape) : FowlResult<Ndarray<'K, 'T>> =
        result {
            let! validShape = Shape.validate shape
            let expectedSize = Shape.numel validShape
            if data.Length <> expectedSize then
                return! Error.invalidShape 
                    (sprintf "Data length %d does not match shape %A" data.Length validShape)
            return Dense {
                Data = Array.copy data
                Shape = validShape
                Strides = stridesC validShape
                Offset = 0
                Layout = CLayout
            }
        }
    
    /// <summary>Convert to flat array (copy).
    /// </summary>
    let toArray (arr: Ndarray<'K, 'T>) : FowlResult<'T array> =
        match arr with
        | Dense d -> Ok (Array.copy d.Data)
        | Sparse _ -> Error.notImplemented "toArray for sparse arrays"
    
    /// <summary>Map function over all elements.
    /// </summary>
    let map<'K, 'T, 'U> (f: 'T -> 'U) (arr: Ndarray<'K, 'T>) : FowlResult<Ndarray<'K, 'U>> =
        match arr with
        | Dense d ->
            let newData = Array.map f d.Data
            ofArray newData d.Shape
        | Sparse _ -> Error.notImplemented "map for sparse arrays"
    
    /// <summary>Element-wise addition.
    /// </summary>
    let add (a: Float64Ndarray) (b: Float64Ndarray) : FowlResult<Float64Ndarray> =
        result {
            let shapeA = shape a
            let shapeB = shape b
            if shapeA <> shapeB then
                return! Error.dimensionMismatch 
                    (sprintf "Shape mismatch: %A vs %A" shapeA shapeB)
            let! dataA = toArray a
            let! dataB = toArray b
            let result = Array.map2 (+) dataA dataB
            return! ofArray result shapeA
        }
    
    /// <summary>Element-wise multiplication.
    /// </summary>
    let mul (a: Float64Ndarray) (b: Float64Ndarray) : FowlResult<Float64Ndarray> =
        result {
            let shapeA = shape a
            let shapeB = shape b
            if shapeA <> shapeB then
                return! Error.dimensionMismatch 
                    (sprintf "Shape mismatch: %A vs %A" shapeA shapeB)
            let! dataA = toArray a
            let! dataB = toArray b
            let result = Array.map2 (*) dataA dataB
            return! ofArray result shapeA
        }
    
    /// <summary>Sum all elements.
    /// </summary>
    let sum (arr: Float64Ndarray) : FowlResult<float> =
        result {
            let! data = toArray arr
            return Array.sum data
        }
    
    /// <summary>Mean of all elements.
    /// </summary>
    let mean (arr: Float64Ndarray) : FowlResult<float> =
        result {
            let! s = sum arr
            let n = numel arr
            if n = 0 then
                return! Error.invalidState "Cannot compute mean of empty array"
            return s / float n
        }

// ============================================================================
// Linear Algebra
// ============================================================================

/// <summary>Matrix operations.
/// </summary>
module Matrix =
    open Ndarray
    
    /// <summary>Matrix multiplication.
    /// </summary>
    let matmul (a: Float64Ndarray) (b: Float64Ndarray) : FowlResult<Float64Ndarray> =
        result {
            let shapeA = Ndarray.shape a
            let shapeB = Ndarray.shape b
            
            // Validate 2D matrices
            if shapeA.Length <> 2 || shapeB.Length <> 2 then
                return! Error.invalidShape "matmul requires 2D matrices"
            
            let m = shapeA.[0]
            let n = shapeA.[1]
            let p = shapeB.[1]
            
            if n <> shapeB.[0] then
                return! Error.dimensionMismatch 
                    (sprintf "Inner dimensions mismatch: %d vs %d" n shapeB.[0])
            
            let! dataA = Ndarray.toArray a
            let! dataB = Ndarray.toArray b
            let resultData = Array.zeroCreate (m * p)
            
            // Naive matrix multiplication
            for i = 0 to m - 1 do
                for j = 0 to p - 1 do
                    let mutable sum = 0.0
                    for k = 0 to n - 1 do
                        sum <- sum + dataA.[i * n + k] * dataB.[k * p + j]
                    resultData.[i * p + j] <- sum
            
            return! Ndarray.ofArray resultData [|m; p|]
        }
    
    /// <summary>Transpose matrix.
    /// </summary>
    let transpose (arr: Float64Ndarray) : FowlResult<Float64Ndarray> =
        result {
            let shape = Ndarray.shape arr
            if shape.Length <> 2 then
                return! Error.invalidShape "transpose requires 2D matrix"
            let m = shape.[0]
            let n = shape.[1]
            let! data = Ndarray.toArray arr
            let result = Array.zeroCreate (m * n)
            for i = 0 to m - 1 do
                for j = 0 to n - 1 do
                    result.[j * m + i] <- data.[i * n + j]
            return! Ndarray.ofArray result [|n; m|]
        }

// ============================================================================
// Generation Functions
// ============================================================================

/// <summary>Array generation functions.
/// </summary>
module Generate =
    /// <summary>Generate linearly spaced values.
    /// </summary>
    let linspace (start: float) (stop: float) (num: int) : FowlResult<Float64Ndarray> =
        result {
            if num < 2 then
                return! Error.invalidArgument "linspace requires num >= 2"
            let step = (stop - start) / float (num - 1)
            let data = Array.init num (fun i -> start + float i * step)
            return! Ndarray.ofArray data [|num|]
        }
    
    /// <summary>Generate values with given step.
    /// </summary>
    let arange (start: float) (stop: float) (step: float) : FowlResult<Float64Ndarray> =
        result {
            if step = 0.0 then
                return! Error.invalidArgument "arange: step cannot be zero"
            let n = int ((stop - start) / step)
            if n <= 0 then
                return! Error.invalidArgument "arange: invalid range"
            let data = Array.init n (fun i -> start + float i * step)
            return! Ndarray.ofArray data [|n|]
        }
    
    /// <summary>Generate random values (uniform [0, 1)).
    /// </summary>
    let random (shape: Shape) : FowlResult<Float64Ndarray> =
        result {
            let! validShape = Shape.validate shape
            let n = Shape.numel validShape
            let rng = System.Random()
            let data = Array.init n (fun _ -> rng.NextDouble())
            return! Ndarray.ofArray data validShape
        }
