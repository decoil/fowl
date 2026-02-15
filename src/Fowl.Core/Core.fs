/// <summary>
/// Fowl - Functional Numerical Computing for F#
/// 
/// Clean implementation based on:
/// - Architecture of Advanced Numerical Analysis Systems (Liang Wang)
/// - Owl (OCaml numerical library)
/// - F# for Fun and Profit (Scott Wlaschin)
/// 
/// Core module providing foundation types and operations.
/// </summary>
namespace Fowl

open System

// ============================================================================
// Phantom Types for Type Safety
// ============================================================================

/// <summary>Phantom type for 32-bit floating point.
/// </summary>
type Float32 = class end

/// <summary>Phantom type for 64-bit floating point (double precision).
/// </summary>
type Float64 = class end

/// <summary>Phantom type for 32-bit complex numbers.
/// </summary>
type Complex32 = class end

/// <summary>Phantom type for 64-bit complex numbers.
/// </summary>
type Complex64 = class end

// ============================================================================
// Core Type Definitions
// ============================================================================

/// <summary>Shape is an array of dimension sizes. For example, a 3x4 matrix has shape [|3; 4|].
/// </summary>
type Shape = int array

/// <summary>Memory layout for array storage.
/// </summary>
type Layout =
    | CLayout       /// Row-major (C-style)
    | FortranLayout /// Column-major (Fortran-style)

/// <summary>Dense array storage with shape and strides.
/// </summary>
type DenseArray<'T> = {
    Data: 'T array
    Shape: Shape
    Strides: int array
    Offset: int
    Layout: Layout
}

/// <summary>Sparse array formats.
/// </summary>
type SparseFormat =
    | CSR /// Compressed Sparse Row
    | CSC /// Compressed Sparse Column
    | COO /// Coordinate format

/// <summary>Sparse array storage.
/// </summary>
type SparseArray<'T> = {
    Indices: int array array
    Values: 'T array
    Shape: Shape
    Format: SparseFormat
}

/// <summary>N-dimensional array with phantom type for kind.
/// <typeparam name="'K">Phantom type for element kind (Float32, Float64, etc.).</typeparam>
/// <typeparam name="'T">Actual element type (float32, float, Complex, etc.).</typeparam>
type Ndarray<'K, 'T> =
    | Dense of DenseArray<'T>
    | Sparse of SparseArray<'T>

/// <summary>Type alias for float64 arrays.
/// </summary>
type Float64Ndarray = Ndarray<Float64, float>

// ============================================================================
// Error Handling
// ============================================================================

/// <summary>Error types for Fowl operations.
/// All operations return Result types with these error variants for safe error handling.
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
    /// <summary>Create an InvalidShape error.
    /// </summary>
    let inline invalidShape msg = Error (InvalidShape msg)
    
    /// <summary>Create a DimensionMismatch error.
    /// </summary>
    let inline dimensionMismatch msg = Error (DimensionMismatch msg)
    
    /// <summary>Create an IndexOutOfRange error.
    /// </summary>
    let inline indexOutOfRange msg = Error (IndexOutOfRange msg)
    
    /// <summary>Create an InvalidArgument error.
    /// </summary>
    let inline invalidArgument msg = Error (InvalidArgument msg)
    
    /// <summary>Create a NotImplemented error.
    /// </summary>
    let inline notImplemented msg = Error (NotImplemented msg)
    
    /// <summary>Create an InvalidState error.
    /// </summary>
    let inline invalidState msg = Error (InvalidState msg)
    
    /// <summary>Create a NativeLibraryError.
    /// </summary>
    let inline nativeLibraryError msg = Error (NativeLibraryError msg)

/// <summary>Computation expression builder for Result type.
/// Enables clean monadic syntax for error handling.
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

// ============================================================================
// Shape Operations
// ============================================================================

/// <summary>Operations on array shapes.
/// </summary>
module Shape =
    /// <summary>Calculate total number of elements from shape.
    /// </summary>
    /// <param name="shape">The shape array.</param>
    /// <returns>Total number of elements.</returns>
    let numel (shape: Shape) : int =
        if shape.Length = 0 then 0
        else shape |> Array.fold (*) 1
    
    /// <summary>Calculate strides for C-layout (row-major).
    /// </summary>
    /// <param name="shape">The shape array.</param>
    /// <returns>Strides array for C-layout.</returns>
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
    /// <param name="shape">The shape array.</param>
    /// <returns>Strides array for Fortran-layout.</returns>
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
    /// <param name="shape">The shape to validate.</param>
    /// <returns>Result containing validated shape or error.</returns>
    let validate (shape: Shape) : FowlResult<Shape> =
        if shape |> Array.exists (fun d -> d <= 0) then
            Error.invalidShape "Shape dimensions must be positive"
        else
            Ok shape
    
    /// <summary>Check if two shapes are equal.
    /// </summary>
    /// <param name="shape1">First shape.</param>
    /// <param name="shape2">Second shape.</param>
    /// <returns>True if shapes are equal.</returns>
    let equals (shape1: Shape) (shape2: Shape) : bool =
        shape1.Length = shape2.Length &&
        (shape1, shape2) ||> Array.forall2 (=)

// ============================================================================
// Ndarray Operations
// ============================================================================

/// <summary>Core operations on N-dimensional arrays.
/// </summary>
module Ndarray =
    /// <summary>Local result computation expression instance.
    /// </summary>
    let private result = ResultBuilder()
    
    // ------------------------------------------------------------------------
    // Inspection
    // ------------------------------------------------------------------------
    
    /// <summary>Get the shape of an array.
    /// </summary>
    /// <param name="arr">The input array.</param>
    /// <returns>Shape array.</returns>
    let shape (arr: Ndarray<'K, 'T>) : Shape =
        match arr with
        | Dense d -> d.Shape
        | Sparse s -> s.Shape
    
    /// <summary>Get number of dimensions.
    /// </summary>
    /// <param name="arr">The input array.</param>
    /// <returns>Number of dimensions.</returns>
    let ndim (arr: Ndarray<'K, 'T>) : int =
        (shape arr).Length
    
    /// <summary>Get total number of elements.
    /// </summary>
    /// <param name="arr">The input array.</param>
    /// <returns>Total element count.</returns>
    let numel (arr: Ndarray<'K, 'T>) : int =
        shape arr |> Shape.numel
    
    // ------------------------------------------------------------------------
    // Creation
    // ------------------------------------------------------------------------
    
    /// <summary>Create empty array with given shape.
    /// </summary>
    /// <typeparam name="'K">Phantom type for element kind.</typeparam>
    /// <typeparam name="'T">Element type with default constructor.</typeparam>
    /// <param name="shape">Shape of the array.</param>
    /// <returns>Result containing empty array or error.</returns>
    let empty<'K, 'T when 'T: (new: unit -> 'T) and 'T: struct> 
            (shape: Shape) : FowlResult<Ndarray<'K, 'T>> =
        result {
            let! validShape = Shape.validate shape
            let n = Shape.numel validShape
            let data = Array.zeroCreate n
            return Dense {
                Data = data
                Shape = validShape
                Strides = Shape.stridesC validShape
                Offset = 0
                Layout = CLayout
            }
        }
    
    /// <summary>Create array filled with zeros.
    /// </summary>
    /// <param name="shape">Shape of the array.</param>
    /// <returns>Result containing zero array or error.</returns>
    let zeros (shape: Shape) : FowlResult<Float64Ndarray> =
        empty<Float64, float> shape
    
    /// <summary>Create array filled with ones.
    /// </summary>
    /// <param name="shape">Shape of the array.</param>
    /// <returns>Result containing ones array or error.</returns>
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
    
    /// <summary>Create array filled with a specific value.
    /// </summary>
    /// <typeparam name="'K">Phantom type for element kind.</typeparam>
    /// <typeparam name="'T">Element type.</typeparam>
    /// <param name="shape">Shape of the array.</param>
    /// <param name="value">Fill value.</param>
    /// <returns>Result containing filled array or error.</returns>
    let create<'K, 'T> (shape: Shape) (value: 'T) : FowlResult<Ndarray<'K, 'T>> =
        result {
            let! validShape = Shape.validate shape
            let n = Shape.numel validShape
            let data = Array.create n value
            return Dense {
                Data = data
                Shape = validShape
                Strides = Shape.stridesC validShape
                Offset = 0
                Layout = CLayout
            }
        }
    
    /// <summary>Create array from flat data.
    /// </summary>
    /// <typeparam name="'K">Phantom type for element kind.</typeparam>
    /// <typeparam name="'T">Element type.</typeparam>
    /// <param name="data">Flat data array.</param>
    /// <param name="shape">Target shape.</param>
    /// <returns>Result containing ndarray or error.</returns>
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
                Strides = Shape.stridesC validShape
                Offset = 0
                Layout = CLayout
            }
        }
    
    /// <summary>Convert to flat array (copy).
    /// </summary>
    /// <param name="arr">Input array.</param>
    /// <returns>Result containing flat array or error.</returns>
    let toArray (arr: Ndarray<'K, 'T>) : FowlResult<'T array> =
        match arr with
        | Dense d -> Ok (Array.copy d.Data)
        | Sparse _ -> Error.notImplemented "toArray for sparse arrays"
    
    // ------------------------------------------------------------------------
    // Element-wise Operations
    // ------------------------------------------------------------------------
    
    /// <summary>Map function over all elements.
    /// </summary>
    /// <typeparam name="'K">Phantom type for element kind.</typeparam>
    /// <typeparam name="'T">Input element type.</typeparam>
    /// <typeparam name="'U">Output element type.</typeparam>
    /// <param name="f">Mapping function.</param>
    /// <param name="arr">Input array.</param>
    /// <returns>Result containing mapped array or error.</returns>
    let map<'K, 'T, 'U> (f: 'T -> 'U) (arr: Ndarray<'K, 'T>) : FowlResult<Ndarray<'K, 'U>> =
        match arr with
        | Dense d ->
            let newData = Array.map f d.Data
            ofArray newData d.Shape
        | Sparse _ -> Error.notImplemented "map for sparse arrays"
    
    /// <summary>Element-wise addition.
    /// </summary>
    /// <param name="a">First array.</param>
    /// <param name="b">Second array.</param>
    /// <returns>Result containing sum or error.</returns>
    let add (a: Float64Ndarray) (b: Float64Ndarray) : FowlResult<Float64Ndarray> =
        result {
            let shapeA = shape a
            let shapeB = shape b
            if not (Shape.equals shapeA shapeB) then
                return! Error.dimensionMismatch 
                    (sprintf "Shape mismatch: %A vs %A" shapeA shapeB)
            let! dataA = toArray a
            let! dataB = toArray b
            let result = Array.map2 (+) dataA dataB
            return! ofArray result shapeA
        }
    
    /// <summary>Element-wise subtraction.
    /// </summary>
    /// <param name="a">First array.</param>
    /// <param name="b">Second array.</param>
    /// <returns>Result containing difference or error.</returns>
    let sub (a: Float64Ndarray) (b: Float64Ndarray) : FowlResult<Float64Ndarray> =
        result {
            let shapeA = shape a
            let shapeB = shape b
            if not (Shape.equals shapeA shapeB) then
                return! Error.dimensionMismatch 
                    (sprintf "Shape mismatch: %A vs %A" shapeA shapeB)
            let! dataA = toArray a
            let! dataB = toArray b
            let result = Array.map2 (-) dataA dataB
            return! ofArray result shapeA
        }
    
    /// <summary>Element-wise multiplication.
    /// </summary>
    /// <param name="a">First array.</param>
    /// <param name="b">Second array.</param>
    /// <returns>Result containing product or error.</returns>
    let mul (a: Float64Ndarray) (b: Float64Ndarray) : FowlResult<Float64Ndarray> =
        result {
            let shapeA = shape a
            let shapeB = shape b
            if not (Shape.equals shapeA shapeB) then
                return! Error.dimensionMismatch 
                    (sprintf "Shape mismatch: %A vs %A" shapeA shapeB)
            let! dataA = toArray a
            let! dataB = toArray b
            let result = Array.map2 (*) dataA dataB
            return! ofArray result shapeA
        }
    
    /// <summary>Element-wise division.
    /// </summary>
    /// <param name="a">First array.</param>
    /// <param name="b">Second array.</param>
    /// <returns>Result containing quotient or error.</returns>
    let div (a: Float64Ndarray) (b: Float64Ndarray) : FowlResult<Float64Ndarray> =
        result {
            let shapeA = shape a
            let shapeB = shape b
            if not (Shape.equals shapeA shapeB) then
                return! Error.dimensionMismatch 
                    (sprintf "Shape mismatch: %A vs %A" shapeA shapeB)
            let! dataA = toArray a
            let! dataB = toArray b
            let result = Array.map2 (/) dataA dataB
            return! ofArray result shapeA
        }
    
    // ------------------------------------------------------------------------
    // Reductions
    // ------------------------------------------------------------------------
    
    /// <summary>Sum all elements.
    /// </summary>
    /// <param name="arr">Input array.</param>
    /// <returns>Result containing sum or error.</returns>
    let sum (arr: Float64Ndarray) : FowlResult<float> =
        result {
            let! data = toArray arr
            return Array.sum data
        }
    
    /// <summary>Mean of all elements.
    /// </summary>
    /// <param name="arr">Input array.</param>
    /// <returns>Result containing mean or error.</returns>
    let mean (arr: Float64Ndarray) : FowlResult<float> =
        result {
            let! s = sum arr
            let n = numel arr
            if n = 0 then
                return! Error.invalidState "Cannot compute mean of empty array"
            return s / float n
        }
    
    /// <summary>Maximum element.
    /// </summary>
    /// <param name="arr">Input array.</param>
    /// <returns>Result containing max or error.</returns>
    let max (arr: Float64Ndarray) : FowlResult<float> =
        result {
            let! data = toArray arr
            if data.Length = 0 then
                return! Error.invalidState "Cannot find max of empty array"
            return Array.max data
        }
    
    /// <summary>Minimum element.
    /// </summary>
    /// <param name="arr">Input array.</param>
    /// <returns>Result containing min or error.</returns>
    let min (arr: Float64Ndarray) : FowlResult<float> =
        result {
            let! data = toArray arr
            if data.Length = 0 then
                return! Error.invalidState "Cannot find min of empty array"
            return Array.min data
        }

// ============================================================================
// Matrix Operations
// ============================================================================

/// <summary>Matrix operations for 2D arrays.
/// </summary>
module Matrix =
    /// <summary>Local result computation expression instance.
    /// </summary>
    let private result = ResultBuilder()
    
    /// <summary>Matrix multiplication.
    /// </summary>
    /// <param name="a">Left matrix (m x n).</param>
    /// <param name="b">Right matrix (n x p).</param>
    /// <returns>Result containing product matrix (m x p) or error.</returns>
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
    /// <param name="arr">Input matrix.</param>
    /// <returns>Result containing transposed matrix or error.</returns>
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
    
    /// <summary>Create identity matrix.
    /// </summary>
    /// <param name="n">Size of matrix (n x n).</param>
    /// <returns>Result containing identity matrix or error.</returns>
    let eye (n: int) : FowlResult<Float64Ndarray> =
        result {
            if n <= 0 then
                return! Error.invalidArgument "eye requires positive n"
            let data = Array.zeroCreate (n * n)
            for i = 0 to n - 1 do
                data.[i * n + i] <- 1.0
            return! Ndarray.ofArray data [|n; n|]
        }
    
    /// <summary>Create diagonal matrix from vector.
    /// </summary>
    /// <param name="diag">Diagonal elements.</param>
    /// <returns>Result containing diagonal matrix or error.</returns>
    let diag (diag: float array) : FowlResult<Float64Ndarray> =
        result {
            let n = diag.Length
            if n = 0 then
                return! Error.invalidArgument "diag requires non-empty array"
            let data = Array.zeroCreate (n * n)
            for i = 0 to n - 1 do
                data.[i * n + i] <- diag.[i]
            return! Ndarray.ofArray data [|n; n|]
        }

// ============================================================================
// Array Generation
// ============================================================================

/// <summary>Array generation functions.
/// </summary>
module Generate =
    /// <summary>Local result computation expression instance.
    /// </summary>
    let private result = ResultBuilder()
    
    /// <summary>Generate linearly spaced values.
    /// </summary>
    /// <param name="start">Start value.</param>
    /// <param name="stop">Stop value (inclusive).</param>
    /// <param name="num">Number of points.</param>
    /// <returns>Result containing 1D array or error.</returns>
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
    /// <param name="start">Start value.</param>
    /// <param name="stop">Stop value (exclusive).</param>
    /// <param name="step">Step size.</param>
    /// <returns>Result containing 1D array or error.</returns>
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
    /// <param name="shape">Shape of array.</param>
    /// <returns>Result containing random array or error.</returns>
    let random (shape: Shape) : FowlResult<Float64Ndarray> =
        result {
            let! validShape = Shape.validate shape
            let n = Shape.numel validShape
            let rng = System.Random()
            let data = Array.init n (fun _ -> rng.NextDouble())
            return! Ndarray.ofArray data validShape
        }
    
    /// <summary>Generate random values with seed.
    /// </summary>
    /// <param name="shape">Shape of array.</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>Result containing random array or error.</returns>
    let randomSeed (shape: Shape) (seed: int) : FowlResult<Float64Ndarray> =
        result {
            let! validShape = Shape.validate shape
            let n = Shape.numel validShape
            let rng = System.Random(seed)
            let data = Array.init n (fun _ -> rng.NextDouble())
            return! Ndarray.ofArray data validShape
        }
    
    /// <summary>Generate normally distributed random values.
    /// </summary>
    /// <param name="shape">Shape of array.</param>
    /// <param name="mu">Mean.</param>
    /// <param name="sigma">Standard deviation.</param>
    /// <returns>Result containing random array or error.</returns>
    let normal (shape: Shape) (mu: float) (sigma: float) : FowlResult<Float64Ndarray> =
        result {
            let! validShape = Shape.validate shape
            let n = Shape.numel validShape
            let rng = System.Random()
            let data = 
                Array.init n (fun _ ->
                    // Box-Muller transform
                    let u1 = 1.0 - rng.NextDouble()
                    let u2 = rng.NextDouble()
                    let radius = sqrt (-2.0 * log u1)
                    let theta = 2.0 * Math.PI * u2
                    mu + sigma * radius * cos theta
                )
            return! Ndarray.ofArray data validShape
        }
