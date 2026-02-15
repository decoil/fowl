module Fowl.Core.Types

open System

/// <summary>Phantom type for 32-bit floating point numbers.</summary>
type Float32 = class end

/// <summary>Phantom type for 64-bit floating point numbers (double precision).</summary>
type Float64 = class end

/// <summary>Phantom type for 32-bit complex numbers.</summary>
type Complex32 = class end

/// <summary>Phantom type for 64-bit complex numbers (double precision).</summary>
type Complex64 = class end

/// <summary>Shape is an array of dimension sizes. For example, a 3x4 matrix has shape [|3; 4|].</summary>
type Shape = int array

/// Layout for memory ordering
type Layout =
    | CLayout       // Row-major (C-style)
    | FortranLayout // Column-major (Fortran-style)

/// Dense n-dimensional array
type DenseArray<'T> = {
    Data: 'T array
    Shape: Shape
    Strides: int array
    Offset: int
    Layout: Layout
}

/// Sparse array formats
type SparseFormat =
    | CSR // Compressed Sparse Row
    | CSC // Compressed Sparse Column
    | COO // Coordinate format

/// Sparse n-dimensional array
type SparseArray<'T> = {
    Indices: int array array
    Values: 'T array
    Shape: Shape
    Format: SparseFormat
}

/// Ndarray discriminated union
type Ndarray<'K, 'T> =
    | Dense of DenseArray<'T>
    | Sparse of SparseArray<'T>

/// Type aliases for common ndarray types
type Ndarray<'T> = Ndarray<Float64, 'T>
type Float32Ndarray = Ndarray<Float32, float32>
type Float64Ndarray = Ndarray<Float64, float>
type Complex32Ndarray = Ndarray<Complex32, System.Numerics.Complex>
type Complex64Ndarray = Ndarray<Complex64, System.Numerics.Complex>

/// Slice specification for indexing
type SliceSpec =
    | All                    // Take all elements in this dimension
    | Index of int           // Single index
    | Range of (int option * int option * int option)  // (start, stop, step)
    | IndexArray of int array  // Array of indices

/// <summary>Error types for Fowl operations.</summary>
/// <remarks>All Fowl operations return Result types with these error variants for safe error handling.</remarks>
type FowlError =
    /// <summary>Invalid shape for operation (e.g., mismatched dimensions).</summary>
    | InvalidShape of string
    /// <summary>Dimension mismatch between arrays.</summary>
    | DimensionMismatch of string
    /// <summary>Index out of valid range.</summary>
    | IndexOutOfRange of string
    /// <summary>Invalid argument value.</summary>
    | InvalidArgument of string
    /// <summary>Native library (OpenBLAS) not available or failed.</summary>
    | NativeLibraryError of string
    /// <summary>Feature not yet implemented.</summary>
    | NotImplemented of string
    /// <summary>Invalid internal state.</summary>
    | InvalidState of string

/// <summary>Result type alias for Fowl operations.</summary>
/// <typeparam name="T">The success type.</typeparam>
type FowlResult<'T> = Result<'T, FowlError>

/// <summary>Helper module for error handling.</summary>
module Error =
    /// <summary>Create an InvalidShape error.</summary>
    /// <param name="msg">Error message.</param>
    let inline invalidShape msg = Error (InvalidShape msg)
    
    /// <summary>Create a DimensionMismatch error.</summary>
    /// <param name="msg">Error message.</param>
    let inline dimensionMismatch msg = Error (DimensionMismatch msg)
    
    /// <summary>Create an IndexOutOfRange error.</summary>
    /// <param name="msg">Error message.</param>
    let inline indexOutOfRange msg = Error (IndexOutOfRange msg)
    
    /// <summary>Create an InvalidArgument error.</summary>
    /// <param name="msg">Error message.</param>
    let inline invalidArgument msg = Error (InvalidArgument msg)
    
    /// <summary>Create a NativeLibraryError.</summary>
    /// <param name="msg">Error message.</param>
    let inline nativeLibraryError msg = Error (NativeLibraryError msg)
    
    /// <summary>Create a NotImplemented error.</summary>
    /// <param name="msg">Error message.</param>
    let inline notImplemented msg = Error (NotImplemented msg)
    
    /// <summary>Create an InvalidState error.</summary>
    /// <param name="msg">Error message.</param>
    let inline invalidState msg = Error (InvalidState msg)
    
    /// <summary>Convert exception to FowlError.</summary>
    /// <param name="ex">The exception to convert.</param>
    let ofException (ex: exn) : FowlError =
        InvalidState (ex.Message)
    
    /// <summary>Helper to wrap functions in try-catch.</summary>
    /// <param name="f">Function to wrap.</param>
    let tryCatch (f: unit -> 'T) : FowlResult<'T> =
        try
            Ok (f())
        with
        | ex -> Error (ofException ex)

/// <summary>Computation expression builder for Result type.</summary>
type ResultBuilder() =
    member _.Bind(m, f) = Result.bind f m
    member _.Return(x) = Ok x
    member _.ReturnFrom(m) = m
    member _.Zero() = Ok ()
    member _.Combine(a, b) = Result.bind (fun _ -> b) a
    member _.Delay(f) = f()

/// <summary>Global result computation expression instance.</summary>
let result = ResultBuilder()
