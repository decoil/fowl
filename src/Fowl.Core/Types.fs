module Fowl.Core.Types

open System

/// Phantom types for ndarray kind
type Float32 = class end
type Float64 = class end
type Complex32 = class end
type Complex64 = class end

/// Shape is an array of dimension sizes
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

/// Slice specification for indexing
type SliceSpec =
    | All                    // Take all elements in this dimension
    | Index of int           // Single index
    | Range of (int option * int option * int option)  // (start, stop, step)
    | IndexArray of int array  // Array of indices

/// Error types for Fowl operations
type FowlError =
    | InvalidShape of string
    | DimensionMismatch of string
    | IndexOutOfRange of string
    | InvalidArgument of string
    | NativeLibraryError of string
    | NotImplemented of string
    | InvalidState of string

/// Result type alias for Fowl operations
type FowlResult<'T> = Result<'T, FowlError>

/// Helper module for error handling
module Error =
    let inline invalidShape msg = Error (InvalidShape msg)
    let inline dimensionMismatch msg = Error (DimensionMismatch msg)
    let inline indexOutOfRange msg = Error (IndexOutOfRange msg)
    let inline invalidArgument msg = Error (InvalidArgument msg)
    let inline nativeLibraryError msg = Error (NativeLibraryError msg)
    let inline notImplemented msg = Error (NotImplemented msg)
    let inline invalidState msg = Error (InvalidState msg)
    
    /// Convert exception to FowlError
    let ofException (ex: exn) : FowlError =
        InvalidState (ex.Message)
    
    /// Helper to wrap functions in try-catch
    let tryCatch (f: unit -> 'T) : FowlResult<'T> =
        try
            Ok (f())
        with
        | ex -> Error (ofException ex)
