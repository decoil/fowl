/// <summary>Fowl NdarrayView - Zero-Copy Array Views</summary>
/// <remarks>
/// Provides view types that reference existing array data without copying.
/// 
/// Unlike Ndarray.slice which creates a copy, NdarrayView creates a view
/// into the original data, enabling:
/// - Zero-allocation slicing
/// - Multiple views of same data
/// - Memory-efficient matrix operations
/// 
/// Example:
/// <code>
/// let arr = Ndarray.zeros<Float64> [|10; 10|]
/// let row = NdarrayView.row arr 5  // View of row 5, no copy
/// let sub = NdarrayView.slice arr [|0..4; 0..4|]  // View of top-left 5x5
/// </code>
/// </remarks>
module Fowl.Memory.NdarrayView

open System
open Fowl
open Fowl.Core.Types

// ============================================================================
// View Types
// ============================================================================

/// <summary>A view into existing Ndarray data without copying.</summary>
/// <typeparam name="K">Element type phantom (Float32, Float64).</typeparam>
/// <typeparam name="T">Element type.</typeparam>
/// <remarks>
/// Holds a reference to source array plus offset and strides.
/// Changes to the view affect the source array.
/// </remarks>
type NdarrayView<'K, 'T> = {
    /// <summary>Source array being viewed.</summary>    Source: Ndarray<'K, 'T>
    /// <summary>Offset into source data.</summary>    Offset: int
    /// <summary>Shape of the view.</summary>    Shape: Shape
    /// <summary>Strides for indexing (may differ from source).</summary>    Strides: int[]
}

/// <summary>Get the shape of a view.</summary>let shape (view: NdarrayView<'K, 'T>) : Shape =
    view.Shape

/// <summary>Get the number of dimensions.</summary>let ndim (view: NdarrayView<'K, 'T>) : int =
    view.Shape.Length

/// <summary>Get total number of elements.</summary>let numel (view: NdarrayView<'K, 'T>) : int =
    view.Shape |> Array.fold (*) 1

/// <summary>Check if view is contiguous (can be converted to Span).</summary>let isContiguous (view: NdarrayView<'K, 'T>) : bool =
    match view.Source with
    | Dense d ->
        // Check if strides match C-contiguous layout
        let expectedStrides = Shape.stridesC view.Shape
        view.Strides = expectedStrides
    | Sparse _ -> false

// ============================================================================
// Creating Views
// ============================================================================

/// <summary>Create a view of entire Ndarray.</summary>/// <param name="arr">Source array.</param>/// <returns>View of entire array.</returns>let ofNdarray (arr: Ndarray<'K, 'T>) : NdarrayView<'K, 'T> =
    match arr with
    | Dense d ->
        { Source = arr
          Offset = d.Offset
          Shape = d.Shape
          Strides = d.Strides }
    | Sparse _ -> failwith "Views not supported for sparse arrays"

/// <summary>Create a view of a 1D slice (no copy).</summary>/// <param name="arr">Source array.</param>/// <param name="start">Start index.</param>
/// <param name="length">Length of slice.</param>/// <returns>View of slice.</returns>
/// <example>
/// <code>
/// let arr = Ndarray.zeros<Float64> [|100|]
/// let view = NdarrayView.slice1D arr 10 20  // View of elements 10-29
/// </code>
/// </example>
let slice1D (arr: Ndarray<'K, 'T>) (start: int) (length: int) : NdarrayView<'K, 'T> =
    match arr with
    | Dense d when d.Shape.Length = 1 ->
        if start < 0 || start + length > d.Shape.[0] then
            invalidArg "start/length" "Slice out of bounds"
        
        let newOffset = d.Offset + start * d.Strides.[0]
        { Source = arr
          Offset = newOffset
          Shape = [|length|]
          Strides = d.Strides }
    | Dense _ -> failwith "slice1D requires 1D array"
    | Sparse _ -> failwith "Views not supported for sparse arrays"

/// <summary>Create a view of a matrix row (no copy).</summary>/// <param name="arr">Source 2D array.</param>/// <param name="row">Row index.</param>
/// <returns>View of row.</returns>/// <example>
/// <code>
/// let matrix = Ndarray.zeros<Float64> [|10; 20|]  // 10x20 matrix
/// let row = NdarrayView.row matrix 5  // View of row 5 (20 elements)
/// </code>
/// </example>
let row (arr: Ndarray<'K, 'T>) (row: int) : NdarrayView<'K, 'T> =
    match arr with
    | Dense d when d.Shape.Length = 2 ->
        if row < 0 || row >= d.Shape.[0] then
            invalidArg "row" "Row index out of bounds"
        
        let newOffset = d.Offset + row * d.Strides.[0]
        { Source = arr
          Offset = newOffset
          Shape = [|d.Shape.[1]|]  // Row has shape [|cols|]
          Strides = [|d.Strides.[1]|] }
    | Dense _ -> failwith "row requires 2D array"
    | Sparse _ -> failwith "Views not supported for sparse arrays"

/// <summary>Create a view of a matrix column (may be strided).</summary>/// <param name="arr">Source 2D array.</param>/// <param name="col">Column index.</param>
/// <returns>View of column.</returns>/// <remarks>
/// Column views are strided (not contiguous) for row-major arrays.
/// </remarks>
/// <example>
/// <code>
/// let matrix = Ndarray.zeros<Float64> [|10; 20|]  // 10x20 matrix
/// let col = NdarrayView.column matrix 3  // View of column 3 (10 elements, strided)
/// </code>
/// </example>
let column (arr: Ndarray<'K, 'T>) (col: int) : NdarrayView<'K, 'T> =
    match arr with
    | Dense d when d.Shape.Length = 2 ->
        if col < 0 || col >= d.Shape.[1] then
            invalidArg "col" "Column index out of bounds"
        
        let newOffset = d.Offset + col * d.Strides.[1]
        { Source = arr
          Offset = newOffset
          Shape = [|d.Shape.[0]|]  // Column has shape [|rows|]
          Strides = [|d.Strides.[0]|] }  // Stride is row stride
    | Dense _ -> failwith "column requires 2D array"
    | Sparse _ -> failwith "Views not supported for sparse arrays"

/// <summary>Create a view of a sub-matrix (no copy).</summary>/// <param name="arr">Source 2D array.</param>/// <param name="rowStart">Starting row.</param>
/// <param name="rowCount">Number of rows.</param>/// <param name="colStart">Starting column.</param>
/// <param name="colCount">Number of columns.</param>/// <returns>View of sub-matrix.</returns>
/// <example>
/// <code>
/// let matrix = Ndarray.zeros<Float64> [|100; 100|]
/// let sub = NdarrayView.subMatrix matrix 10 20 30 40  // 20x40 submatrix at (10,30)
/// </code>
/// </example>
let subMatrix (arr: Ndarray<'K, 'T>) (rowStart: int) (rowCount: int) (colStart: int) (colCount: int) : NdarrayView<'K, 'T> =
    match arr with
    | Dense d when d.Shape.Length = 2 ->
        if rowStart < 0 || rowStart + rowCount > d.Shape.[0] then
            invalidArg "rowStart/rowCount" "Row slice out of bounds"
        if colStart < 0 || colStart + colCount > d.Shape.[1] then
            invalidArg "colStart/colCount" "Column slice out of bounds"
        
        let newOffset = d.Offset + rowStart * d.Strides.[0] + colStart * d.Strides.[1]
        { Source = arr
          Offset = newOffset
          Shape = [|rowCount; colCount|]
          Strides = d.Strides }  // Same strides as source
    | Dense _ -> failwith "subMatrix requires 2D array"
    | Sparse _ -> failwith "Views not supported for sparse arrays"

// ============================================================================
// Accessing View Data
// ============================================================================

/// <summary>Get element at indices in view.</summary>/// <param name="view">Array view.</param>/// <param name="indices">Indices in view coordinates.</param>
/// <returns>Element value.</returns>let get (view: NdarrayView<'K, 'T>) (indices: int[]) : FowlResult<'T> =
    if indices.Length <> view.Shape.Length then
        Error.indexOutOfRange (sprintf "Expected %d indices, got %d" view.Shape.Length indices.Length)
    elif indices |> Array.mapi (fun i idx -> idx < 0 || idx >= view.Shape.[i]) |> Array.exists id then
        Error.indexOutOfRange "Indices out of range"
    else
        // Calculate source index from view coordinates
        let mutable srcIdx = view.Offset
        for i = 0 to indices.Length - 1 do
            srcIdx <- srcIdx + indices.[i] * view.Strides.[i]
        
        match view.Source with
        | Dense d -> Ok d.Data.[srcIdx]
        | Sparse _ -> Error.notImplemented "Views not supported for sparse"

/// <summary>Set element at indices in view.</summary>/// <param name="view">Array view.</param>/// <param name="indices">Indices in view coordinates.</param>/// <param name="value">Value to set.</param>
/// <returns>Unit result.</returns>let set (view: NdarrayView<'K, 'T>) (indices: int[]) (value: 'T) : FowlResult<unit> =
    if indices.Length <> view.Shape.Length then
        Error.indexOutOfRange (sprintf "Expected %d indices, got %d" view.Shape.Length indices.Length)
    elif indices |> Array.mapi (fun i idx -> idx < 0 || idx >= view.Shape.[i]) |> Array.exists id then
        Error.indexOutOfRange "Indices out of range"
    else
        let mutable srcIdx = view.Offset
        for i = 0 to indices.Length - 1 do
            srcIdx <- srcIdx + indices.[i] * view.Strides.[i]
        
        match view.Source with
        | Dense d ->
            d.Data.[srcIdx] <- value
            Ok ()
        | Sparse _ -> Error.notImplemented "Views not supported for sparse"

/// <summary>Convert contiguous view to Span.</summary>/// <param name="view">Array view.</param>
/// <returns>Span of data if contiguous, None otherwise.</returns>
let toSpan (view: NdarrayView<'K, 'T>) : Span<'T> option =
    if isContiguous view then
        match view.Source with
        | Dense d ->
            Some (Span(d.Data, view.Offset, numel view))
        | Sparse _ -> None
    else
        None

/// <summary>Convert view to new Ndarray (copy).</summary>
/// <param name="view">Array view.</param>
/// <returns>New copied array.</returns>
let toNdarray (view: NdarrayView<'K, 'T>) : FowlResult<Ndarray<'K, 'T>> =
    match view.Source with
    | Dense d ->
        let n = numel view
        let data = Array.zeroCreate n
        
        // Copy data from view
        // For now, simple copy assuming contiguous
        // TODO: Handle strided views
        if isContiguous view then
            Span(d.Data, view.Offset, n).CopyTo(Span(data))
        else
            // Strided copy - iterate and copy element by element
            failwith "Strided toNdarray not yet implemented"
        
        Ndarray.ofArray data view.Shape
    | Sparse _ ->
        Error.notImplemented "Views not supported for sparse"

// ============================================================================
// View Operations
// ============================================================================

/// <summary>Create a new view with different shape (must have same numel).</summary>
/// <param name="view">Source view.</param>
/// <param name="newShape">New shape.</param>
/// <returns>Reshaped view.</returns>
let reshape (view: NdarrayView<'K, 'T>) (newShape: Shape) : NdarrayView<'K, 'T> =
    let newNumel = newShape |> Array.fold (*) 1
    if newNumel <> numel view then
        invalidArg "newShape" "New shape must have same number of elements"
    
    if not (isContiguous view) then
        invalidArg "view" "Cannot reshape non-contiguous view"
    
    { view with
        Shape = newShape
        Strides = Shape.stridesC newShape }

/// <summary>Transpose a 2D view (swaps strides).</summary>
/// <param name="view">2D view to transpose.</param>
/// <returns>Transposed view.</returns>
let transpose (view: NdarrayView<'K, 'T>) : NdarrayView<'K, 'T> =
    if view.Shape.Length <> 2 then
        invalidArg "view" "transpose requires 2D view"
    
    { view with
        Shape = [|view.Shape.[1]; view.Shape.[0]|]
        Strides = [|view.Strides.[1]; view.Strides.[0]|] }

// ============================================================================
// Iteration
// ============================================================================

/// <summary>Apply action to each element in view.</summary>
/// <param name="action">Action to apply.</param>
/// <param name="view">View to iterate.</param>
let iter (action: 'T -> unit) (view: NdarrayView<'K, 'T>) : unit =
    match view.Source with
    | Dense d ->
        // Simple iteration for 1D contiguous views
        if view.Shape.Length = 1 && isContiguous view then
            let span = Span(d.Data, view.Offset, view.Shape.[0])
            for i = 0 to span.Length - 1 do
                action span.[i]
        else
            failwith "iter not implemented for strided/multi-dim views"
    | Sparse _ ->
        failwith "Views not supported for sparse"

/// <summary>Map function over view and return new array.</summary>
/// <param name="mapping">Mapping function.</param>
/// <param name="view">View to map.</param>
/// <returns>New array with mapped values.</returns>
let map (mapping: 'T -> 'U) (view: NdarrayView<'K, 'T>) : 'U[] =
    match view.Source with
    | Dense d when isContiguous view && view.Shape.Length = 1 ->
        let span = ReadOnlySpan(d.Data, view.Offset, view.Shape.[0])
        Array.init span.Length (fun i -> mapping span.[i])
    | _ ->
        failwith "map only implemented for 1D contiguous views"

/// <summary>Fold over view elements.</summary>
/// <param name="folder">Folder function.</param>
/// <param name="state">Initial state.</param>
/// <param name="view">View to fold.</param>
/// <returns>Final state.</returns>
let fold (folder: 'State -> 'T -> 'State) (state: 'State) (view: NdarrayView<'K, 'T>) : 'State =
    match view.Source with
    | Dense d when isContiguous view && view.Shape.Length = 1 ->
        let span = ReadOnlySpan(d.Data, view.Offset, view.Shape.[0])
        let mutable acc = state
        for i = 0 to span.Length - 1 do
            acc <- folder acc span.[i]
        acc
    | _ ->
        failwith "fold only implemented for 1D contiguous views"
