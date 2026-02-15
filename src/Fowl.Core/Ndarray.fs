module Fowl.Core.Ndarray

open Fowl.Core.Types
open Fowl.Core.Shape

/// <summary>Create empty ndarray with given shape.</summary>
/// <typeparam name="K">The element type phantom (Float32, Float64, etc).</typeparam>
/// <param name="shape">The shape of the array.</param>
/// <returns>Result containing empty array or error.</returns>
let empty<'K> (shape: Shape) : FowlResult<Ndarray<'K, 'T>> =
    validateShape shape
    |> Result.map (fun validShape ->
        let n = numel validShape
        let data = Array.zeroCreate n
        Dense {
            Data = data
            Shape = validShape
            Strides = stridesC validShape
            Offset = 0
            Layout = CLayout
        })

/// <summary>Create ndarray filled with zeros.</summary>
/// <typeparam name="K">The element type phantom.</typeparam>
/// <param name="shape">The shape of the array.</param>
/// <returns>Result containing zero array or error.</returns>
/// <example>
/// <code>
/// let arr = Ndarray.zeros<Float64> [|3; 4|]
/// // Creates a 3x4 matrix of zeros
/// </code>
/// </example>
let zeros<'K> (shape: Shape) : FowlResult<Ndarray<'K, float>> =
    empty<'K> shape
    |> Result.map (fun arr ->
        match arr with
        | Dense d -> Dense { d with Data = Array.zeroCreate (Array.length d.Data) }
        | Sparse _ -> arr)  // Shouldn't happen from empty

/// <summary>Create ndarray filled with ones.</summary>
/// <typeparam name="K">The element type phantom.</typeparam>
/// <param name="shape">The shape of the array.</param>
/// <returns>Result containing ones array or error.</returns>
let ones<'K> (shape: Shape) : FowlResult<Ndarray<'K, float>> =
    validateShape shape
    |> Result.map (fun validShape ->
        let n = numel validShape
        let data = Array.create n 1.0
        Dense {
            Data = data
            Shape = validShape
            Strides = stridesC validShape
            Offset = 0
            Layout = CLayout
        })

/// <summary>Create ndarray filled with a specific value.</summary>
/// <typeparam name="K">The element type phantom.</typeparam>
/// <param name="shape">The shape of the array.</param>
/// <param name="value">The fill value.</param>
/// <returns>Result containing filled array or error.</returns>
let create<'K> (shape: Shape) (value: 'T) : FowlResult<Ndarray<'K, 'T>> =
    validateShape shape
    |> Result.map (fun validShape ->
        let n = numel validShape
        let data = Array.create n value
        Dense {
            Data = data
            Shape = validShape
            Strides = stridesC validShape
            Offset = 0
            Layout = CLayout
        })

/// Get shape of ndarray
let shape = function
    | Dense d -> d.Shape
    | Sparse s -> s.Shape

/// Get number of dimensions
let ndim arr =
    (shape arr).Length

/// Get total number of elements
let numel arr =
    shape arr |> Shape.numel

/// Calculate flat index from multi-dimensional indices
let flatIndex (strides: int array) (indices: int array) (offset: int) =
    let mutable idx = offset
    for i = 0 to indices.Length - 1 do
        idx <- idx + indices.[i] * strides.[i]
    idx

/// Get element at multi-dimensional indices
let get (arr: Ndarray<'K, 'T>) (indices: int array) : FowlResult<'T> =
    let arrShape = shape arr
    if indices.Length <> arrShape.Length then
        Error.indexOutOfRange (sprintf "Expected %d indices, got %d" arrShape.Length indices.Length)
    elif indices |> Array.mapi (fun i idx -> idx < 0 || idx >= arrShape.[i]) |> Array.exists id then
        Error.indexOutOfRange (sprintf "Indices %A out of range for shape %A" indices arrShape)
    else
        match arr with
        | Dense d ->
            let flatIdx = flatIndex d.Strides indices d.Offset
            Ok d.Data.[flatIdx]
        | Sparse _ -> Error.notImplemented "get not implemented for sparse arrays"

/// Set element at multi-dimensional indices
let set (arr: Ndarray<'K, 'T>) (indices: int array) (value: 'T) : FowlResult<unit> =
    let arrShape = shape arr
    if indices.Length <> arrShape.Length then
        Error.indexOutOfRange (sprintf "Expected %d indices, got %d" arrShape.Length indices.Length)
    elif indices |> Array.mapi (fun i idx -> idx < 0 || idx >= arrShape.[i]) |> Array.exists id then
        Error.indexOutOfRange (sprintf "Indices %A out of range for shape %A" indices arrShape)
    else
        match arr with
        | Dense d ->
            let flatIdx = flatIndex d.Strides indices d.Offset
            d.Data.[flatIdx] <- value
            Ok ()
        | Sparse _ -> Error.notImplemented "set not implemented for sparse arrays"

/// Map function over all elements
let map (f: 'T -> 'U) (arr: Ndarray<'K, 'T>) : Ndarray<'K, 'U> =
    match arr with
    | Dense d ->
        let newData = Array.map f d.Data
        Dense { d with Data = newData }
    | Sparse _ -> failwith "map not implemented for sparse arrays"

/// Fold over all elements
let fold (f: 'State -> 'T -> 'State) (init: 'State) (arr: Ndarray<'K, 'T>) : 'State =
    match arr with
    | Dense d -> Array.fold f init d.Data
    | Sparse _ -> failwith "fold not implemented for sparse arrays"

/// Reshape array (returns view if possible, copy otherwise)
let reshape (newShape: Shape) (arr: Ndarray<'K, 'T>) : FowlResult<Ndarray<'K, 'T>> =
    validateShape newShape
    |> Result.bind (fun validShape ->
        let currentNumel = numel arr
        let newNumel = Shape.numel validShape
        if currentNumel <> newNumel then
            Error.invalidShape (sprintf "Cannot reshape: current has %d elements, new shape has %d" currentNumel newNumel)
        else
            match arr with
            | Dense d ->
                Ok (Dense { d with Shape = validShape; Strides = stridesC validShape })
            | Sparse _ -> Error.notImplemented "reshape not implemented for sparse arrays")

/// Convert to flat array (copy)
let toArray (arr: Ndarray<'K, 'T>) : 'T array =
    match arr with
    | Dense d -> Array.copy d.Data
    | Sparse _ -> failwith "toArray not implemented for sparse arrays"

/// Create from flat array
let ofArray (data: 'T array) (shape: Shape) : FowlResult<Ndarray<'K, 'T>> =
    validateShape shape
    |> Result.bind (fun validShape ->
        let expectedSize = Shape.numel validShape
        if data.Length <> expectedSize then
            Error.invalidShape (sprintf "Data length %d does not match shape %A (%d elements)" data.Length validShape expectedSize)
        else
            Ok (Dense {
                Data = Array.copy data
                Shape = validShape
                Strides = stridesC validShape
                Offset = 0
                Layout = CLayout
            }))

/// Generate linearly spaced values
let linspace (start: float) (stop: float) (num: int) : FowlResult<Ndarray<Float64, float>> =
    if num < 2 then
        Error.invalidArgument "linspace requires num >= 2"
    else
        let step = (stop - start) / float (num - 1)
        let data = Array.init num (fun i -> start + float i * step)
        ofArray data [|num|]

/// Generate values with given step
let arange (start: float) (stop: float) (step: float) : FowlResult<Ndarray<Float64, float>> =
    if step = 0.0 then
        Error.invalidArgument "arange: step cannot be zero"
    elif (step > 0.0 && start >= stop) || (step < 0.0 && start <= stop) then
        Error.invalidArgument "arange: step direction must match range direction"
    else
        let n = int ((stop - start) / step)
        let data = Array.init n (fun i -> start + float i * step)
        ofArray data [|n|]

/// Element-wise addition with Result type
let add (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
    match a, b with
    | Dense da, Dense db ->
        if da.Shape <> db.Shape then
            Error.dimensionMismatch (sprintf "Shape mismatch in add: %A vs %A" da.Shape db.Shape)
        else
            let newData = Array.map2 (+) da.Data db.Data
            Ok (Dense { da with Data = newData })
    | _ -> Error.notImplemented "add not implemented for sparse arrays"

/// Element-wise subtraction with Result type
let sub (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
    match a, b with
    | Dense da, Dense db ->
        if da.Shape <> db.Shape then
            Error.dimensionMismatch (sprintf "Shape mismatch in sub: %A vs %A" da.Shape db.Shape)
        else
            let newData = Array.map2 (-) da.Data db.Data
            Ok (Dense { da with Data = newData })
    | _ -> Error.notImplemented "sub not implemented for sparse arrays"

/// Element-wise multiplication with Result type
let mul (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
    match a, b with
    | Dense da, Dense db ->
        if da.Shape <> db.Shape then
            Error.dimensionMismatch (sprintf "Shape mismatch in mul: %A vs %A" da.Shape db.Shape)
        else
            let newData = Array.map2 (*) da.Data db.Data
            Ok (Dense { da with Data = newData })
    | _ -> Error.notImplemented "mul not implemented for sparse arrays"

/// Element-wise division with Result type
let div (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
    match a, b with
    | Dense da, Dense db ->
        if da.Shape <> db.Shape then
            Error.dimensionMismatch (sprintf "Shape mismatch in div: %A vs %A" da.Shape db.Shape)
        elif db.Data |> Array.exists (fun x -> x = 0.0) then
            Error.invalidArgument "Division by zero detected"
        else
            let newData = Array.map2 (/) da.Data db.Data
            Ok (Dense { da with Data = newData })
    | _ -> Error.notImplemented "div not implemented for sparse arrays"

/// <summary>Create 2D Ndarray from 2D array.</summary>
/// <param name="data">2D array data.</param>
/// <returns>2D Ndarray.</returns>
let ofArray2D (data: 'T[,]) : FowlResult<Ndarray<'K, 'T>> =
    let rows = data.GetLength(0)
    let cols = data.GetLength(1)
    
    // Flatten 2D to 1D in row-major order
    let flatData = Array.init (rows * cols) (fun i ->
        let r = i / cols
        let c = i % cols
        data.[r, c])
    
    ofArray flatData [|rows; cols|]

/// <summary>Convert Ndarray to 2D array.</summary>
/// <param name="arr">Ndarray to convert.</param>
/// <returns>2D array.</returns>
let toArray2D (arr: Ndarray<'K, 'T>) : FowlResult<'T[,]> =
    match arr with
    | Dense d ->
        if d.Shape.Length <> 2 then
            Error.invalidShape "toArray2D requires 2D array"
        else
            let rows = d.Shape.[0]
            let cols = d.Shape.[1]
            let result = Array2D.zeroCreate rows cols
            
            for r = 0 to rows - 1 do
                for c = 0 to cols - 1 do
                    let idx = flatIndex d.Strides [|r; c|] d.Offset
                    result.[r, c] <- d.Data.[idx]
            
            Ok result
    | Sparse _ -> Error.notImplemented "toArray2D not implemented for sparse arrays"
