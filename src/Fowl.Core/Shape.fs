module Fowl.Core.Shape

open Fowl.Core.Types

/// Calculate total number of elements from shape
let numel (shape: Shape) =
    if shape.Length = 0 then 0
    else shape |> Array.fold (*) 1

/// Validate that shape is valid (all dimensions positive)
let validateShape (shape: Shape) : FowlResult<Shape> =
    if shape |> Array.exists (fun d -> d < 0) then
        Error.invalidShape "Shape cannot contain negative dimensions"
    elif shape |> Array.exists (fun d -> d = 0) then
        Error.invalidShape "Shape cannot contain zero dimensions (use empty arrays instead)"
    else
        Ok shape

/// Calculate strides from shape (C-layout / row-major)
let stridesC (shape: Shape) : int array =
    let n = shape.Length
    let s = Array.zeroCreate n
    let mutable stride = 1
    for i = n - 1 downto 0 do
        s.[i] <- stride
        stride <- stride * shape.[i]
    s

/// Calculate strides from shape (Fortran-layout / column-major)
let stridesF (shape: Shape) : int array =
    let n = shape.Length
    let s = Array.zeroCreate n
    let mutable stride = 1
    for i = 0 to n - 1 do
        s.[i] <- stride
        stride <- stride * shape.[i]
    s

/// Check if two shapes are compatible for broadcasting
let broadcastable (shape1: Shape) (shape2: Shape) : bool =
    let n1 = shape1.Length
    let n2 = shape2.Length
    let maxN = max n1 n2
    
    let rec check i =
        if i >= maxN then true
        else
            let dim1 = if i < n1 then shape1.[n1 - 1 - i] else 1
            let dim2 = if i < n2 then shape2.[n2 - 1 - i] else 1
            if dim1 = dim2 || dim1 = 1 || dim2 = 1 then
                check (i + 1)
            else
                false
    check 0

/// Calculate broadcasted shape from two shapes
let broadcastShape (shape1: Shape) (shape2: Shape) : FowlResult<Shape> =
    if not (broadcastable shape1 shape2) then
        Error.dimensionMismatch (sprintf "Shapes %A and %A are not broadcastable" shape1 shape2)
    else
        let n1 = shape1.Length
        let n2 = shape2.Length
        let maxN = max n1 n2
        
        let result = Array.zeroCreate maxN
        for i = 0 to maxN - 1 do
            let dim1 = if i < n1 then shape1.[n1 - 1 - i] else 1
            let dim2 = if i < n2 then shape2.[n2 - 1 - i] else 1
            result.[maxN - 1 - i] <- max dim1 dim2
        Ok result
