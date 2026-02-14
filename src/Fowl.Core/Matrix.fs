module Fowl.Core.Matrix

open Fowl
open Fowl.Core.Slice

/// Active pattern for matrix shape checking
let (|Matrix|_|) (arr: Ndarray<'K, 'T>) =
    let shape = Ndarray.shape arr
    if shape.Length = 2 then
        Some (shape.[0], shape.[1])
    else
        None

/// Active pattern for vector shape checking
let (|Vector|_|) (arr: Ndarray<'K, 'T>) =
    let shape = Ndarray.shape arr
    if shape.Length = 1 then
        Some shape.[0]
    elif shape.Length = 2 && shape.[1] = 1 then
        Some shape.[0]
    elif shape.Length = 2 && shape.[0] = 1 then
        Some shape.[1]
    else
        None

/// Transpose a matrix (swap rows and columns)
let transpose (arr: Ndarray<'K, 'T>) : FowlResult<Ndarray<'K, 'T>> =
    match arr with
    | Dense d ->
        let nDims = d.Shape.Length
        if nDims < 2 then
            Error.invalidArgument "transpose requires at least 2D array"
        else
            Ok (
                // For now, use slice with index swap for transpose
                // This is a view-based approach that should work
                let newShape = Array.copy d.Shape
                newShape.[nDims - 2] <- d.Shape.[nDims - 1]
                newShape.[nDims - 1] <- d.Shape.[nDims - 2]
                
                Dense {
                    Data = d.Data
                    Shape = newShape
                    Strides = Array.init nDims (fun i ->
                        if i = nDims - 2 then d.Strides.[nDims - 1]
                        elif i = nDims - 1 then d.Strides.[nDims - 2]
                        else d.Strides.[i])
                    Offset = d.Offset
                    Layout = d.Layout
                }
            )
    | Sparse _ -> Error.notImplemented "transpose not implemented for sparse arrays"

/// Matrix multiplication (2D arrays only)
let matmul (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
    match a, b with
    | Dense da, Dense db ->
        match da.Shape, db.Shape with
        | [|m; n|], [|p; q|] when n = p ->
            let result = Array.zeroCreate (m * q)

            // Naive matrix multiplication
            // TODO: Replace with BLAS call for performance
            for i = 0 to m - 1 do
                for j = 0 to q - 1 do
                    let mutable sum = 0.0
                    for k = 0 to n - 1 do
                        let aIdx = i * da.Strides.[0] + k * da.Strides.[1] + da.Offset
                        let bIdx = k * db.Strides.[0] + j * db.Strides.[1] + db.Offset
                        sum <- sum + da.Data.[aIdx] * db.Data.[bIdx]
                    result.[i * q + j] <- sum

            Ndarray.ofArray result [|m; q|]
        | _ -> Error.dimensionMismatch (sprintf "Incompatible shapes for matrix multiplication: %A and %A" da.Shape db.Shape)
    | _ -> Error.notImplemented "matmul not implemented for sparse arrays"

/// Dot product of two 1D arrays
let dot (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : FowlResult<float> =
    match a, b with
    | Dense da, Dense db ->
        if da.Shape.Length <> 1 || db.Shape.Length <> 1 then
            Error.invalidArgument "dot requires 1D arrays"
        elif da.Shape.[0] <> db.Shape.[0] then
            Error.dimensionMismatch (sprintf "dot requires arrays of same length: %d vs %d" da.Shape.[0] db.Shape.[0])
        else
            let mutable sum = 0.0
            for i = 0 to da.Shape.[0] - 1 do
                sum <- sum + da.Data.[da.Offset + i * da.Strides.[0]] * db.Data.[db.Offset + i * db.Strides.[0]]
            Ok sum
    | _ -> Error.notImplemented "dot not implemented for sparse arrays"

/// Outer product of two 1D arrays
let outer (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
    match a, b with
    | Dense da, Dense db ->
        if da.Shape.Length <> 1 || db.Shape.Length <> 1 then
            Error.invalidArgument "outer requires 1D arrays"
        else
            let m = da.Shape.[0]
            let n = db.Shape.[0]
            let result = Array.zeroCreate (m * n)

            for i = 0 to m - 1 do
                for j = 0 to n - 1 do
                    result.[i * n + j] <-
                        da.Data.[da.Offset + i * da.Strides.[0]] *
                        db.Data.[db.Offset + j * db.Strides.[0]]

            Ndarray.ofArray result [|m; n|]
    | _ -> Error.notImplemented "outer not implemented for sparse arrays"

/// Sum all elements, or along a specific axis
let sum (?axis: int) (arr: Ndarray<'K, float>) : Ndarray<'K, float> =
    match arr with
    | Dense d ->
        match axis with
        | None ->
            // Sum all elements
            let total = Array.sum d.Data
            Ndarray.ofArray [|total|] [||]
        | Some ax ->
            if ax < 0 || ax >= d.Shape.Length then
                failwithf "Axis %d out of range" ax

            // Calculate output shape (remove axis dimension)
            let outShape =
                d.Shape
                |> Array.mapi (fun i dim -> if i = ax then -1 else dim)
                |> Array.filter ((<>) -1)

            let outN = Shape.numel outShape
            let outData = Array.zeroCreate outN

            // Iterate over output indices
            let rec iterate (outIdx: int array) (dim: int) =
                if dim = outShape.Length then
                    // Sum along axis
                    let mutable sum = 0.0
                    for i = 0 to d.Shape.[ax] - 1 do
                        // Construct full index
                        let fullIdx = Array.zeroCreate d.Shape.Length
                        let mutable outDim = 0
                        for j = 0 to d.Shape.Length - 1 do
                            if j = ax then
                                fullIdx.[j] <- i
                            else
                                fullIdx.[j] <- outIdx.[outDim]
                                outDim <- outDim + 1

                        let flatIdx = Ndarray.flatIndex d.Strides fullIdx d.Offset
                        sum <- sum + d.Data.[flatIdx]

                    // Store result
                    let outFlat =
                        if outShape.Length = 0 then 0
                        else Ndarray.flatIndex (Shape.stridesC outShape) outIdx 0
                    outData.[outFlat] <- sum
                else
                    for i = 0 to outShape.[dim] - 1 do
                        outIdx.[dim] <- i
                        iterate outIdx (dim + 1)

            iterate (Array.zeroCreate outShape.Length) 0
            Ndarray.ofArray outData outShape
    | Sparse _ -> failwith "sum not implemented for sparse arrays"

/// Mean of all elements, or along a specific axis
let mean (?axis: int) (arr: Ndarray<'K, float>) : Ndarray<'K, float> =
    match axis with
    | None ->
        let s = sum arr
        let n = float (Ndarray.numel arr)
        Ndarray.map (fun x -> x / n) s
    | Some ax ->
        let s = sum ~axis:ax arr
        let n = float (Ndarray.shape arr).[ax]
        Ndarray.map (fun x -> x / n) s

/// Stack arrays along a new axis
let stack (axis: int) (arrays: Ndarray<'K, 'T> array) : FowlResult<Ndarray<'K, 'T>> =
    if arrays.Length = 0 then
        Error.invalidArgument "stack requires at least one array"
    else
        let firstShape = Ndarray.shape arrays.[0]

        // Check all arrays have same shape
        for arr in arrays do
            if Ndarray.shape arr <> firstShape then
                Error.invalidArgument "All arrays must have the same shape"
        // Check axis is valid
        if axis < 0 || axis > firstShape.Length then
            Error.indexOutOfRange (sprintf "Axis %d out of range for shape with %d dimensions" axis firstShape.Length)
        else
            // Calculate output shape
            let outShape = Array.insertAt axis arrays.Length firstShape
            let outN = Shape.numel outShape
            let outData : 'T array = Array.zeroCreate outN
            let outStrides = Shape.stridesC outShape

            // Copy data from each array
            for arrIdx = 0 to arrays.Length - 1 do
                match arrays.[arrIdx] with
                | Dense d ->
                    // Iterate over all elements in input
                    let rec copy (inIdx: int array) (dim: int) =
                        if dim = firstShape.Length then
                            // Construct output index
                            let outIdx = Array.zeroCreate outShape.Length
                            let mutable inDim = 0
                            for j = 0 to outShape.Length - 1 do
                                if j = axis then
                                    outIdx.[j] <- arrIdx
                                else
                                    outIdx.[j] <- inIdx.[inDim]
                                    inDim <- inDim + 1

                            let srcFlat = Ndarray.flatIndex d.Strides inIdx d.Offset
                            let dstFlat = Ndarray.flatIndex outStrides outIdx 0
                            outData.[dstFlat] <- d.Data.[srcFlat]
                        else
                            for i = 0 to firstShape.[dim] - 1 do
                                inIdx.[dim] <- i
                                copy inIdx (dim + 1)

                    copy (Array.zeroCreate firstShape.Length) 0
                | Sparse _ -> return Error.notImplemented "stack not implemented for sparse arrays"

            Ndarray.ofArray outData outShape

/// Concatenate arrays along an existing axis
let concatenate (axis: int) (arrays: Ndarray<'K, 'T> array) : FowlResult<Ndarray<'K, 'T>> =
    if arrays.Length = 0 then
        Error.invalidArgument "concatenate requires at least one array"
    else
        let firstShape = Ndarray.shape arrays.[0]

        // Check axis is valid
        if axis < 0 || axis >= firstShape.Length then
            Error.indexOutOfRange (sprintf "Axis %d out of range for shape with %d dimensions" axis firstShape.Length)
        else
            // Check shapes are compatible (all same except concat axis)
            let totalSize =
                arrays
                |> Array.sumBy (fun arr -> (Ndarray.shape arr).[axis])

            for arr in arrays do
                let shape = Ndarray.shape arr
                if shape.Length <> firstShape.Length then
                    Error.invalidArgument "All arrays must have same number of dimensions"
                else
                    for i = 0 to shape.Length - 1 do
                        if i <> axis && shape.[i] <> firstShape.[i] then
                            Error.invalidArgument "All arrays must have same shape except along concat axis"

            // Calculate output shape
            let outShape = Array.copy firstShape
            outShape.[axis] <- totalSize

            // TODO: Implement actual concatenation
            Error.notImplemented "concatenate not fully implemented yet"

/// Split array into multiple sub-arrays
let split (axis: int) (indices: int array) (arr: Ndarray<'K, 'T>) : FowlResult<Ndarray<'K, 'T> array> =
    let shape = Ndarray.shape arr
    if axis < 0 || axis >= shape.Length then
        Error.indexOutOfRange (sprintf "Axis %d out of range for shape with %d dimensions" axis shape.Length)
    else
        // Sort indices
        let sortedIdx = indices |> Array.sort

        // Calculate number of splits
        let nSplits = sortedIdx.Length + 1

        let splits = Array.init nSplits (fun i ->
            let start = if i = 0 then 0 else sortedIdx.[i - 1]
            let stop = if i = sortedIdx.Length then shape.[axis] else sortedIdx.[i]

            // Create slice spec for this split
            let specs =
                shape
                |> Array.mapi (fun j _ ->
                    if j = axis then
                        Range (Some start, Some stop, None)
                    else
                        All)

            slice arr specs)

        // Check if any split failed
        if splits |> Array.exists (function Error _ -> true | _ -> false) then
            Error.invalidArgument "split failed due to invalid indices"
        else
            splits |> Array.map (function Ok x -> x | Error _ -> failwith "unreachable") |> Ok
