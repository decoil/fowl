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
let transpose (arr: Ndarray<'K, 'T>) : Ndarray<'K, 'T> =
    match arr with
    | Dense d ->
        let nDims = d.Shape.Length
        if nDims < 2 then
            failwith "transpose requires at least 2D array"
        
        // Swap last two dimensions
        let newShape = Array.copy d.Shape
        newShape.[nDims - 2] <- d.Shape.[nDims - 1]
        newShape.[nDims - 1] <- d.Shape.[nDims - 2]
        
        // Calculate new strides
        let newStrides = Shape.stridesC newShape
        
        // For true transpose, we'd need to rearrange data
        // For now, create a copy with transposed layout
        let n = Shape.numel newShape
        let newData = Array.zeroCreate n
        
        // Simple implementation: iterate and copy
        let rec copyTransposed (inIdx: int array) (dim: int) =
            if dim = nDims then
                // Calculate output index with swapped last two dims
                let outIdx = Array.copy inIdx
                let temp = outIdx.[nDims - 2]
                outIdx.[nDims - 2] <- outIdx.[nDims - 1]
                outIdx.[nDims - 1] <- temp
                
                let srcFlat = Ndarray.flatIndex d.Strides inIdx d.Offset
                let dstFlat = Ndarray.flatIndex newStrides outIdx 0
                newData.[dstFlat] <- d.Data.[srcFlat]
            else
                for i = 0 to d.Shape.[dim] - 1 do
                    inIdx.[dim] <- i
                    copyTransposed inIdx (dim + 1)
        
        copyTransposed (Array.zeroCreate nDims) 0
        
        Dense {
            Data = newData
            Shape = newShape
            Strides = newStrides
            Offset = 0
            Layout = CLayout
        }
    | Sparse _ -> failwith "transpose not implemented for sparse arrays"

/// Matrix multiplication (2D arrays only)
let matmul (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : Ndarray<'K, float> =
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
        | _ -> failwithf "Incompatible shapes for matrix multiplication: %A and %A" da.Shape db.Shape
    | _ -> failwith "matmul not implemented for sparse arrays"

/// Dot product of two 1D arrays
let dot (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : float =
    match a, b with
    | Dense da, Dense db ->
        if da.Shape.Length <> 1 || db.Shape.Length <> 1 then
            failwith "dot requires 1D arrays"
        if da.Shape.[0] <> db.Shape.[0] then
            failwith "dot requires arrays of same length"
        
        let mutable sum = 0.0
        for i = 0 to da.Shape.[0] - 1 do
            sum <- sum + da.Data.[da.Offset + i * da.Strides.[0]] * db.Data.[db.Offset + i * db.Strides.[0]]
        sum
    | _ -> failwith "dot not implemented for sparse arrays"

/// Outer product of two 1D arrays
let outer (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : Ndarray<'K, float> =
    match a, b with
    | Dense da, Dense db ->
        if da.Shape.Length <> 1 || db.Shape.Length <> 1 then
            failwith "outer requires 1D arrays"
        
        let m = da.Shape.[0]
        let n = db.Shape.[0]
        let result = Array.zeroCreate (m * n)
        
        for i = 0 to m - 1 do
            for j = 0 to n - 1 do
                result.[i * n + j] <- 
                    da.Data.[da.Offset + i * da.Strides.[0]] * 
                    db.Data.[db.Offset + j * db.Strides.[0]]
        
        Ndarray.ofArray result [|m; n|]
    | _ -> failwith "outer not implemented for sparse arrays"

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
let stack (axis: int) (arrays: Ndarray<'K, 'T> array) : Ndarray<'K, 'T> =
    if arrays.Length = 0 then
        failwith "stack requires at least one array"
    
    let firstShape = Ndarray.shape arrays.[0]
    
    // Check all arrays have same shape
    for arr in arrays do
        if Ndarray.shape arr <> firstShape then
            failwith "All arrays must have the same shape"
    
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
        | Sparse _ -> failwith "stack not implemented for sparse arrays"
    
    Ndarray.ofArray outData outShape

/// Concatenate arrays along an existing axis
let concatenate (axis: int) (arrays: Ndarray<'K, 'T> array) : Ndarray<'K, 'T> =
    if arrays.Length = 0 then
        failwith "concatenate requires at least one array"
    
    let firstShape = Ndarray.shape arrays.[0]
    
    // Check shapes are compatible (all same except concat axis)
    let totalSize = 
        arrays 
        |> Array.sumBy (fun arr -> (Ndarray.shape arr).[axis])
    
    for arr in arrays do
        let shape = Ndarray.shape arr
        if shape.Length <> firstShape.Length then
            failwith "All arrays must have same number of dimensions"
        for i = 0 to shape.Length - 1 do
            if i <> axis && shape.[i] <> firstShape.[i] then
                failwith "All arrays must have same shape except along concat axis"
    
    // Calculate output shape
    let outShape = Array.copy firstShape
    outShape.[axis] <- totalSize
    
    // TODO: Implement actual concatenation
    failwith "concatenate not fully implemented yet"

/// Split array into multiple sub-arrays
let split (axis: int) (indices: int array) (arr: Ndarray<'K, 'T>) : Ndarray<'K, 'T> array =
    let shape = Ndarray.shape arr
    if axis < 0 || axis >= shape.Length then
        failwithf "Axis %d out of range" axis
    
    // Sort indices
    let sortedIdx = indices |> Array.sort
    
    // Calculate number of splits
    let nSplits = sortedIdx.Length + 1
    Array.init nSplits (fun i ->
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
        
        slice arr specs
    )
