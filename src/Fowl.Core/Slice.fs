module Fowl.Core.Slice

open Fowl

/// Parse a slice specification into concrete indices for a given dimension size
let parseSlice (dimSize: int) (spec: SliceSpec) : int array =
    match spec with
    | All -> [|0 .. dimSize - 1|]
    | Index i -> 
        let idx = if i < 0 then dimSize + i else i
        if idx < 0 || idx >= dimSize then
            failwithf "Index %d out of range for dimension of size %d" i dimSize
        [|idx|]
    | Range (start, stop, step) ->
        let startIdx = 
            match start with
            | None -> 0
            | Some s -> if s < 0 then dimSize + s else s
        let stopIdx =
            match stop with
            | None -> dimSize
            | Some s -> if s < 0 then dimSize + s else s
        let stepSize =
            match step with
            | None -> 1
            | Some s -> if s = 0 then failwith "Step cannot be zero" else s
        
        let actualStop = if stepSize > 0 then stopIdx else stopIdx + 1
        [|startIdx .. stepSize .. actualStop - 1|] |> Array.filter (fun i -> i >= 0 && i < dimSize)
    | IndexArray indices ->
        indices |> Array.map (fun i -> 
            let idx = if i < 0 then dimSize + i else i
            if idx < 0 || idx >= dimSize then
                failwithf "Index %d out of range for dimension of size %d" i dimSize
            idx)

/// Calculate output shape from slice specifications
let sliceShape (shape: Shape) (specs: SliceSpec array) : Shape =
    if specs.Length > shape.Length then
        failwith "Too many slice specifications"
    
    let mutable result = []
    for i = 0 to specs.Length - 1 do
        let indices = parseSlice shape.[i] specs.[i]
        result <- indices.Length :: result
    
    // Remaining dimensions (if any) are kept as-is
    for i = specs.Length to shape.Length - 1 do
        result <- shape.[i] :: result
    
    result |> List.rev |> List.toArray

/// Slice an ndarray (returns a copy, not a view)
let slice (arr: Ndarray<'K, 'T>) (specs: SliceSpec array) : Ndarray<'K, 'T> =
    match arr with
    | Dense d ->
        let outShape = sliceShape d.Shape specs
        let n = Shape.numel outShape
        let outData = Array.zeroCreate n
        
        // Parse all specs into index arrays
        let indexArrays = specs |> Array.mapi (fun i spec -> parseSlice d.Shape.[i] spec)
        
        // Helper to iterate over all combinations of indices
        let rec iterate (dims: int array array) (current: int list) (outIdx: int ref) (f: int array -> unit) =
            if current.Length = dims.Length then
                f (current |> List.rev |> List.toArray)
                outIdx := !outIdx + 1
            else
                let dimIdx = current.Length
                for i in dims.[dimIdx] do
                    iterate dims (i :: current) outIdx f
        
        let outIdx = ref 0
        iterate indexArrays [] outIdx (fun outIndices ->
            // Map output indices back to input indices
            let inputIndices = Array.zeroCreate d.Shape.Length
            for i = 0 to specs.Length - 1 do
                inputIndices.[i] <- outIndices.[i]
            for i = specs.Length to d.Shape.Length - 1 do
                inputIndices.[i] <- outIndices.[i]
            
            let srcIdx = Ndarray.flatIndex d.Strides inputIndices d.Offset
            outData.[!outIdx] <- d.Data.[srcIdx]
        )
        
        Ndarray.ofArray outData outShape
    | Sparse _ -> failwith "slice not implemented for sparse arrays"

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
let broadcastShape (shape1: Shape) (shape2: Shape) : Shape =
    if not (broadcastable shape1 shape2) then
        failwithf "Shapes %A and %A are not broadcastable" shape1 shape2
    
    let n1 = shape1.Length
    let n2 = shape2.Length
    let maxN = max n1 n2
    
    let result = Array.zeroCreate maxN
    for i = 0 to maxN - 1 do
        let dim1 = if i < n1 then shape1.[n1 - 1 - i] else 1
        let dim2 = if i < n2 then shape2.[n2 - 1 - i] else 1
        result.[maxN - 1 - i] <- max dim1 dim2
    result

/// Broadcast an array to a target shape
let broadcastTo (targetShape: Shape) (arr: Ndarray<'K, 'T>) : Ndarray<'K, 'T> =
    let currentShape = Ndarray.shape arr
    if currentShape = targetShape then
        arr
    elif not (broadcastable currentShape targetShape) then
        failwithf "Cannot broadcast shape %A to %A" currentShape targetShape
    else
        match arr with
        | Dense d ->
            // For now, create a new array with the broadcasted shape
            // This is inefficient but correct
            let n = Shape.numel targetShape
            let outData = Array.zeroCreate n
            
            // Calculate strides for broadcasting
            let inStrides = d.Strides
            let outStrides = Shape.stridesC targetShape
            
            // Pad input shape with 1s on the left
            let nIn = currentShape.Length
            let nOut = targetShape.Length
            let paddedShape = Array.append (Array.create (nOut - nIn) 1) currentShape
            let paddedStrides = Array.append (Array.create (nOut - nIn) 0) inStrides
            
            // Iterate over output indices
            let rec iterate idx outFlatIdx =
                if idx.Length = nOut then
                    // Calculate input flat index
                    let inFlatIdx = 
                        paddedStrides |> Array.mapi (fun i s -> 
                            let dim = paddedShape.[i]
                            if dim = 1 then 0 else idx.[i] * s)
                        |> Array.sum
                        |> (+) d.Offset
                    outData.[outFlatIdx] <- d.Data.[inFlatIdx]
                else
                    let dimIdx = idx.Length
                    for i = 0 to targetShape.[dimIdx] - 1 do
                        iterate (Array.append idx [|i|]) (outFlatIdx * targetShape.[dimIdx] + i)
            
            iterate [||] 0
            Ndarray.ofArray outData targetShape
        | Sparse _ -> failwith "broadcastTo not implemented for sparse arrays"
