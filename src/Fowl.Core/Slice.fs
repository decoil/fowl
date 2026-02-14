module Fowl.Core.Slice

open Fowl

/// Parse a slice specification into concrete indices for a given dimension size
let parseSlice (dimSize: int) (spec: SliceSpec) : int Result array =
    match spec with
    | All -> Ok [|0 .. dimSize - 1|]
    | Index i ->
        let idx = if i < 0 then dimSize + i else i
        if idx < 0 || idx >= dimSize then
            Error.indexOutOfRange (sprintf "Index %d out of range for dimension of size %d" i dimSize)
        else
            Ok [|idx|]
    | Range (start, stop, step) ->
        let startIdx =
            match start with
            | None -> Ok 0
            | Some s -> Ok (if s < 0 then dimSize + s else s)
        let stopIdx =
            match stop with
            | None -> Ok dimSize
            | Some s -> Ok (if s < 0 then dimSize + s else s)
        let stepSize =
            match step with
            | None -> Ok 1
            | Some s ->
                if s = 0 then Error.invalidArgument "Step cannot be zero"
                else Ok s
        
        Result.map3 (fun s e st ->
            let actualStop = if st > 0 then e else e + 1
            [|s .. st .. actualStop - 1|] |> Array.filter (fun i -> i >= 0 && i < dimSize)) startIdx stopIdx stepSize
    | IndexArray indices ->
        let result = indices |> Array.map (fun i ->
            let idx = if i < 0 then dimSize + i else i
            if idx < 0 || idx >= dimSize then
                Error.indexOutOfRange (sprintf "Index %d out of range for dimension of size %d" i dimSize)
            else
                Ok idx)
        // If any index is error, return error
        if result |> Array.exists (function Error _ -> true | _ -> false) then
            result |> Array.tryPick (function Error e -> Some e | _ -> None) |> Option.defaultWith (fun () -> Error.indexOutOfRange "Invalid index")
        else
            result |> Array.map (function Ok i -> i | _ -> failwith "unreachable") |> Ok

/// Calculate output shape from slice specifications
let sliceShape (shape: Shape) (specs: SliceSpec array) : FowlResult<Shape> =
    if specs.Length > shape.Length then
        Error.invalidArgument "Too many slice specifications"
    else
        // Parse all specs and check for errors
        let parseResults = specs |> Array.mapi (fun i spec -> parseSlice shape.[i] spec)
        let indicesArrays =
            parseResults
            |> Array.mapi (fun i r ->
                match r with
                | Ok indices -> indices
                | Error e -> failwithf "parseSlice error at dim %d: %A" i e)
        
        // Build result shape
        let mutable result = []
        for i = 0 to specs.Length - 1 do
            result <- indicesArrays.[i].Length :: result
        for i = specs.Length to shape.Length - 1 do
            result <- shape.[i] :: result
        result |> List.rev |> List.toArray |> Ok

/// Slice an ndarray (returns a copy, not a view)
let slice (arr: Ndarray<'K, 'T>) (specs: SliceSpec array) : FowlResult<Ndarray<'K, 'T>> =
    match arr with
    | Dense d ->
        result {
            let! outShape = sliceShape d.Shape specs
            let n = Shape.numel outShape
            let outData = Array.zeroCreate n

            // Parse all specs into index arrays
            let indexArrays = specs |> Array.mapi (fun i spec ->
                match parseSlice d.Shape.[i] spec with
                | Ok indices -> indices
                | Error e -> return! Error e)

            // Generate all index combinations
            let generateCombinations (dims: int array array) : int array array =
                let rec helper (dims: int array array) (current: int list) (acc: int array list) =
                    if current.Length = dims.Length then
                        (current |> List.rev |> List.toArray) :: acc
                    else
                        let dimIdx = current.Length
                        dims.[dimIdx]
                        |> Array.fold (fun acc2 i ->
                            helper dims (i :: current) acc2) acc
                helper dims [] [] |> List.toArray

            // Generate all input index combinations
            let allIndices = generateCombinations indexArrays

            // Fill output data using Array.mapi (no mutable state needed)
            let outData' =
                allIndices
                |> Array.mapi (fun outIdx outIndices ->
                    // Map output indices back to input indices
                    let inputIndices = Array.zeroCreate d.Shape.Length
                    for i = 0 to specs.Length - 1 do
                        inputIndices.[i] <- outIndices.[i]
                    for i = specs.Length to d.Shape.Length - 1 do
                        inputIndices.[i] <- outIndices.[i]

                    let srcIdx = Ndarray.flatIndex d.Strides inputIndices d.Offset
                    d.Data.[srcIdx])

            return! Ndarray.ofArray outData' outShape
        }
    | Sparse _ -> Error.notImplemented "slice not implemented for sparse arrays"

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

/// Broadcast an array to a target shape
let broadcastTo (targetShape: Shape) (arr: Ndarray<'K, 'T>) : FowlResult<Ndarray<'K, 'T>> =
    let currentShape = Ndarray.shape arr
    if currentShape = targetShape then
        Ok arr
    elif not (broadcastable currentShape targetShape) then
        Error.dimensionMismatch (sprintf "Cannot broadcast shape %A to %A" currentShape targetShape)
    else
        match arr with
        | Dense d ->
            result {
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

                // Generate all output indices
                let generateIndices (shape: Shape) : int array array =
                    let rec helper (shape: Shape) (current: int list) (acc: int array list) =
                        if current.Length = shape.Length then
                            (current |> List.rev |> List.toArray) :: acc
                        else
                            let dimIdx = current.Length
                            [|0 .. shape.[dimIdx] - 1|]
                            |> Array.fold (fun acc2 i ->
                                helper shape (i :: current) acc2) acc
                    helper shape [] [] |> List.toArray

                let allIndices = generateIndices targetShape

                // Fill output data using Array.mapi (no mutable state needed)
                let outData' =
                    allIndices
                    |> Array.mapi (fun _ idx ->
                        // Calculate input flat index
                        let inFlatIdx =
                            paddedStrides |> Array.mapi (fun i s ->
                                let dim = paddedShape.[i]
                                if dim = 1 then 0 else idx.[i] * s)
                            |> Array.sum
                            |> (+) d.Offset
                        d.Data.[inFlatIdx])

                return! Ndarray.ofArray outData' targetShape
            }
        | Sparse _ -> Error.notImplemented "broadcastTo not implemented for sparse arrays"
