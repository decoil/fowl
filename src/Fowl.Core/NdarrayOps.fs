namespace Fowl.Core

open System
open Fowl.Core.Types

/// <summary>Additional ndarray operations.
/// Sorting, tiling, repeating, and utility functions.
/// </summary>module NdarrayOps =
    
    /// <summary>Sort array in ascending order.
/// Returns sorted array (copy).
/// </summary>let sort (arr: Ndarray<float>) : FowlResult<Ndarray<float>> =
        result {
            let data = Ndarray.toArray arr
            let sorted = Array.sort data
            return! Ndarray.ofArray sorted (Ndarray.shape arr)
        }
    
    /// <summary>Sort array along specified axis.
/// </summary>let sortAxis (arr: Ndarray<float>) (axis: int) : FowlResult<Ndarray<float>> =
        result {
            let! shape = Ndarray.shape arr |> Result.ofOption (Error.invalidState "Cannot get shape")
            let rank = shape.Length
            
            if axis < 0 || axis >= rank then
                return! Error.invalidArgument (sprintf "sortAxis: axis %d out of range for rank %d" axis rank)
            
            let data = Ndarray.toArray arr
            
            if rank = 1 then
                return! sort arr
            elif rank = 2 then
                let rows = shape.[0]
                let cols = shape.[1]
                let result = Array2D.zeroCreate rows cols
                
                if axis = 0 then
                    // Sort each column
                    for c = 0 to cols - 1 do
                        let col = Array.init rows (fun r -> data.[r * cols + c])
                        let sortedCol = Array.sort col
                        for r = 0 to rows - 1 do
                            result.[r, c] <- sortedCol.[r]
                else
                    // Sort each row
                    for r = 0 to rows - 1 do
                        let row = Array.init cols (fun c -> data.[r * cols + c])
                        let sortedRow = Array.sort row
                        for c = 0 to cols - 1 do
                            result.[r, c] <- sortedRow.[c]
                
                return! Ndarray.ofArray2D result
            else
                return! Error.notImplemented "sortAxis for rank > 2 not yet implemented"
        }
    
    /// <summary>Return indices that would sort the array.
/// </summary>let argsort (arr: Ndarray<float>) : FowlResult<int[]> =
        result {
            let data = Ndarray.toArray arr
            let indexed = data |> Array.mapi (fun i x -> (i, x))
            let sorted = indexed |> Array.sortBy snd
            return sorted |> Array.map fst
        }
    
    /// <summary>Tile array by repeating it.
/// </summary>let tile (arr: Ndarray<float>) (reps: int[]) : FowlResult<Ndarray<float>> =
        result {
            let! shape = Ndarray.shape arr |> Result.ofOption (Error.invalidState "Cannot get shape")
            
            if reps.Length < shape.Length then
                return! Error.invalidArgument "tile: reps length must be >= array rank"
            
            // Pad shape with 1s to match reps length
            let paddedShape = 
                Array.init reps.Length (fun i -
                    if i < reps.Length - shape.Length then 1
                    else shape.[i - (reps.Length - shape.Length)])
            
            let resultShape = Array.map2 (*) paddedShape reps
            let resultSize = Array.reduce (*) resultShape
            let resultData = Array.zeroCreate resultSize
            
            let sourceData = Ndarray.toArray arr
            let sourceSize = sourceData.Length
            
            // Simple case: 1D array tiled to 1D
            if reps.Length = 1 then
                for i = 0 to resultSize - 1 do
                    resultData.[i] <- sourceData.[i % sourceSize]
            // 2D case
            elif reps.Length = 2 then
                let srcRows = paddedShape.[0]
                let srcCols = paddedShape.[1]
                let dstRows = resultShape.[0]
                let dstCols = resultShape.[1]
                
                for r = 0 to dstRows - 1 do
                    for c = 0 to dstCols - 1 do
                        let srcR = r % srcRows
                        let srcC = c % srcCols
                        if srcR < shape.[0] && srcC < shape.[1] then
                            resultData.[r * dstCols + c] <- sourceData.[srcR * shape.[1] + srcC]
                        else
                            resultData.[r * dstCols + c] <- 0.0
            else
                return! Error.notImplemented "tile for rank > 2 not yet implemented"
            
            return! Ndarray.ofArray resultData resultShape
        }
    
    /// <summary>Repeat elements of array.
/// </summary>let repeat (arr: Ndarray<float>) (repeats: int) : FowlResult<Ndarray<float>> =
        result {
            if repeats < 0 then
                return! Error.invalidArgument "repeat: repeats must be non-negative"
            
            let data = Ndarray.toArray arr
            let! shape = Ndarray.shape arr |> Result.ofOption (Error.invalidState "Cannot get shape")
            
            if repeats = 0 then
                return! Ndarray.zeros shape
            
            let resultSize = data.Length * repeats
            let result = Array.zeroCreate resultSize
            
            for i = 0 to resultSize - 1 do
                result.[i] <- data.[i % data.Length]
            
            // New shape: last dimension multiplied by repeats
            let resultShape = Array.copy shape
            resultShape.[resultShape.Length - 1] <- resultShape.[resultShape.Length - 1] * repeats
            
            return! Ndarray.ofArray result resultShape
        }
    
    /// <summary>Reverse array (flip).
/// </summary>let reverse (arr: Ndarray<float>) : FowlResult<Ndarray<float>> =
        result {
            let data = Ndarray.toArray arr
            let reversed = Array.rev data
            return! Ndarray.ofArray reversed (Ndarray.shape arr)
        }
    
    /// <summary>Flip array along specified axis.
/// </summary>let flipAxis (arr: Ndarray<float>) (axis: int) : FowlResult<Ndarray<float>> =
        result {
            let! shape = Ndarray.shape arr |> Result.ofOption (Error.invalidState "Cannot get shape")
            let rank = shape.Length
            
            if axis < 0 || axis >= rank then
                return! Error.invalidArgument (sprintf "flipAxis: axis %d out of range" axis)
            
            let data = Ndarray.toArray arr
            
            if rank = 1 then
                return! reverse arr
            elif rank = 2 then
                let rows = shape.[0]
                let cols = shape.[1]
                let result = Array2D.zeroCreate rows cols
                
                if axis = 0 then
                    // Flip rows
                    for r = 0 to rows - 1 do
                        for c = 0 to cols - 1 do
                            result.[rows - 1 - r, c] <- data.[r * cols + c]
                else
                    // Flip columns
                    for r = 0 to rows - 1 do
                        for c = 0 to cols - 1 do
                            result.[r, cols - 1 - c] <- data.[r * cols + c]
                
                return! Ndarray.ofArray2D result
            else
                return! Error.notImplemented "flipAxis for rank > 2 not yet implemented"
        }
    
    /// <summary>Return indices of maximum values.
/// </summary>let argmax (arr: Ndarray<float>) : FowlResult<int[]> =
        result {
            let! shape = Ndarray.shape arr |> Result.ofOption (Error.invalidState "Cannot get shape")
            let data = Ndarray.toArray arr
            
            if shape.Length = 1 then
                let maxIdx = data |> Array.mapi (fun i x -> (i, x))
                            |> Array.maxBy snd
                            |> fst
                return [|maxIdx|]
            elif shape.Length = 2 then
                let rows = shape.[0]
                let cols = shape.[1]
                let result = Array.zeroCreate rows
                
                for r = 0 to rows - 1 do
                    let row = Array.init cols (fun c -> data.[r * cols + c])
                    let maxIdx = row |> Array.mapi (fun i x -> (i, x))
                                |> Array.maxBy snd
                                |> fst
                    result.[r] <- maxIdx
                
                return result
            else
                return! Error.notImplemented "argmax for rank > 2 not yet implemented"
        }
    
    /// <summary>Return indices of minimum values.
/// </summary>let argmin (arr: Ndarray<float>) : FowlResult<int[]> =
        result {
            let! shape = Ndarray.shape arr |> Result.ofOption (Error.invalidState "Cannot get shape")
            let data = Ndarray.toArray arr
            
            if shape.Length = 1 then
                let minIdx = data |> Array.mapi (fun i x -> (i, x))
                            |> Array.minBy snd
                            |> fst
                return [|minIdx|]
            elif shape.Length = 2 then
                let rows = shape.[0]
                let cols = shape.[1]
                let result = Array.zeroCreate rows
                
                for r = 0 to rows - 1 do
                    let row = Array.init cols (fun c -> data.[r * cols + c])
                    let minIdx = row |> Array.mapi (fun i x -> (i, x))
                                |> Array.minBy snd
                                |> fst
                    result.[r] <- minIdx
                
                return result
            else
                return! Error.notImplemented "argmin for rank > 2 not yet implemented"
        }
    
    /// <summary>Clip (limit) values in array.
/// </summary>let clip (arr: Ndarray<float>) (minVal: float) (maxVal: float) : FowlResult<Ndarray<float>> =
        result {
            let data = Ndarray.toArray arr
            let clipped = data |> Array.map (fun x -
                if x < minVal then minVal
                elif x > maxVal then maxVal
                else x)
            return! Ndarray.ofArray clipped (Ndarray.shape arr)
        }