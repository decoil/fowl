# Chapter 3: Array Manipulation

## 3.1 Reshaping Arrays

### Changing Dimensions

```fsharp
open Fowl
open Fowl.Core.Types

let arr = Ndarray.ofArray [|1.0 .. 24.0|] [||] |> unwrap

// Reshape to 2D
let matrix = Ndarray.reshape [|4; 6|] arr |> unwrap
// Shape: [4; 6]

// Reshape to 3D
let tensor = Ndarray.reshape [|2; 3; 4|] arr |> unwrap
// Shape: [2; 3; 4]

// Flatten
let flat = Ndarray.reshape [|-1|] matrix |> unwrap
// Shape: [24]

// Automatic dimension inference
let auto = Ndarray.reshape [|-1; 3|] arr |> unwrap
// Shape: [8; 3]
```

### Expanding and Squeezing

```fsharp
// Add singleton dimension
let rowVector = Ndarray.reshape [|1; 5|] data |> unwrap
// Shape: [1; 5]

let colVector = Ndarray.reshape [|5; 1|] data |> unwrap
// Shape: [5; 1]

// Remove singleton dimensions
let squeezed = Ndarray.reshape [|5|] rowVector |> unwrap
// Shape: [5]
```

## 3.2 Joining Arrays

### Concatenation

```fsharp
open Fowl.Core.Matrix

let a = Ndarray.ones<Float64> [|3; 3|] |> unwrap
let b = Ndarray.zeros<Float64> [|3; 3|] |> unwrap

// Vertical stack (axis 0)
let vstack = stack 0 [a; b] |> unwrap
// Shape: [6; 3]

// Horizontal stack (axis 1)
let hstack = stack 1 [a; b] |> unwrap
// Shape: [3; 6]

// Depth stack (axis 2)
let dstack = stack 2 [a; b] |> unwrap
// Shape: [3; 3; 2]
```

### Splitting

```fsharp
open Fowl.Core.NdarrayOps

let arr = Ndarray.ofArray [|1.0 .. 12.0|] [|3; 4|] |> unwrap

// Split into equal parts
let parts = split arr 1 2 |> unwrap
// Two arrays of shape [3; 2] each

// Split at specific indices
let splitAt = splitAt arr 1 [|1; 3|] |> unwrap
// Splits at columns 1 and 3
```

## 3.3 Repeating and Tiling

```fsharp
// Tile (repeat entire array)
let tiled = tile arr [|2; 3|] |> unwrap
// Repeats arr 2 times vertically, 3 times horizontally

// Repeat elements
let repeated = repeat arr 3 |> unwrap
// Each element repeated 3 times
```

## 3.4 Advanced Indexing

### Boolean Indexing

```fsharp
let arr = Ndarray.ofArray [|1.0; -2.0; 3.0; -4.0; 5.0|] [||] |> unwrap

// Select positive values
let positive = 
    arr
    |> Ndarray.toArray
    |> Array.filter (fun x -> x > 0.0)
    |> fun data -> Ndarray.ofArray data [||]
// [|1.0; 3.0; 5.0|]

// Select based on condition
let mask = [|true; false; true; false; true|]
let selected = 
    mask
    |> Array.mapi (fun i keep -> if keep then Some i else None)
    |> Array.choose id
    |> Array.map (fun i -
        match Ndarray.get arr [|i|] with
        | Ok v -> v
        | Error _ -> 0.0)
    |> fun data -> Ndarray.ofArray data [||]
```

### Fancy Indexing

```fsharp
let arr = Ndarray.ofArray [|10.0; 20.0; 30.0; 40.0; 50.0|] [||] |> unwrap

// Select specific indices
let indices = [|0; 2; 4|]
let selected =
    indices
    |> Array.map (fun i -
        match Ndarray.get arr [|i|] with
        | Ok v -> v
        | Error _ -> 0.0)
    |> fun data -> Ndarray.ofArray data [||]
// [|10.0; 30.0; 50.0|]
```

## 3.5 Sorting and Searching

### Sorting

```fsharp
let arr = Ndarray.ofArray [|3.0; 1.0; 4.0; 1.0; 5.0; 9.0|] [||] |> unwrap

// Sort in ascending order
let sorted = NdarrayOps.sort arr |> unwrap
// [|1.0; 1.0; 3.0; 4.0; 5.0; 9.0|]

// Get sort indices
let indices = NdarrayOps.argsort arr |> unwrap
// [|1; 3; 0; 2; 4; 5|] (indices that would sort the array)

// Sort 2D array along axis
let matrix = Ndarray.ofArray [|3.0; 1.0; 4.0; 2.0|] [|2; 2|] |> unwrap
let sortedRows = NdarrayOps.sortAxis matrix 1 |> unwrap
// Each row sorted: [|1.0; 3.0|], [|2.0; 4.0|]
```

### Finding Extrema

```fsharp
// Argmax - index of maximum
let maxIdx = NdarrayOps.argmax arr |> unwrap
// [|5|] (index of 9.0)

// Argmin - index of minimum
let minIdx = NdarrayOps.argmin arr |> unwrap
// [|1|] or [|3|] (index of 1.0)

// 2D argmax
let matrix = Ndarray.ofArray [|1.0; 5.0; 3.0; 2.0|] [|2; 2|] |> unwrap
let rowMaxIndices = NdarrayOps.argmax matrix |> unwrap
// [|1; 0|] (max in each row at column 1 and 0)
```

### Clipping

```fsharp
// Clip values to range
let clipped = NdarrayOps.clip arr 2.0 5.0 |> unwrap
// Values < 2 become 2, values > 5 become 5
// [|2.0; 2.0; 3.0; 4.0; 5.0; 5.0|]
```

## 3.6 Set Operations

```fsharp
// Unique values
let unique =
    arr
    |> Ndarray.toArray
    |> Array.distinct
    |> fun data -> Ndarray.ofArray data [||]

// Set intersection
let set1 = [|1.0; 2.0; 3.0; 4.0|]
let set2 = [|3.0; 4.0; 5.0; 6.0|]
let intersection = Set.intersect (Set.ofArray set1) (Set.ofArray set2)

// Set union
let union = Set.union (Set.ofArray set1) (Set.ofArray set2)
```

## 3.7 Broadcasting

### Broadcasting Rules

```fsharp
// Scalar broadcast
let a = Ndarray.ones<Float64> [|3; 3|] |> unwrap
let scalar = Ndarray.ofArray [|5.0|] [||] |> unwrap
let result = Ndarray.add a scalar |> unwrap
// 5.0 is broadcast to [3; 3]

// Row broadcast
let row = Ndarray.ofArray [|1.0; 2.0; 3.0|] [||] |> unwrap
let rowBroadcast = Ndarray.reshape [|1; 3|] row |> unwrap
let rowResult = Ndarray.add a rowBroadcast |> unwrap
// Row is broadcast to each row

// Column broadcast
let col = Ndarray.ofArray [|1.0; 2.0; 3.0|] [||] |> unwrap
let colBroadcast = Ndarray.reshape [|3; 1|] col |> unwrap
let colResult = Ndarray.add a colBroadcast |> unwrap
// Column is broadcast to each column
```

## 3.8 Views and Copies

```fsharp
open Fowl.Memory.NdarrayView

// Create a view (shares data)
let arr = Ndarray.ofArray [|1.0 .. 12.0|] [|3; 4|] |> unwrap
let view = NdarrayView.row arr 0 |> unwrap
// View of first row, no data copied

// Slice creates view
let sub = Slice.slice arr [|Index 0; All|] |> unwrap

// Explicit copy
let copy = Ndarray.toArray arr |> fun data -> Ndarray.ofArray data (Ndarray.shape arr)
```

## 3.9 Exercises

### Exercise 3.1: Matrix Reversal

```fsharp
// Reverse matrix in both dimensions
let reverseMatrix (arr: Ndarray<'K, float>) : FowlResult<Ndarray<'K, float>> =
    result {
        let! shape = Ndarray.shape arr |> Result.ofOption (Error.invalidState "No shape")
        
        if shape.Length <> 2 then
            return! Error.invalidShape "reverseMatrix requires 2D array"
        
        let data = Ndarray.toArray arr
        let rows = shape.[0]
        let cols = shape.[1]
        
        let reversed = Array2D.init rows cols (fun i j -
            data.[(rows - 1 - i) * cols + (cols - 1 - j)])
        
        return! Ndarray.ofArray2D reversed
    }
```

### Exercise 3.2: Sliding Window

```fsharp
// Create sliding windows over array
let slidingWindow (arr: Ndarray<'K, float>) (windowSize: int) : FowlResult<Ndarray<'K, float>> =
    result {
        let data = Ndarray.toArray arr
        let n = data.Length
        
        if windowSize > n then
            return! Error.invalidArgument "Window size larger than array"
        
        let numWindows = n - windowSize + 1
        let windows =
            [|
                for i = 0 to numWindows - 1 do
                    yield data.[i .. i + windowSize - 1]
            |]
            |> Array.concat
        
        return! Ndarray.ofArray windows [|numWindows; windowSize|]
    }
```

### Exercise 3.3: One-Hot Encoding

```fsharp
// Convert class indices to one-hot vectors
let oneHot (indices: int[]) (numClasses: int) : Ndarray<Float64, float> =
    let n = indices.Length
    let data = Array.zeroCreate (n * numClasses)
    
    for i = 0 to n - 1 do
        let idx = indices.[i]
        if idx >= 0 && idx < numClasses then
            data.[i * numClasses + idx] <- 1.0
    
    Ndarray.ofArray data [|n; numClasses|] |> function Ok x -> x | Error _ -> failwith "Invalid"
```

## 3.10 Summary

Key concepts:
- Reshaping changes dimensions without copying data
- Stacking combines arrays along new or existing axes
- Broadcasting aligns arrays of different shapes
- Views share data; copies are independent
- Sorting returns values or indices
- Clipping enforces value bounds

---

*Next: [Chapter 4: Linear Algebra Fundamentals](chapter04.md)*
