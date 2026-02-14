namespace Fowl

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

module Shape =
    /// Calculate total number of elements from shape
    let numel (shape: Shape) =
        shape |> Array.fold (*) 1
    
    /// Calculate strides from shape (C-layout)
    let stridesC (shape: Shape) =
        let n = shape.Length
        let s = Array.zeroCreate n
        let mutable stride = 1
        for i = n - 1 downto 0 do
            s.[i] <- stride
            stride <- stride * shape.[i]
        s
    
    /// Calculate strides from shape (Fortran-layout)
    let stridesF (shape: Shape) =
        let n = shape.Length
        let s = Array.zeroCreate n
        let mutable stride = 1
        for i = 0 to n - 1 do
            s.[i] <- stride
            stride <- stride * shape.[i]
        s

module Ndarray =
    open Shape
    
    /// Create empty ndarray
    let empty<'K> (shape: Shape) : Ndarray<'K, 'T> =
        let n = numel shape
        let data = Array.zeroCreate n
        Dense {
            Data = data
            Shape = shape
            Strides = stridesC shape
            Offset = 0
            Layout = CLayout
        }
    
    /// Create ndarray filled with zeros
    let zeros<'K> (shape: Shape) : Ndarray<'K, float> =
        let n = numel shape
        let data = Array.zeroCreate n
        Dense {
            Data = data
            Shape = shape
            Strides = stridesC shape
            Offset = 0
            Layout = CLayout
        }
    
    /// Create ndarray filled with ones
    let ones<'K> (shape: Shape) : Ndarray<'K, float> =
        let n = numel shape
        let data = Array.create n 1.0
        Dense {
            Data = data
            Shape = shape
            Strides = stridesC shape
            Offset = 0
            Layout = CLayout
        }
    
    /// Create ndarray filled with specific value
    let create<'K> (shape: Shape) (value: 'T) : Ndarray<'K, 'T> =
        let n = numel shape
        let data = Array.create n value
        Dense {
            Data = data
            Shape = shape
            Strides = stridesC shape
            Offset = 0
            Layout = CLayout
        }
    
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
    
    /// Get element at flat index
    let getFlat (arr: Ndarray<'K, 'T>) (idx: int) : 'T =
        match arr with
        | Dense d -> d.Data.[d.Offset + idx]
        | Sparse _ -> failwith "getFlat not implemented for sparse arrays"
    
    /// Set element at flat index
    let setFlat (arr: Ndarray<'K, 'T>) (idx: int) (value: 'T) : unit =
        match arr with
        | Dense d -> d.Data.[d.Offset + idx] <- value
        | Sparse _ -> failwith "setFlat not implemented for sparse arrays"
    
    /// Calculate flat index from multi-dimensional indices
    let flatIndex (strides: int array) (indices: int array) (offset: int) =
        let mutable idx = offset
        for i = 0 to indices.Length - 1 do
            idx <- idx + indices.[i] * strides.[i]
        idx
    
    /// Get element at multi-dimensional indices
    let get (arr: Ndarray<'K, 'T>) (indices: int array) : 'T =
        match arr with
        | Dense d -> 
            let flatIdx = flatIndex d.Strides indices d.Offset
            d.Data.[flatIdx]
        | Sparse _ -> failwith "get not implemented for sparse arrays"
    
    /// Set element at multi-dimensional indices
    let set (arr: Ndarray<'K, 'T>) (indices: int array) (value: 'T) : unit =
        match arr with
        | Dense d ->
            let flatIdx = flatIndex d.Strides indices d.Offset
            d.Data.[flatIdx] <- value
        | Sparse _ -> failwith "set not implemented for sparse arrays"
    
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
    
    /// Apply function to each element (in-place)
    let apply (f: 'T -> 'T) (arr: Ndarray<'K, 'T>) : unit =
        match arr with
        | Dense d ->
            for i = 0 to d.Data.Length - 1 do
                d.Data.[i] <- f d.Data.[i]
        | Sparse _ -> failwith "apply not implemented for sparse arrays"
    
    /// Reshape array (returns view if possible, copy otherwise)
    let reshape (newShape: Shape) (arr: Ndarray<'K, 'T>) : Ndarray<'K, 'T> =
        match arr with
        | Dense d ->
            if Shape.numel newShape <> Shape.numel d.Shape then
                failwith "Cannot reshape: different number of elements"
            Dense { d with Shape = newShape; Strides = stridesC newShape }
        | Sparse _ -> failwith "reshape not implemented for sparse arrays"
    
    /// Convert to flat array (copy)
    let toArray (arr: Ndarray<'K, 'T>) : 'T array =
        match arr with
        | Dense d -> Array.copy d.Data
        | Sparse _ -> failwith "toArray not implemented for sparse arrays"
    
    /// Create from flat array
    let ofArray (data: 'T array) (shape: Shape) : Ndarray<'K, 'T> =
        if data.Length <> Shape.numel shape then
            failwith "Data length does not match shape"
        Dense {
            Data = Array.copy data
            Shape = shape
            Strides = stridesC shape
            Offset = 0
            Layout = CLayout
        }
    
    /// Generate linearly spaced values
    let linspace (start: float) (stop: float) (num: int) : Ndarray<Float64, float> =
        if num < 2 then failwith "linspace requires num >= 2"
        let step = (stop - start) / float (num - 1)
        let data = Array.init num (fun i -> start + float i * step)
        ofArray data [|num|]
    
    /// Generate values with given step
    let arange (start: float) (stop: float) (step: float) : Ndarray<Float64, float> =
        if step = 0.0 then failwith "arange: step cannot be zero"
        let n = int ((stop - start) / step)
        let data = Array.init n (fun i -> start + float i * step)
        ofArray data [|n|]
    
    /// Create array with random values (uniform [0, 1))
    let random (shape: Shape) : Ndarray<Float64, float> =
        let rng = System.Random()
        let n = numel shape
        let data = Array.init n (fun _ -> rng.NextDouble())
        ofArray data shape
    
    /// Element-wise addition
    let add (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : Ndarray<'K, float> =
        match a, b with
        | Dense da, Dense db ->
            if da.Shape <> db.Shape then failwith "Shape mismatch in add"
            let newData = Array.map2 (+) da.Data db.Data
            Dense { da with Data = newData }
        | _ -> failwith "add not implemented for sparse arrays"
    
    /// Element-wise subtraction
    let sub (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : Ndarray<'K, float> =
        match a, b with
        | Dense da, Dense db ->
            if da.Shape <> db.Shape then failwith "Shape mismatch in sub"
            let newData = Array.map2 (-) da.Data db.Data
            Dense { da with Data = newData }
        | _ -> failwith "sub not implemented for sparse arrays"
    
    /// Element-wise multiplication
    let mul (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : Ndarray<'K, float> =
        match a, b with
        | Dense da, Dense db ->
            if da.Shape <> db.Shape then failwith "Shape mismatch in mul"
            let newData = Array.map2 (*) da.Data db.Data
            Dense { da with Data = newData }
        | _ -> failwith "mul not implemented for sparse arrays"
    
    /// Element-wise division
    let div (a: Ndarray<'K, float>) (b: Ndarray<'K, float>) : Ndarray<'K, float> =
        match a, b with
        | Dense da, Dense db ->
            if da.Shape <> db.Shape then failwith "Shape mismatch in div"
            let newData = Array.map2 (/) da.Data db.Data
            Dense { da with Data = newData }
        | _ -> failwith "div not implemented for sparse arrays"
    
    /// Scalar addition
    let addScalar (a: Ndarray<'K, float>) (s: float) : Ndarray<'K, float> =
        map (fun x -> x + s) a
    
    /// Scalar multiplication
    let mulScalar (a: Ndarray<'K, float>) (s: float) : Ndarray<'K, float> =
        map (fun x -> x * s) a
