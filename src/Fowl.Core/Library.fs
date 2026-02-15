/// <summary>Fowl Library - Core ndarray operations.</summary>
/// <remarks>
/// Provides fundamental ndarray creation and manipulation functions.
/// All functions work with the Ndarray types defined in Fowl.Core.Types.
/// </remarks>
namespace Fowl

open System
open Fowl.Core.Types

/// <summary>Module for shape-related operations.</summary>
module Shape =
    /// <summary>Calculate total number of elements from shape.</summary>
    /// <param name="shape">The shape array.</param>
    /// <returns>Total number of elements.</returns>
    let numel (shape: Shape) =
        shape |> Array.fold (*) 1

    /// <summary>Calculate strides from shape (C-layout/row-major).</summary>
    /// <param name="shape">The shape array.</param>
    /// <returns>Strides array for C-layout.</returns>
    let stridesC (shape: Shape) =
        let n = shape.Length
        let s = Array.zeroCreate n
        let mutable stride = 1
        for i = n - 1 downto 0 do
            s.[i] <- stride
            stride <- stride * shape.[i]
        s

    /// <summary>Calculate strides from shape (Fortran-layout/column-major).</summary>
    /// <param name="shape">The shape array.</param>
    /// <returns>Strides array for Fortran-layout.</returns>
    let stridesF (shape: Shape) =
        let n = shape.Length
        let s = Array.zeroCreate n
        let mutable stride = 1
        for i = 0 to n - 1 do
            s.[i] <- stride
            stride <- stride * shape.[i]
        s

/// <summary>Module for ndarray operations.</summary>
module Ndarray =
    open Shape

    /// <summary>Create empty ndarray with given shape.</summary>
    /// <typeparam name="'K">The phantom type for element kind.</typeparam>
    /// <param name="shape">The shape of the array.</param>
    /// <returns>Empty ndarray.</returns>
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

    /// <summary>Create ndarray filled with zeros.</summary>
    /// <typeparam name="'K">The phantom type for element kind.</typeparam>
    /// <param name="shape">The shape of the array.</param>
    /// <returns>Ndarray filled with zeros.</returns>
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

    /// <summary>Create ndarray filled with ones.</summary>
    /// <typeparam name="'K">The phantom type for element kind.</typeparam>
    /// <param name="shape">The shape of the array.</param>
    /// <returns>Ndarray filled with ones.</returns>
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

    /// <summary>Create ndarray filled with specific value.</summary>
    /// <typeparam name="'K">The phantom type for element kind.</typeparam>
    /// <param name="shape">The shape of the array.</param>
    /// <param name="value">The fill value.</param>
    /// <returns>Ndarray filled with value.</returns>
    let create<'K, 'T> (shape: Shape) (value: 'T) : Ndarray<'K, 'T> =
        let n = numel shape
        let data = Array.create n value
        Dense {
            Data = data
            Shape = shape
            Strides = stridesC shape
            Offset = 0
            Layout = CLayout
        }

    /// <summary>Get shape of ndarray.</summary>
    /// <param name="arr">The input array.</param>
    /// <returns>Shape array.</returns>
    let shape arr =
        match arr with
        | Dense d -> d.Shape
        | Sparse s -> s.Shape

    /// <summary>Get number of dimensions.</summary>
    /// <param name="arr">The input array.</param>
    /// <returns>Number of dimensions.</returns>
    let ndim arr =
        (shape arr).Length

    /// <summary>Get total number of elements.</summary>
    /// <param name="arr">The input array.</param>
    /// <returns>Total element count.</returns>
    let numel arr =
        shape arr |> Shape.numel

    /// <summary>Calculate flat index from multi-dimensional indices.</summary>
    /// <param name="strides">The strides array.</param>
    /// <param name="indices">The multi-dimensional indices.</param>
    /// <param name="offset">The offset.</param>
    /// <returns>Flat index.</returns>
    let flatIndex (strides: int array) (indices: int array) (offset: int) =
        let mutable idx = offset
        for i = 0 to indices.Length - 1 do
            idx <- idx + indices.[i] * strides.[i]
        idx

    /// <summary>Create ndarray from flat array.</summary>
    /// <typeparam name="'K">The phantom type for element kind.</typeparam>
    /// <typeparam name="'T">The element type.</typeparam>
    /// <param name="data">The flat data array.</param>
    /// <param name="shape">The target shape.</param>
    /// <returns>Ndarray.</returns>
    /// <exception cref="System.Exception">Thrown when data length doesn't match shape.</exception>
    let ofArray<'K, 'T> (data: 'T array) (shape: Shape) : Ndarray<'K, 'T> =
        if data.Length <> Shape.numel shape then
            failwith "Data length does not match shape"
        Dense {
            Data = Array.copy data
            Shape = shape
            Strides = stridesC shape
            Offset = 0
            Layout = CLayout
        }

    /// <summary>Convert ndarray to flat array (copy).</summary>
    /// <typeparam name="'K">The phantom type for element kind.</typeparam>
    /// <typeparam name="'T">The element type.</typeparam>
    /// <param name="arr">The input array.</param>
    /// <returns>Flat array copy.</returns>
    let toArray<'K, 'T> (arr: Ndarray<'K, 'T>) : 'T array =
        match arr with
        | Dense d -> Array.copy d.Data
        | Sparse _ -> failwith "toArray not implemented for sparse arrays"

    /// <summary>Map function over all elements.</summary>
    /// <typeparam name="'K">The phantom type for element kind.</typeparam>
    /// <typeparam name="'T">The input element type.</typeparam>
    /// <typeparam name="'U">The output element type.</typeparam>
    /// <param name="f">Mapping function.</param>
    /// <param name="arr">Input array.</param>
    /// <returns>Mapped array.</returns>
    let map<'K, 'T, 'U> (f: 'T -> 'U) (arr: Ndarray<'K, 'T>) : Ndarray<'K, 'U> =
        match arr with
        | Dense d ->
            let newData = Array.map f d.Data
            Dense { d with Data = newData }
        | Sparse _ -> failwith "map not implemented for sparse arrays"

    /// <summary>Generate linearly spaced values.</summary>
    /// <param name="start">Start value.</param>
    /// <param name="stop">Stop value.</param>
    /// <param name="num">Number of points.</param>
    /// <returns>1D array with linearly spaced values.</returns>
    let linspace (start: float) (stop: float) (num: int) : Ndarray<Float64, float> =
        if num < 2 then failwith "linspace requires num >= 2"
        let step = (stop - start) / float (num - 1)
        let data = Array.init num (fun i -> start + float i * step)
        ofArray data [|num|]

    /// <summary>Generate values with given step.</summary>
    /// <param name="start">Start value.</param>
    /// <param name="stop">Stop value (exclusive).</param>
    /// <param name="step">Step size.</param>
    /// <returns>1D array with values from start to stop.</returns>
    let arange (start: float) (stop: float) (step: float) : Ndarray<Float64, float> =
        if step = 0.0 then failwith "arange: step cannot be zero"
        let n = int ((stop - start) / step)
        let data = Array.init n (fun i -> start + float i * step)
        ofArray data [|n|]
