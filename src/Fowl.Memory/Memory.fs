/// <summary>Fowl Memory Optimization Module - Span-based Operations</summary>/// <remarks>
/// Provides zero-copy operations using System.Span<T> and ArrayPool<T>.
/// 
/// Key benefits:
/// - Zero-copy slicing: Views into existing memory without allocation
/// - Reduced GC pressure: Array pooling for temporary buffers
/// - Unified API: Work with arrays, slices, and memory uniformly
/// 
/// Example usage:
/// <code>
/// open Fowl.Memory
/// 
/// // Zero-copy slice (no allocation)
/// let slice = SpanOps.slice arr 10 100
/// 
/// // Rent temporary buffer from pool
/// use buffer = ArrayPoolOps.rentDouble 1000
/// // ... use buffer.Array ...
/// // Automatically returned to pool
/// </code>
/// </remarks>
module Fowl.Memory

open System
open System.Buffers

// ============================================================================
// Span-based Array Operations
// ============================================================================

/// <summary>Module for Span<T>-based array operations.</summary>module SpanOps =
    /// <summary>Create a Span view of an entire array.</summary>    /// <param name="array">Source array.</param>    /// <returns>Span view of the array.</returns>    let inline ofArray (array: 'T[]) : Span<'T> =
        Span(array)

    /// <summary>Create a ReadOnlySpan view of an entire array.</summary>    /// <param name="array">Source array.</param>    /// <returns>ReadOnlySpan view of the array.</returns>    let inline ofArrayReadOnly (array: 'T[]) : ReadOnlySpan<'T> =
        ReadOnlySpan(array)

    /// <summary>Create a slice view without copying.</summary>    /// <param name="array">Source array.</param>    /// <param name="start">Start index.</param>
    /// <param name="length">Length of slice.</param>
    /// <returns>Span view into array.</returns>
    /// <example>
    /// <code>
    /// let arr = [|1; 2; 3; 4; 5|]
    /// let slice = SpanOps.slice arr 1 3  // View of [|2; 3; 4|]
    /// slice.[0] <- 99  // Modifies arr.[1]!
    /// </code>
    /// </example>
    let inline slice (array: 'T[]) (start: int) (length: int) : Span<'T> =
        if start < 0 || start >= array.Length then
            raise (ArgumentOutOfRangeException(nameof(start)))
        if length < 0 || start + length > array.Length then
            raise (ArgumentOutOfRangeException(nameof(length)))
        Span(array, start, length)

    /// <summary>Create a read-only slice view.</summary>    /// <param name="array">Source array.</param>    /// <param name="start">Start index.</param>    /// <param name="length">Length of slice.</param>    /// <returns>ReadOnlySpan view into array.</returns>    let inline sliceReadOnly (array: 'T[]) (start: int) (length: int) : ReadOnlySpan<'T> =
        if start < 0 || start >= array.Length then
            raise (ArgumentOutOfRangeException(nameof(start)))
        if length < 0 || start + length > array.Length then
            raise (ArgumentOutOfRangeException(nameof(length)))
        ReadOnlySpan(array, start, length)

    /// <summary>Copy span to new array.</summary>    /// <param name="span">Source span.</param>
    /// <returns>New array with copied data.</returns>    let inline toArray (span: Span<'T>) : 'T[] =
        span.ToArray()

    /// <summary>Copy ReadOnlySpan to new array.</summary>    /// <param name="span">Source span.</param>    /// <returns>New array with copied data.</returns>    let inline toArrayReadOnly (span: ReadOnlySpan<'T>) : 'T[] =
        span.ToArray()

    /// <summary>Fill span with a value.</summary>    /// <param name="span">Span to fill.</param>    /// <param name="value">Value to fill with.</param>    let inline fill (span: Span<'T>) (value: 'T) : unit =
        span.Fill(value)

    /// <summary>Clear span (set to default value).</summary>    /// <param name="span">Span to clear.</param>    let inline clear (span: Span<'T>) : unit =
        span.Clear()

    /// <summary>Copy from source span to destination span.</summary>    /// <param name="source">Source span.</param>    /// <param name="destination">Destination span.</param>    let inline copyTo (source: ReadOnlySpan<'T>) (destination: Span<'T>) : unit =
        source.CopyTo(destination)

    /// <summary>Get length of span.</summary>    /// <param name="span">Input span.</param>    /// <returns>Length.</returns>
    let inline length (span: Span<'T>) : int =
        span.Length

// ============================================================================
// ArrayPool Operations
// ============================================================================

/// <summary>Disposable wrapper for rented array that returns to pool on dispose.</summary>type PooledArray<'T> =
    { Array: 'T[]
      Pool: ArrayPool<'T>
      mutable IsReturned: bool }
    
    interface IDisposable with
        member this.Dispose() =
            if not this.IsReturned then
                this.Pool.Return(this.Array)
                this.IsReturned <- true

/// <summary>Module for ArrayPool-based memory management.</summary>module ArrayPoolOps =
    /// <summary>Shared array pool instance.</summary>    let shared<'T> = ArrayPool<'T>.Shared

    /// <summary>Rent an array from the shared pool.</summary>    /// <param name="size">Minimum size needed.</param>    /// <returns>PooledArray that auto-returns on dispose.</returns>    /// <example>
    /// <code>
    /// use buffer = ArrayPoolOps.rentDouble 1000
    /// // Use buffer.Array
    /// // Automatically returned to pool at end of scope
    /// </code>
    /// </example>    let rentDouble (size: int) : PooledArray<double> =
        let pool = shared<double>
        let arr = pool.Rent(size)
        { Array = arr; Pool = pool; IsReturned = false }

    /// <summary>Rent a float array from the shared pool.</summary>    /// <param name="size">Minimum size needed.</param>    /// <returns>PooledArray that auto-returns on dispose.</returns>    let rentSingle (size: int) : PooledArray<single> =
        let pool = shared<single>
        let arr = pool.Rent(size)
        { Array = arr; Pool = pool; IsReturned = false }

    /// <summary>Rent an int array from the shared pool.</summary>    /// <param name="size">Minimum size needed.</param>    /// <returns>PooledArray that auto-returns on dispose.</returns>    let rentInt (size: int) : PooledArray<int> =
        let pool = shared<int>
        let arr = pool.Rent(size)
        { Array = arr; Pool = pool; IsReturned = false }

    /// <summary>Rent a generic array from the shared pool.</summary>    /// <param name="size">Minimum size needed.</param>    /// <returns>PooledArray that auto-returns on dispose.</returns>    let rent<'T> (size: int) : PooledArray<'T> =
        let pool = shared<'T>
        let arr = pool.Rent(size)
        { Array = arr; Pool = pool; IsReturned = false }

    /// <summary>Return array to pool manually (if not using 'use').</summary>    /// <param name="pooled">PooledArray to return.</param>    let returnToPool<'T> (pooled: PooledArray<'T>) : unit =
        (pooled :> IDisposable).Dispose()

    /// <summary>Return array to pool with optional clearing.</summary>    /// <param name="array">Array to return.</param>    /// <param name="clearArray">Whether to clear array before returning.</param>
    let returnArray<'T> (array: 'T[]) (clearArray: bool) : unit =
        shared<'T>.Return(array, clearArray)

// ============================================================================
// Stack Allocation (for small buffers)
// ============================================================================

/// <summary>Module for stack-allocated Span operations.</summary>
/// <remarks>
/// Stack allocation is very fast but limited in size.
/// Use for small temporary buffers (typically < 1KB).
/// </remarks>
module StackOps =
    /// <summary>Execute operation with stack-allocated buffer.</summary>    /// <param name="size">Size of stack buffer.</param>    /// <param name="operation">Function using the buffer.</param>
    /// <returns>Result of operation.</returns>
    /// <example>
    /// <code>
    /// let result = StackOps.withBuffer 10 (fun span ->
    ///     span.[0] <- 1.0
    ///     span.[1] <- 2.0
    ///     span.[0] + span.[1]
    /// )
    /// // result = 3.0
    /// </code>
    /// </example>    let inline withBuffer<'T, 'Result> (size: int) (operation: Span<'T> -> 'Result) : 'Result =
        let mutable buffer = stackalloc<'T> size
        operation(Span(buffer))

    /// <summary>Execute operation with stack-allocated byte buffer.</summary>    /// <param name="size">Size in bytes.</param>
    /// <param name="operation">Function using the buffer.</param>
    /// <returns>Result of operation.</returns>
    let inline withByteBuffer<'Result> (size: int) (operation: Span<byte> -> 'Result) : 'Result =
        let mutable buffer = stackalloc<byte> size
        operation(Span(buffer))

// ============================================================================
// Memory-efficient Operations
// ============================================================================

/// <summary>Module for memory-efficient array operations.</summary>module EfficientOps =
    /// <summary>Map operation with pooled result buffer.</summary>    /// <param name="mapping">Mapping function.</param>    /// <param name="source">Source array.</param>
    /// <returns>New array with mapped values.</returns>
    let mapPooled<'T, 'U> (mapping: 'T -> 'U) (source: 'T[]) : 'U[] =
        use pooled = ArrayPoolOps.rent<'U> source.Length
        let buffer = pooled.Array
        
        for i = 0 to source.Length - 1 do
            buffer.[i] <- mapping source.[i]
        
        // Copy to appropriately-sized array
        let result = Array.zeroCreate source.Length
        Span(buffer, 0, source.Length).CopyTo(Span(result))
        result

    /// <summary>Map operation in-place (no allocation for result).</summary>    /// <param name="mapping">Mapping function.</param>    /// <param name="array">Array to modify in place.</param>
    let mapInPlace<'T> (mapping: 'T -> 'T) (array: 'T[]) : unit =
        let span = Span(array)
        for i = 0 to span.Length - 1 do
            span.[i] <- mapping span.[i]

    /// <summary>Zip two arrays with pooled result buffer.</summary>    /// <param name="mapping">Zip function.</param>    /// <param name="a">First array.</param>
    /// <param name="b">Second array.</param>    /// <returns>New array with zipped values.</returns>
    let zipPooled<'T1, 'T2, 'U> (mapping: 'T1 -> 'T2 -> 'U) (a: 'T1[]) (b: 'T2[]) : 'U[] =
        if a.Length <> b.Length then
            invalidArg "b" "Arrays must have same length"
        
        use pooled = ArrayPoolOps.rent<'U> a.Length
        let buffer = pooled.Array
        
        for i = 0 to a.Length - 1 do
            buffer.[i] <- mapping a.[i] b.[i]
        
        let result = Array.zeroCreate a.Length
        Span(buffer, 0, a.Length).CopyTo(Span(result))
        result

    /// <summary>Fold operation using spans.</summary>    /// <param name="folder">Folder function.</param>    /// <param name="state">Initial state.</param>
    /// <param name="array">Source array.</param>    /// <returns>Final state.</returns>
    let foldSpan<'T, 'State> (folder: 'State -> 'T -> 'State) (state: 'State) (array: 'T[]) : 'State =
        let span = ReadOnlySpan(array)
        let mutable acc = state
        for i = 0 to span.Length - 1 do
            acc <- folder acc span.[i]
        acc

// ============================================================================
// Memory Diagnostics
// ============================================================================

/// <summary>Module for memory diagnostics and monitoring.</summary>module MemoryDiagnostics =
    /// <summary>Get current memory usage information.</summary>    /// <returns>Tuple of (total memory, working set).</returns>
    let getMemoryInfo () : (int64 * int64) =
        let proc = Diagnostics.Process.GetCurrentProcess()
        proc.Refresh()
        (GC.GetTotalMemory(false), proc.WorkingSet64)

    /// <summary>Format memory size for display.</summary>    /// <param name="bytes">Size in bytes.</param>
    /// <returns>Formatted string.</returns>
    let formatBytes (bytes: int64) : string =
        if bytes < 1024L then
            sprintf "%d B" bytes
        elif bytes < 1024L * 1024L then
            sprintf "%.2f KB" (float bytes / 1024.0)
        elif bytes < 1024L * 1024L * 1024L then
            sprintf "%.2f MB" (float bytes / (1024.0 * 1024.0))
        else
            sprintf "%.2f GB" (float bytes / (1024.0 * 1024.0 * 1024.0))

    /// <summary>Print current memory usage.</summary>    let printMemoryUsage () : unit =
        let total, working = getMemoryInfo()
        printfn "Memory Usage:"
        printfn "  GC Total:    %s" (formatBytes total)
        printfn "  Working Set: %s" (formatBytes working)

    /// <summary>Force garbage collection and wait for completion.</summary>    let forceGC () : unit =
        GC.Collect()
        GC.WaitForPendingFinalizers()
        GC.Collect()

    /// <summary>Measure memory allocated by an operation.</summary>    /// <param name="operation">Operation to measure.</param>    /// <returns>Result and bytes allocated.</returns>
    let measureAllocation<'T> (operation: unit -> 'T) : 'T * int64 =
        let before = GC.GetTotalMemory(true)
        let result = operation()
        let after = GC.GetTotalMemory(false)
        result, (after - before)
