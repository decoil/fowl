/// <summary>Fowl Thread-Safe Random Module</summary>
/// <remarks>
/// Provides thread-safe random number generation for parallel operations.
/// 
/// Uses ThreadLocal to ensure each thread has its own Random instance,
/// avoiding contention and correlation issues.
/// 
/// Example:
/// <code>
/// open Fowl.Parallel.Random
/// 
/// // Thread-safe random generation
/// let values = Array.init 1000 (fun _ -> ThreadSafeRandom.nextDouble())
/// </code>
/// </remarks>
module Fowl.Parallel.ThreadSafeRandom

open System
open System.Threading

// ============================================================================
// Thread-Local Random Instances
// ============================================================================

/// <summary>
/// Thread-local Random instances.
/// </summary>
/// <remarks>
/// Each thread gets its own Random instance seeded uniquely.
/// This avoids:
/// 1. Contention (no locking needed)
/// 2. Correlation (different seeds per thread)
/// 3. Global state mutation
/// </remarks>
let private threadRandom = new ThreadLocal<Random>(fun () ->
    // Seed with thread-specific value + current time
    let seed = Thread.CurrentThread.ManagedThreadId + int (DateTime.Now.Ticks &&& 0xFFFFFFFFL)
    Random(seed))

/// <summary>
/// Get thread-local Random instance.
/// </summary>
let private getRandom () = threadRandom.Value

// ============================================================================
// Double Generation
// ============================================================================

/// <summary>
/// Generate random double in [0.0, 1.0).
/// </summary>
/// <returns>Random double.</returns>
let nextDouble () : double =
    getRandom().NextDouble()

/// <summary>
/// Generate random double in range [min, max).
/// </summary>
/// <param name="min">Minimum value (inclusive).</param>
/// <param name="max">Maximum value (exclusive).</param>
/// <returns>Random double in range.</returns>
let nextDoubleRange (min: double) (max: double) : double =
    min + (getRandom().NextDouble() * (max - min))

// ============================================================================
// Integer Generation
// ============================================================================

/// <summary>
/// Generate random int.
/// </summary>
/// <returns>Random int.</returns>
let nextInt () : int =
    getRandom().Next()

/// <summary>
/// Generate random int in range [0, max).
/// </summary>
/// <param name="max">Maximum value (exclusive).</param>
/// <returns>Random int.</returns>
let nextIntMax (max: int) : int =
    getRandom().Next(max)

/// <summary>
/// Generate random int in range [min, max).
/// </summary>
/// <param name="min">Minimum value (inclusive).</param>
/// <param name="max">Maximum value (exclusive).</param>
/// <returns>Random int.</returns>
let nextIntRange (min: int) (max: int) : int =
    getRandom().Next(min, max)

// ============================================================================
// Array Generation
// ============================================================================

/// <summary>
/// Generate array of random doubles in parallel.
/// </summary>
/// <param name="length">Array length.</param>
/// <returns>Array of random doubles in [0, 1).</returns>
/// <remarks>
/// Uses thread-safe random generation in parallel.
/// </remarks>
let nextDoubleArray (length: int) : double[] =
    let result = Array.zeroCreate length
    System.Threading.Tasks.Parallel.For(0, length, fun i ->
        result.[i] <- nextDouble()
    ) |> ignore
    result

/// <summary>
/// Generate array of random doubles with custom range in parallel.
/// </summary>
/// <param name="length">Array length.</param>
/// <param name="min">Minimum value.</param>
/// <param name="max">Maximum value.</param>
/// <returns>Array of random doubles.</returns>
let nextDoubleArrayRange (length: int) (min: double) (max: double) : double[] =
    let result = Array.zeroCreate length
    System.Threading.Tasks.Parallel.For(0, length, fun i ->
        result.[i] <- nextDoubleRange min max
    ) |> ignore
    result

/// <summary>
/// Generate array of standard normal random numbers (Box-Muller).
/// </summary>
/// <param name="length">Array length.</param>
/// <returns>Array of N(0,1) random values.</returns>
/// <remarks>
/// Uses Box-Muller transform for normal distribution.
/// Thread-safe for parallel generation.
/// </remarks>
let nextNormalArray (length: int) : double[] =
    let result = Array.zeroCreate length
    
    // Process pairs for Box-Muller
    System.Threading.Tasks.Parallel.For(0, (length + 1) / 2, fun pairIdx ->
        let i = pairIdx * 2
        if i < length then
            let u1 = 1.0 - nextDouble()  // Avoid 0
            let u2 = nextDouble()
            let radius = sqrt (-2.0 * log u1)
            let theta = 2.0 * Math.PI * u2
            result.[i] <- radius * cos theta
            if i + 1 < length then
                result.[i + 1] <- radius * sin theta
    ) |> ignore
    
    result

// ============================================================================
// Shuffling
// ============================================================================

/// <summary>
/// Shuffle array in-place using Fisher-Yates algorithm.
/// </summary>
/// <param name="array">Array to shuffle.</param>
/// <remarks>
/// Not thread-safe for concurrent modifications.
/// Each shuffle call uses thread-local random.
/// </remarks>
let shuffleInPlace (array: 'T[]) : unit =
    let rng = getRandom()
    let n = array.Length
    for i = n - 1 downto 1 do
        let j = rng.Next(i + 1)
        let temp = array.[i]
        array.[i] <- array.[j]
        array.[j] <- temp

/// <summary>
/// Return shuffled copy of array.
/// </summary>
/// <param name="array">Source array.</param>
/// <returns>Shuffled copy.</returns>
let shuffle (array: 'T[]) : 'T[] =
    let copy = Array.copy array
    shuffleInPlace copy
    copy

// ============================================================================
// Sampling
// ============================================================================

/// <summary>
/// Sample k elements from array without replacement.
/// </summary>
/// <param name="array">Source array.</param>
/// <param name="k">Number of samples.</param>
/// <returns>Array of k sampled elements.</returns>
/// <exception cref="System.ArgumentException">Thrown when k > array length.</exception>
let sampleWithoutReplacement (array: 'T[]) (k: int) : 'T[] =
    if k > array.Length then
        invalidArg "k" "Sample size cannot exceed array length"
    
    // Fisher-Yates shuffle first k elements
    let result = Array.copy array
    let rng = getRandom()
    
    for i = 0 to k - 1 do
        let j = rng.Next(i, array.Length)
        let temp = result.[i]
        result.[i] <- result.[j]
        result.[j] <- temp
    
    // Return first k elements
    Array.sub result 0 k

/// <summary>
/// Sample k elements from array with replacement.
/// </summary>
/// <param name="array">Source array.</param>
/// <param name="k">Number of samples.</param>
/// <returns>Array of k sampled elements.</returns>
let sampleWithReplacement (array: 'T[]) (k: int) : 'T[] =
    let n = array.Length
    Array.init k (fun _ -> array.[getRandom().Next(n)])

// ============================================================================
// Probability
// ============================================================================

/// <summary>
/// Generate true with given probability.
/// </summary>
/// <param name="probability">Probability of true (0.0 to 1.0).</param>
/// <returns>true with given probability.</returns>
let nextBool (probability: double) : bool =
    getRandom().NextDouble() < probability

/// <summary>
/// Generate Bernoulli trial outcome.
/// </summary>
/// <param name="p">Success probability.</param>
/// <returns>1 with probability p, 0 otherwise.</returns>
let bernoulli (p: double) : int =
    if nextBool p then 1 else 0

// ============================================================================
// Distributions
// ============================================================================

/// <summary>
/// Generate from exponential distribution.
/// </summary>
/// <param name="lambda">Rate parameter (1/mean).</param>
/// <returns>Random value from Exp(lambda).</returns>
let exponential (lambda: double) : double =
    -log (1.0 - nextDouble()) / lambda

/// <summary>
/// Generate from uniform distribution.
/// </summary>
/// <param name="min">Minimum value.</param>
/// <param name="max">Maximum value.</param>
/// <returns>Random value from U(min, max).</returns>
let uniform (min: double) (max: double) : double =
    nextDoubleRange min max

// ============================================================================
// Seeding (Advanced)
// ============================================================================

/// <summary>
/// Create new Random with specific seed for this thread.
/// </summary>
/// <param name="seed">Random seed.</param>
/// <remarks>
/// Replaces the thread-local Random with a new one using the specified seed.
/// Use with caution - can cause correlation if seeds are not well-distributed.
/// </remarks>
let reseed (seed: int) : unit =
    threadRandom.Value <- Random(seed)

/// <summary>
/// Reset thread-local Random to default (time-based seeding).
/// </summary>
let reset () : unit =
    let seed = Thread.CurrentThread.ManagedThreadId + int (DateTime.Now.Ticks &&& 0xFFFFFFFFL)
    threadRandom.Value <- Random(seed)
