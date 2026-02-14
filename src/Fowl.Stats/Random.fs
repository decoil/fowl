module Fowl.Stats.Random

open System

/// Thread-safe random number generator
module private SafeRandom =
    let private globalLock = obj()
    let private seedGenerator = Random()
    
    /// Create a new Random instance with a unique seed
    let create() =
        lock globalLock (fun () -
003e
            let seed = seedGenerator.Next()
            Random(seed))

/// Random state for functional random number generation
type RandomState = {
    Seed: int
    Internal: Random
}

/// Create a new random state with optional seed
let init (?seed: int) : RandomState =
    let rng = 
        match seed with
        | Some s -> Random(s)
        | None -> SafeRandom.create()
    { Seed = seed |u003e Option.defaultValue 0; Internal = rng }

/// Generate a random float in [0, 1)
let nextFloat (state: RandomState) : float * RandomState =
    let value = state.Internal.NextDouble()
    value, state

/// Generate a random int in range
let nextInt (min: int) (max: int) (state: RandomState) : int * RandomState =
    let value = state.Internal.Next(min, max)
    value, state

/// Generate array of random floats
let nextFloats (count: int) (state: RandomState) : float array * RandomState =
    let values = Array.init count (fun _ -> state.Internal.NextDouble())
    values, state

/// Standard normal random (Box-Muller)
let nextStandardNormal (state: RandomState) : float * RandomState =
    let u1 = state.Internal.NextDouble()
    let u2 = state.Internal.NextDouble()
    let z = sqrt (-2.0 * log u1) * cos (2.0 * System.Math.PI * u2)
    z, state

/// Generate array of standard normal values
let nextStandardNormals (count: int) (state: RandomState) : float array * RandomState =
    let values = 
        Array.init count (fun i ->
            if i % 2 = 0 then
                let u1 = state.Internal.NextDouble()
                let u2 = state.Internal.NextDouble()
                let z1 = sqrt (-2.0 * log u1) * cos (2.0 * System.Math.PI * u2)
                let _z2 = sqrt (-2.0 * log u1) * sin (2.0 * System.Math.PI * u2)  // Can cache for efficiency
                z1
            else
                // Reuse previous calculation - need proper implementation
                let u1 = state.Internal.NextDouble()
                let u2 = state.Internal.NextDouble()
                sqrt (-2.0 * log u1) * sin (2.0 * System.Math.PI * u2))
    values, state
