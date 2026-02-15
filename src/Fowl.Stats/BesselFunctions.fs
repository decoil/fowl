namespace Fowl.Stats

open System
open Fowl
open Fowl.Core.Types

/// <summary>
/// Bessel and related special functions.
/// Used in physics, engineering, and signal processing.
/// </summary>
module BesselFunctions =
    
    /// <summary>
    /// Bessel function of the first kind, order 0.
    /// J₀(x) = Σ (-1)^k (x/2)^(2k) / (k!)²
    /// </summary>
    let j0 (x: float) : float =
        // Use polynomial approximation for |x| < 8
        // Asymptotic expansion for |x| >= 8
        
        if abs x < 8.0 then
            // Polynomial approximation
            let y = x * x
            let p1 = 1.0
            let p2 = -0.25 * y
            let p3 = y * y / 64.0
            let p4 = -y * y * y / 2304.0
            p1 + p2 + p3 + p4
        else
            // Asymptotic: sqrt(2/(πx)) * cos(x - π/4)
            let z = 8.0 / abs x
            let xx = x - System.Math.PI / 4.0
            sqrt (2.0 / (System.Math.PI * abs x)) * cos xx
    
    /// <summary>
    /// Bessel function of the first kind, order 1.
    /// J₁(x) = Σ (-1)^k (x/2)^(2k+1) / (k!(k+1)!)
    /// </summary>
    let j1 (x: float) : float =
        if x = 0.0 then 0.0
        elif abs x < 8.0 then
            let y = x * x
            let p1 = 0.5 * x
            let p2 = -x * y / 16.0
            let p3 = x * y * y / 384.0
            p1 + p2 + p3
        else
            let z = 8.0 / abs x
            let xx = x - 3.0 * System.Math.PI / 4.0
            sqrt (2.0 / (System.Math.PI * abs x)) * cos xx
    
    /// <summary>
    /// Bessel function of the first kind, order n (integer).
    /// Uses recurrence relation for n > 1.
    /// </summary>
    let jn (n: int) (x: float) : float =
        if n = 0 then j0 x
        elif n = 1 then j1 x
        elif n < 0 then
            // J_{-n}(x) = (-1)^n J_n(x)
            let sign = if n % 2 = 0 then 1.0 else -1.0
            sign * jn (-n) x
        else
            // Forward recurrence for small n
            // J_{n+1}(x) = (2n/x) J_n(x) - J_{n-1}(x)
            if abs x < 1e-10 then
                0.0
            else
                let mutable jnm1 = j0 x
                let mutable jn = j1 x
                for k = 1 to n - 1 do
                    let jnp1 = (2.0 * float k / x) * jn - jnm1
                    jnm1 <- jn
                    jn <- jnp1
                jn
    
    /// <summary>
    /// Modified Bessel function of the first kind, order 0.
    /// I₀(x) = Σ (x/2)^(2k) / (k!)²
    /// </summary>
    let i0 (x: float) : float =
        let ax = abs x
        if ax < 3.75 then
            // Polynomial approximation
            let y = x / 3.75
            let y2 = y * y
            1.0 + 3.5156229 * y2 + 3.0899424 * y2 * y2 + 1.2067492 * y2 * y2 * y2
        else
            // Asymptotic: exp(x) / sqrt(2πx)
            exp ax / sqrt (2.0 * System.Math.PI * ax)
    
    /// <summary>
    /// Modified Bessel function of the first kind, order 1.
    /// I₁(x) = Σ (x/2)^(2k+1) / (k!(k+1)!)
    /// </summary>
    let i1 (x: float) : float =
        let ax = abs x
        let result =
            if ax < 3.75 then
                let y = x / 3.75
                let y2 = y * y
                x * (0.5 + 0.87890594 * y2 + 0.51498869 * y2 * y2)
            else
                let z = 3.75 / ax
                exp ax / sqrt (2.0 * System.Math.PI * ax) * (1.0 - 0.03988 * z)
        if x < 0.0 then -result else result
    
    /// <summary>
    /// Modified Bessel function of the second kind, order 0.
    /// K₀(x) - decays exponentially.
    /// </summary>
    let k0 (x: float) : FowlResult<float> =
        result {
            if x <= 0.0 then
                return! Error.invalidArgument "K₀ requires x > 0"
            
            if x <= 2.0 then
                // Small argument approximation
                let y = x * x / 4.0
                return -log (x / 2.0) * i0 x - 0.57721566 + 0.42278420 * y
            else
                // Large argument approximation
                return sqrt (System.Math.PI / (2.0 * x)) * exp (-x)
        }
    
    /// <summary>
    /// Modified Bessel function of the second kind, order 1.
    /// K₁(x).
    /// </summary>
    let k1 (x: float) : FowlResult<float> =
        result {
            if x <= 0.0 then
                return! Error.invalidArgument "K₁ requires x > 0"
            
            if x <= 2.0 then
                let y = x * x / 4.0
                return log (x / 2.0) * i1 x + (1.0 / x) * (1.0 + 0.15443144 * y)
            else
                return sqrt (System.Math.PI / (2.0 * x)) * exp (-x) * (1.0 + 3.0 / (8.0 * x))
        }
    
    /// <summary>
    /// Bessel function of the second kind, order 0 (Neumann function).
    /// Y₀(x).
    /// </summary>
    let y0 (x: float) : FowlResult<float> =
        result {
            if x <= 0.0 then
                return! Error.invalidArgument "Y₀ requires x > 0"
            
            if x < 8.0 then
                // Use relation to J₀
                return (j0 x * log (x / 2.0) + 0.36746691) * 2.0 / System.Math.PI
            else
                let z = 8.0 / x
                let xx = x - System.Math.PI / 4.0
                return sqrt (2.0 / (System.Math.PI * x)) * sin xx
        }
    
    /// <summary>
    /// Spherical Bessel function of the first kind, order 0.
    /// j₀(x) = sin(x) / x
    /// </summary>
    let sphericalJ0 (x: float) : float =
        if x = 0.0 then 1.0
        else sin x / x
    
    /// <summary>
    /// Spherical Bessel function of the first kind, order 1.
    /// j₁(x) = sin(x)/x² - cos(x)/x
    /// </summary>
    let sphericalJ1 (x: float) : float =
        if x = 0.0 then 0.0
        else sin x / (x * x) - cos x / x
    
    /// <summary>
    /// Airy function Ai(x) approximation.
    /// Solution to y'' - xy = 0.
    /// </summary>
    let airyAi (x: float) : float =
        if x > 0.0 then
            // Decaying exponential for positive x
            0.35502805 * exp (-2.0 / 3.0 * x ** 1.5)
        else
            // Oscillatory for negative x
            let z = -x
            0.35502805 * sin (2.0 / 3.0 * z ** 1.5 + System.Math.PI / 4.0) / (z ** 0.25)
    
    /// <summary>
    /// Airy function Bi(x) approximation.
    /// </summary>
    let airyBi (x: float) : float =
        if x > 0.0 then
            // Growing exponential for positive x
            0.61492663 * exp (2.0 / 3.0 * x ** 1.5) / (x ** 0.25)
        else
            let z = -x
            0.61492663 * cos (2.0 / 3.0 * z ** 1.5 + System.Math.PI / 4.0) / (z ** 0.25)