module Fowl.Stats.SpecialFunctions

open System

/// Error function approximation (Abramowitz and Stegun)
let erf (x: float) : float =
    let sign = if x < 0.0 then -1.0 else 1.0
    let x = abs x
    
    // Constants
    let a1 =  0.254829592
    let a2 = -0.284496736
    let a3 =  1.421413741
    let a4 = -1.453152027
    let a5 =  1.061405429
    let p  =  0.3275911
    
    let t = 1.0 / (1.0 + p * x)
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp (-x * x)
    
    sign * y

/// Complementary error function
let erfc (x: float) : float = 1.0 - erf x

/// Inverse complementary error function (approximation)
let erfcinv (p: float) : FowlResult<float> =
    if p < 0.0 || p > 2.0 then
        Error.invalidArgument "p must be in [0, 2]"
    else
        // Handle edge cases
        if p = 0.0 then
            Ok infinity
        elif p = 2.0 then
            Ok -infinity
        elif p = 1.0 then
            Ok 0.0
        else
            // Approximation for inverse erfc
            let pp = if p <= 1.0 then p else 2.0 - p
            let t = sqrt (-2.0 * log (pp / 2.0))

            let c0 = 2.515517
            let c1 = 0.802853
            let c2 = 0.010328
            let d1 = 1.432788
            let d2 = 0.189269
            let d3 = 0.001308

            let x = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
            Ok (if p <= 1.0 then x else -x)

/// Gamma function approximation (Lanczos approximation)
let gamma (z: float) : FowlResult<float> =
    if z <= 0.0 then
        Error.invalidArgument "gamma not defined for non-positive integers"
    else
        // Use reflection formula for small values
        if z < 0.5 then
            result {
                let! gammaInv = gamma (1.0 - z)
                return Math.PI / (sin (Math.PI * z) * gammaInv)
            }
        else
            let g = 7.0
            let coefficients = [|
                0.99999999999980993
                676.5203681218851
                -1259.1392167224028
                771.32342877765313
                -176.61502916214059
                12.507343278686905
                -0.13857109526572012
                9.9843695780195716e-6
                1.5056327351493116e-7
            |]

            let z = z - 1.0
            let x = ref coefficients.[0]
            for i = 1 to 8 do
                x := !x + coefficients.[i] / (z + float i)

            let t = z + g + 0.5
            Ok (sqrt (2.0 * Math.PI) * (t ** (z + 0.5)) * exp (-t) * !x)

/// Log gamma function
let logGamma (z: float) : float =
    log (gamma z)

/// Standard normal random number (Box-Muller)
let randn (rng: Random) : float =
    let u1 = rng.NextDouble()
    let u2 = rng.NextDouble()
    sqrt (-2.0 * log u1) * cos (2.0 * Math.PI * u2)

// ============================================================================
// Beta Functions (for Beta distribution)
// ============================================================================

/// Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)
let beta (a: float) (b: float) : FowlResult<float> =
    if a <= 0.0 || b <= 0.0 then
        Error.invalidArgument "Beta function requires a > 0 and b > 0"
    else
        result {
            let! ga = gamma a
            let! gb = gamma b
            let! gab = gamma (a + b)
            return ga * gb / gab
        }

/// Log beta function - more numerically stable
let logBeta (a: float) (b: float) : FowlResult<float> =
    if a <= 0.0 || b <= 0.0 then
        Error.invalidArgument "Log beta function requires a > 0 and b > 0"
    else
        result {
            let! lga = logGamma a |> Ok
            let! lgb = logGamma b |> Ok
            let! lgab = logGamma (a + b) |> Ok
            return lga + lgb - lgab
        }

/// Incomplete beta function I_x(a,b) using continued fraction
/// This is the regularized incomplete beta function
let incompleteBeta (a: float) (b: float) (x: float) : FowlResult<float> =
    if a <= 0.0 || b <= 0.0 then
        Error.invalidArgument "Incomplete beta requires a > 0 and b > 0"
    elif x < 0.0 || x > 1.0 then
        Error.invalidArgument "x must be in [0, 1]"
    elif x = 0.0 then
        Ok 0.0
    elif x = 1.0 then
        Ok 1.0
    else
        // Use symmetry relation if x > (a+1)/(a+b+2)
        let x', a', b' = 
            if x > (a + 1.0) / (a + b + 2.0) then
                (1.0 - x, b, a)
            else
                (x, a, b)
        
        // Lentz's algorithm for continued fraction
        let maxIterations = 200
        let epsilon = 1e-14
        
        let getA n = 
            if n = 0 then 1.0
            elif n % 2 = 1 then float (n + 1) / 2.0 * (b' - float n) / ((a' + float n - 1.0) * (a' + float n))
            else float n / 2.0 * (a' + b' - float n) / ((a' + float n - 1.0) * (a' + float n))
        
        let getB n = 1.0
        
        let rec lentz f0 c d n =
            if n > maxIterations then
                f0
            else
                let an = getA n
                let bn = getB n
                let dn = bn + an / d
                let dn' = if abs dn < epsilon then epsilon else dn
                let cn = bn + an / c
                let cn' = if abs cn < epsilon then epsilon else cn
                let delta = cn' / dn'
                let fn = f0 * delta
                if abs (delta - 1.0) < epsilon then
                    fn
                else
                    lentz fn cn' dn' (n + 1)
        
        result {
            let! lbeta = logBeta a' b'
            let front = exp (a' * log x' + b' * log (1.0 - x') - lbeta) / a'
            let cf = lentz 1.0 1.0 1.0 1
            return front * cf
        }
