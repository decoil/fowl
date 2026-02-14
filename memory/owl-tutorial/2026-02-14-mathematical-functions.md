# Owl Tutorial - Mathematical Functions (Chapter Notes)

**Date:** 2026-02-14  
**Chapter:** Mathematical Functions  
**Source:** https://ocaml.xyz/tutorial/chapters/maths.html

## Overview

This chapter covers scalar mathematical functions in Owl. Functions work on float values - N-dimensional array versions covered in later chapters.

## Basic Functions

### Unary Functions
Standard scalar operations: `abs`, `neg`, `reci`, `floor`, `ceil`, `round`, `trunc`, `sqr`, `sqrt`

### Binary Functions
Arithmetic: `add`, `sub`, `mul`, `div`, `fmod`, `pow`, `hypot`, `atan2`

### Exponential & Logarithmic
Key functions with numerical stability variants:
- `exp`, `exp2`, `exp10`, `expm1` (exp(x)-1, accurate for x~0)
- `log`, `log2`, `log10`, `logn`, `log1p` (inverse of expm1)
- Statistical: `logit`, `expit`, `log1mexp`, `log1pexp`

### Trigonometric Functions
Standard: `sin`, `cos`, `tan`, `cot`, `sec`, `csc`  
Inverses: `asin`, `acos`, `atan`, `acot`, `asec`, `acsc`  
Hyperbolic: `sinh`, `cosh`, `tanh`, `coth`, `sech`, `csch`  
Special: `sinc`, `logsinh`, `logcosh`, `sindg`, `cosdg`, `tandg`, `cotdg`

### Neural Network Functions
- `sigmoid`: 1/(1+exp(-x))
- `relu`: max(0, x)
- `signum`, `softsign`

## Special Functions (via Cephes Library)

### Airy Functions
Solution to y''(x) = xy(x). Returns (Ai, Ai', Bi, Bi') for two linearly independent solutions.

### Bessel Functions
First kind (j0, j1, jv - non-singular at origin)  
Second kind (y0, y1, yv, yn - singular at origin)  
Modified versions (i0, i1, iv, k0, k1)  
Exponentially scaled variants (i0e, i1e, k0e, k1e)

### Elliptic Functions
Jacobian: `ellipj u m` → returns (sn, cn, dn, phi)  
Integrals: `ellipk` (complete 1st kind), `ellipe` (complete 2nd kind), `ellipkinc`, `ellipeinc`

### Gamma Functions
- `gamma z`: Γ(z) = (z-1)! for integers
- `rgamma`: reciprocal
- `loggamma`: log(Γ(z))
- `gammainc`, `gammaincc`: incomplete variants
- `psi`: digamma function

### Beta Functions
B(x,y) = Γ(x)Γ(y)/Γ(x+y)  
- `beta x y`: complete beta
- `betainc a b x`: incomplete beta
- `betaincinv`: inverse of betainc

### Struve Functions
`struve v x`: order v, used in physics (water waves, aerodynamics)

### Zeta Functions
- `zeta x q`: Hurwitz zeta function Σ(k+q)^(-x)
- `zetac x`: Riemann zeta function minus 1
- When q=1, reduces to Riemann zeta function

### Error Functions
`erf x`: (2/√π)∫₀ˣ e^(-t²)dt - used in probability/statistics

## Key Insights for Fowl

1. **Numerical Stability**: Functions like `expm1`, `log1p` handle edge cases where naive implementations lose precision
2. **Cephes Dependency**: Owl uses Cephes C library for special functions - need F# binding strategy
3. **API Pattern**: Simple float → float (unary) or float → float → float (binary) functions
4. **Category Organization**: Grouped by mathematical domain (trig, special functions, etc.)
5. **Statistical Functions**: Many functions (logit, expit) bridge math and ML domains

## F# Mapping Considerations

- Most scalar functions map directly to `System.Math` or `MathNet.Numerics`
- Special functions (Bessel, Gamma, etc.) may need native library bindings
- Consider computation expressions for chaining mathematical operations
- Pattern: `Maths.function_name` in Owl → `Math.functionName` or module organization in F#

---

_Next: Continue with Statistical Functions chapter_
