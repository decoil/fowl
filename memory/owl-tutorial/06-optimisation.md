# Owl Tutorial Chapter 6: Optimisation

## Problem Types
- **Unconstrained vs Constrained** - with/without constraints on variables
- **Local vs Global** - find local minimum vs global minimum
- **Linear vs Nonlinear** - linear/convex functions vs general functions

## Root Finding Methods

### Bisection
- Simple, reliable, slow convergence (~50 iterations for 14 digits)
- Works for continuous functions

### Newton's Method
- Uses derivatives: `x_{n+1} = x_n - f(x_n)/f'(x_n)`
- Quadratic convergence (very fast)
- Requires smooth functions

### Brent's Method
- Combines Bisection, Secant, and Inverse Quadratic Interpolation
- Generally considered the best root-finding routine

## Univariate Optimisation

### Using Derivatives
- Find where `f'(x) = 0` using root finding
- Can't distinguish max/min without second derivative test

### Golden Section Search
- No derivatives required
- Uses triplet [a, b, c] and reduces range
- Guaranteed 0.618x reduction per iteration

## Multivariate Optimisation

### Nelder-Mead Simplex
- Gradient-free method
- Simplex moves downhill via reflection, contraction, shrinkage
- Robust but slow

### Gradient Descent
```ocaml
x_{n+1} = x_n - α * ∇f(x_n)
```
- Uses gradient direction (steepest descent)
- Learning rate α controls step size

### Conjugate Gradient
- Avoids bouncing in narrow valleys
- Uses conjugate directions instead of pure gradient
- More efficient than plain gradient descent

### Newton's Method (Multivariate)
```ocaml
x_{n+1} = x_n - α * H^{-1} * ∇f(x_n)
```
- Uses Hessian matrix H (second derivatives)
- Quadratic convergence
- Problem: Hessian is expensive to compute/invert

### Quasi-Newton (BFGS, L-BFGS)
- Approximates inverse Hessian iteratively
- L-BFGS: memory-efficient version for large problems

## Owl API

```ocaml
(* Low-level: minimise_fun *)
Owl_optimise.D.minimise_fun params objective_function init_value

(* Gradient methods available *)
Owl_optimise.D.Gradient.GD   (* Gradient Descent *)
Owl_optimise.D.Gradient.CG   (* Conjugate Gradient *)
Owl_optimise.D.Gradient.Newton

(* Learning rate schedules *)
Learning_Rate.Const
Learning_Rate.Decay
Learning_Rate.Adagrad
Learning_Rate.Adam
```

## F# Design Notes

### Optimisation Module Structure
```fsharp
namespace Fowl

module Optimise =
    type Params = {
        GradientMethod: GradientMethod
        LearningRate: LearningRate
        BatchSize: int
        Epochs: float
        Regularisation: Regularisation
    }

    val minimise: Params -> (AD -> AD) -> NDArray -> NDArray
```

---
_Learned: 2026-02-13_
