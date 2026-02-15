# Tutorial 7: Algorithmic Differentiation

Compute derivatives automatically using Fowl's forward and reverse mode AD.

## Overview

Algorithmic differentiation (AD) computes exact derivatives of functions, essential for gradient-based optimization, backpropagation, and sensitivity analysis. Fowl provides both forward and reverse mode AD.

## Learning Objectives

- Understand forward mode AD with Dual numbers
- Use reverse mode AD with computation graphs
- Compute gradients, Jacobians, and Hessians
- Apply AD to machine learning
- Understand performance trade-offs

## Setup

```fsharp
open Fowl
open Fowl.Core.Types
open Fowl.AD
open Fowl.Neural.Graph

let unwrap = function Ok x -> x | Error e -> failwith e.Message
```

## Forward Mode AD

### Dual Numbers

```fsharp
// Dual number: x + ε * dx
// Operations preserve both value and derivative

// Create dual number
let d1 = Dual.init [|1.0; 0.0|]  // x = 1.0, dx/dx = 1.0
let d2 = Dual.fromFloat 2.0       // x = 2.0, dx/dx = 0.0 (constant)

printfn "Dual 1: %A" d1  // Dual(value=1.0, tangent=[|1.0|])
printfn "Dual 2: %A" d2  // Dual(value=2.0, tangent=[|0.0|])
```

### Basic Operations

```fsharp
// Addition
let sum = Dual.add d1 d2
printfn "(1.0 + 2.0)' = %.1f" sum.Tangent.[0]  // 1.0 (derivative)

// Multiplication
let product = Dual.mul d1 d2
printfn "(1.0 * 2.0)' = %.1f" product.Tangent.[0]  // 2.0

// Power
let power = Dual.pow d1 3.0
printfn "(1.0^3)' = %.1f" power.Tangent.[0]  // 3.0

// Elementary functions
let sinResult = Dual.sin (Dual.fromFloat System.Math.PI / 2.0)
printfn "sin(π/2)' = %.1f" sinResult.Tangent.[0]  // 0.0

let expResult = Dual.exp (Dual.fromFloat 1.0)
printfn "exp(1)' = %.1f" expResult.Tangent.[0]  // 2.718...
```

### Computing Derivatives

```fsharp
// Simple function: f(x) = x²
let f (x: Dual) = Dual.pow x 2.0

// Compute derivative at x = 3
let d3 = Dual.init [|3.0; 1.0|]
let f3 = f d3
printfn "f(3) = %f, f'(3) = %f" f3.Value f3.Tangent.[0]
// f(3) = 9, f'(3) = 6 (correct: d/dx[x²] = 2x, at x=3 is 6)

// Using diff helper
let df3 = diff f 3.0
printfn "diff(f, 3) = %f" df3
```

### Chain Rule

```fsharp
// f(x) = sin(x²)
let f (x: Dual) = 
    let x2 = Dual.pow x 2.0
    Dual.sin x2

// f'(x) = cos(x²) * 2x
let d2 = Dual.init [|2.0; 1.0|]
let f2 = f d2
printfn "f(2) = %f, f'(2) = %f" f2.Value f2.Tangent.[0]
// f'(2) = cos(4) * 4 ≈ -2.615

// Verify analytically
let analytical = cos 4.0 * 4.0
printfn "Analytical: %f" analytical
```

### Gradients of Multivariate Functions

```fsharp
// f(x, y) = x² + y²
let f (x: Dual) (y: Dual) = 
    Dual.add (Dual.pow x 2.0) (Dual.pow y 2.0)

// Compute gradient at (3, 4)
let dx = Dual.init [|3.0; 1.0|]  // w.r.t. x
let dy = Dual.init [|4.0; 1.0|]  // w.r.t. y

let df_dx = f dx (Dual.fromFloat 4.0)
let df_dy = f (Dual.fromFloat 3.0) dy

printfn "∇f = (%.1f, %.1f)" df_dx.Tangent.[0] df_dy.Tangent.[0]
// Should be (6, 8) at (3, 4)

// Using grad helper
let gradient = grad2 f 3.0 4.0
printfn "grad(f, 3, 4) = %A" gradient
```

### Jacobian

```fsharp
// f: R² -> R²
// f(x, y) = (x², xy)
let f (x: Dual[]) =
    let xDual = x.[0]
    let yDual = x.[1]
    [|Dual.pow xDual 2.0; Dual.mul xDual yDual|]

// Compute Jacobian at (2, 3)
let jacobian = jacobian f [|2.0; 3.0|]

printfn "Jacobian:"
printfn "  ∂f₁/∂x = %.1f, ∂f₁/∂y = %.1f" jacobian.[0, 0] jacobian.[0, 1]
printfn "  ∂f₂/∂x = %.1f, ∂f₂/∂y = %.1f" jacobian.[1, 0] jacobian.[1, 1]
// Should be:
// ∂f₁/∂x = 2x = 4,  ∂f₁/∂y = 0
// ∂f₂/∂x = y = 3,  ∂f₂/∂y = x = 2
```

## Reverse Mode AD

### Computation Graph

```fsharp
// Build graph
let x = input "x" [|1|]
let y = input "y" [|1|]

let x2 = pow x 2.0
let y2 = pow y 2.0
let sum = add x2 y2
let result = sum

// Forward pass with inputs
let inputs = Map.empty |> Map.add "x" [|3.0|] |> Map.add "y" [|4.0|]
Forward.runWithInputs result inputs |> ignore

printfn "f(3, 4) = %f" result.Value.Value
// f(3, 4) = 25
```

### Backpropagation

```fsharp
// Compute gradients via reverse pass
let gradX = grad result x
let gradY = grad result y

printfn "∂f/∂x = %f, ∂f/∂y = %f" 
    (gradX |> Array.head) (gradY |> Array.head)
// Should be (6, 8)
```

### Higher-Order Derivatives

```fsharp
// f(x) = sin(x)
let f (x: Dual) = Dual.sin x

// First derivative
let f' = diff f
let df_2 = f' 2.0
printfn "f'(2) = %f" df_2  // cos(2) ≈ -0.416

// Second derivative (Hessian for 1D)
let f'' = diff f'
let d2f_2 = f'' 2.0
printfn "f''(2) = %f" d2f_2  // -sin(2) ≈ -0.909

// Third derivative
let f''' = diff f''
let d3f_2 = f''' 2.0
printfn "f'''(2) = %f" d3f_2  // -cos(2) ≈ 0.416

// Using hessian helper
let hessian = hessian f 2.0
printfn "Hessian: %f" hessian
```

### Partial Derivatives

```fsharp
// f(x, y, z) = x*y*z + x² + y² + z²
let f (x: Dual) (y: Dual) (z: Dual) = 
    let xyz = Dual.mul (Dual.mul x y) z
    let x2 = Dual.pow x 2.0
    let y2 = Dual.pow y 2.0
    let z2 = Dual.pow z 2.0
    xyz |> Dual.add x2 |> Dual.add y2 |> Dual.add z2

// Compute partial derivatives at (1, 2, 3)
let dxyz = [|1.0; 2.0; 3.0|]

// ∂f/∂x at (1,2,3) = yz + 2x = 2*3 + 2*1 = 8
let dx = Dual.init [|1.0; 1.0|]
let df_dx = f dx (Dual.fromFloat 2.0) (Dual.fromFloat 3.0)

// ∂f/∂y at (1,2,3) = xz + 2y = 1*3 + 2*2 = 7
let dy = Dual.init [|2.0; 1.0|]
let df_dy = f (Dual.fromFloat 1.0) dy (Dual.fromFloat 3.0)

// ∂f/∂z at (1,2,3) = xy + 2z = 1*2 + 2*3 = 8
let dz = Dual.init [|3.0; 1.0|]
let df_dz = f (Dual.fromFloat 1.0) (Dual.fromFloat 2.0) dz

printfn "∂f/∂x = %f, ∂f/∂y = %f, ∂f/∂z = %f" 
    df_dx.Tangent.[0] df_dy.Tangent.[0] df_dz.Tangent.[0]
```

## Practical Applications

### Gradient Descent with AD

```fsharp
// Rosenbrock function
let rosenbrockDual (x: Dual) (y: Dual) = 
    (Dual.fromFloat 1.0 - x) ** Dual.fromFloat 2.0 + 
    Dual.fromFloat 100.0 * (y - x ** Dual.fromFloat 2.0) ** Dual.fromFloat 2.0

// Use AD for gradient (no manual differentiation!)
let x0 = [|0.0; 0.0|]
let mutable x = x0
let learningRate = 0.001

for i = 1 to 1000 do
    // Compute gradient using AD
    let dx = Dual.init [|x.[0]; 1.0|]
    let dy = Dual.init [|x.[1]; 1.0|]
    
    let df_dx = rosenbrockDual dx (Dual.fromFloat x.[1])
    let df_dy = rosenbrockDual (Dual.fromFloat x.[0]) dy
    
    let grad = [|df_dx.Tangent.[0]; df_dy.Tangent.[0]|]
    
    // Update
    x <- Array.mapi (fun i xi - xi - learningRate * grad.[i]) x
    
    if i % 100 = 0 then
        printfn "Iter %d: x = (%.4f, %.4f)" i x.[0] x.[1]

printfn "Final: (%.6f, %.6f)" x.[0] x.[1]
```

### Neural Network Backpropagation

```fsharp
// Simple network: input -> hidden -> output
let input = input "x" [|2|]
let target = input "y" [|1|]

// Parameters
let W1 = parameter "W1" [|2; 3|] (Array.init 6 (fun _ - Random.Shared.NextDouble()))
let b1 = constantArray (Array.zeroCreate 3) [|3|]
let W2 = parameter "W2" [|3; 1|] (Array.init 3 (fun _ - Random.Shared.NextDouble()))
let b2 = constantArray (Array.zeroCreate 1) [|1|]

// Forward pass
let h1 = activate ReLU (add (matmul input W1) b1)
let logits = add (matmul h1 W2) b2
let output = activate Sigmoid logits
let loss = Loss.binaryCrossEntropy output target

// Training step
let trainStep (xData: float[]) (yData: float[]) learningRate =
    // Set inputs
    Forward.runWithInputs loss 
        (Map.empty |> Map.add "x" xData |> Map.add "y" yData) 
    |> ignore
    
    // Backward pass (automatic gradients!)
    Backward.run [loss] |> ignore
    
    // Get gradients
    let dW1 = W1.Grad
    let db1 = b1.Grad
    let dW2 = W2.Grad
    let db2 = b2.Grad
    
    // Update parameters
    W1.Data <- Array.map2 (fun w dw - w - learningRate * dw) W1.Data dW1
    b1.Data <- Array.map2 (fun b db - b - learningRate * db) b1.Data db1
    W2.Data <- Array.map2 (fun w dw - w - learningRate * dw) W2.Data dW2
    b2.Data <- Array.map2 (fun b db - b - learningRate * db) b2.Data db2

// Train for one sample
let xSample = [|0.5; 0.5|]
let ySample = [|1.0|]

for i = 1 to 100 do
    trainStep xSample ySample 0.1
    if i % 20 = 0 then
        Forward.runWithInputs loss 
            (Map.empty |> Map.add "x" xSample |> Map.add "y" ySample) 
            |> ignore
        printfn "Epoch %d: Loss = %.4f" i loss.Value.Value
```

### Sensitivity Analysis

```fsharp
// f(x, α, β) = α * sin(βx)
let f (x: Dual) (alpha: Dual) (beta: Dual) = 
    Dual.mul alpha (Dual.sin (Dual.mul beta x))

// Compute at x = 1.0, α = 2.0, β = 3.0
let x = Dual.fromFloat 1.0
let alpha = Dual.fromFloat 2.0
let beta = Dual.fromFloat 3.0

let f_val = f x alpha beta
printfn "f(1, 2, 3) = %f" f_val.Value

// Sensitivity to x
let dx = Dual.init [|1.0; 1.0|]
let df_dx = f dx (Dual.fromFloat 2.0) (Dual.fromFloat 3.0)
printfn "∂f/∂x = %f" df_dx.Tangent.[0]  // αβ cos(βx) = 2*3*cos(3) ≈ -5.98

// Sensitivity to α
let dalpha = Dual.init [|2.0; 1.0|]
let df_dalpha = f (Dual.fromFloat 1.0) dalpha (Dual.fromFloat 3.0)
printfn "∂f/∂α = %f" df_dalpha.Tangent.[0]  // sin(βx) = sin(3) ≈ 0.14

// Sensitivity to β
let dbeta = Dual.init [|3.0; 1.0|]
let df_dbeta = f (Dual.fromFloat 1.0) (Dual.fromFloat 2.0) dbeta
printfn "∂f/∂β = %f" df_dbeta.Tangent.[0]  // αx cos(βx) = 2*1*cos(3) ≈ -1.99
```

## Performance Considerations

### Forward vs Reverse Mode

```fsharp
// Forward mode: Good for f: R^n -> R^m where n < m
// Reverse mode: Good for f: R^n -> R^m where n > m

// Example: Scalar-valued function of many inputs (use reverse mode)
let sumOfSquares (x: float[]) = 
    x |> Array.sumBy (fun xi - xi * xi)

// Use reverse mode via computation graph
let inputs = [|for _ in 1..100 - input (sprintf "x%d" i) [|1|]|]
let terms = inputs |> Array.map (fun inp - pow inp 2.0)
let sum = terms |> Array.reduce add

Forward.runWithInputs sum (Map.ofArray [|for i, inp in inputs |> Array.mapi (fun i inp - (sprintf "x%d" i, [|Random.Shared.NextDouble()|])) do yield inp|]) |> ignore
Backward.run [sum] |> ignore

// Gradients are available automatically
let gradients = inputs |> Array.map (fun inp - inp.Grad.[0])
```

### Memory vs Speed

```fsharp
// Forward mode: O(n) operations, O(1) memory
// Reverse mode: O(n) operations, O(n) memory

// For large computation graphs, reverse mode requires storing intermediate values
```

## Exercises

1. Compute gradient of f(x,y,z) = xyz + x²y + y²z + z²x
2. Implement Newton's method using both gradient and Hessian
3. Use AD to implement logistic regression training
4. Compute partial derivatives of a physics simulation
5. Compare forward vs reverse mode performance for different problem sizes

## Solutions

```fsharp
// Exercise 1: Partial derivatives
let f (x: Dual) (y: Dual) (z: Dual) = 
    let xyz = Dual.mul (Dual.mul x y) z
    let x2y = Dual.mul (Dual.pow x 2.0) y
    let y2z = Dual.mul (Dual.pow y 2.0) z
    let z2x = Dual.mul (Dual.pow z 2.0) x
    xyz |> Dual.add x2y |> Dual.add y2z |> Dual.add z2x

// At (1, 2, 3):
// ∂f/∂x = yz + 2xy + z² = 2*3 + 2*1*2 + 3² = 6 + 4 + 9 = 19
// ∂f/∂y = xz + x² + 2yz = 1*3 + 1 + 2*2*3 = 3 + 1 + 12 = 16
// ∂f/∂z = xy + y² + 2zx = 1*2 + 4 + 2*3*1 = 2 + 4 + 6 = 12

// Exercise 2: Newton's method with AD
let newton f x0 maxIter tolerance =
    let mutable x = x0
    for i = 1 to maxIter do
        let fVal = f x
        let fPrime = diff f x
        let fDoublePrime = diff fPrime x
        
        if abs fPrime < 1e-10 then
            failwith "Zero derivative"
        
        let step = fPrime / fDoublePrime
        x <- x - step
        
        if abs step < tolerance then
            break
    x

// Solve x³ - x - 1 = 0
let cubic x = x ** 3.0 - x - 1.0
let root = newton cubic 1.5 100 1e-10
printfn "Newton's root: %.6f" root
```

## Next Steps

- [Tutorial 8: Performance & Best Practices](Tutorial8_Performance.md)
- [User Guide](../USER_GUIDE.md#algorithmic-differentiation)

---

*Estimated time: 45 minutes*