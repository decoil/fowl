# Chapter 7: Algorithmic Differentiation

## 7.1 Introduction

Algorithmic Differentiation (AD) computes exact derivatives of functions specified by computer programs. Unlike numerical differentiation (finite differences) or symbolic differentiation, AD provides machine-precision derivatives with minimal computational overhead.

## 7.2 Why Automatic Differentiation?

### Comparison of Methods

```fsharp
// Function: f(x) = sin(x) * exp(x)
let f x = sin x * exp x

// 1. Numerical differentiation (finite differences)
// f'(x) ≈ (f(x+h) - f(x-h)) / 2h
// Problems: truncation error, cancellation error

// 2. Symbolic differentiation
// f'(x) = cos(x)*exp(x) + sin(x)*exp(x)
// Problems: expression swell, hard to implement

// 3. Automatic differentiation (AD)
// Exact derivatives, efficient computation
// Fowl's approach!
```

### The Chain Rule

AD is based on the chain rule from calculus:

If `y = f(g(x))`, then `dy/dx = f'(g(x)) * g'(x)`

For composite functions, this extends naturally:

```
dy/dx = dy/du * du/dv * dv/dw * ... * dz/dx
```

## 7.3 Forward Mode AD

### Dual Numbers

Forward mode uses dual numbers: `x = a + bε` where `ε² = 0`

```fsharp
open Fowl.AD

// Pack a value for differentiation
let x = pack_flt 2.0

// Compute f(x) = x^2
let y = x * x

// Get value and derivative
let value, derivative = diff' (fun x -> x * x) (pack_flt 2.0)
// value = 4.0, derivative = 4.0 (because d(x²)/dx = 2x = 4 at x=2)
```

### How It Works

```fsharp
// Each operation propagates both value and derivative
// For f(x) = x² at x = 3:

// x = 3 + 1ε (1ε means we're computing df/dx)
// x² = (3 + 1ε)² = 9 + 6ε + 1ε² = 9 + 6ε

// So f(3) = 9 and f'(3) = 6 ✓
```

### Elementary Functions

```fsharp
// All elementary functions support automatic differentiation
let x = pack_flt 1.0

let y1 = sin x    // dy/dx = cos(x)
let y2 = cos x    // dy/dx = -sin(x)
let y3 = exp x    // dy/dx = exp(x)
let y4 = log x    // dy/dx = 1/x
let y5 = sqrt x   // dy/dx = 1/(2*sqrt(x))
let y6 = x ** pack_flt 3.0  // dy/dx = 3*x²
```

### Practical Example: Physics Simulation

```fsharp
// Compute velocity from position
// s(t) = s₀ + v₀*t + 0.5*a*t²

let position (s0: dual) (v0: dual) (a: dual) (t: dual) =
    s0 + v0 * t + pack_flt 0.5 * a * t * t

// Get velocity at t=5 (derivative of position)
let t = pack_flt 5.0
let _, velocity = diff' (position (pack_flt 0.0) (pack_flt 10.0) (pack_flt 2.0)) t

// velocity = v0 + a*t = 10 + 2*5 = 20
```

## 7.4 Reverse Mode AD

### When to Use Reverse Mode

Forward mode: O(1) sweeps for single input
Reverse mode: O(1) sweeps for single output

For f: ℝⁿ → ℝᵐ:
- Use forward mode when n ≪ m (few inputs, many outputs)
- Use reverse mode when m ≪ n (many inputs, few outputs) ← Neural networks!

### Backpropagation

```fsharp
// Reverse mode is the foundation of backpropagation
open Fowl.AD

// Compute gradient of f(x,y) = x² + y² at (3, 4)
let f (x: dual) (y: dual) = x * x + y * y

let point = make_dual [|3.0; 4.0|]
let! gradient = grad f point

// gradient = [|6.0; 8.0|] (partial derivatives)
```

### Computational Graph

Reverse mode builds a computation graph:

```
x → [square] → x² ↘
                     [add] → result
y → [square] → y² ↗
```

Forward pass: compute values
Backward pass: compute gradients

```fsharp
// Forward pass
let x_val = 3.0
let y_val = 4.0
let x_sq = x_val * x_val      // 9.0
let y_sq = y_val * y_val      // 16.0
let result = x_sq + y_sq      // 25.0

// Backward pass (chain rule)
// d(result)/d(x_sq) = 1
// d(result)/d(y_sq) = 1
// d(x_sq)/d(x) = 2*x = 6
// d(y_sq)/d(y) = 2*y = 8
// d(result)/d(x) = 1 * 6 = 6
// d(result)/d(y) = 1 * 8 = 8
```

## 7.5 Higher-Order Derivatives

### Second Derivatives (Hessian)

```fsharp
// Compute Hessian matrix of f(x,y)
let f (x: dual) (y: dual) = x * x * y + y * y

// Hessian = [[d²f/dx², d²f/dxdy],
//            [d²f/dydx, d²f/dy²]]
let! hessian = hessian f (make_dual [|1.0; 2.0|])
```

### Applications in Optimization

Second derivatives enable Newton's method:

```fsharp
// Newton's method update: x ← x - f'(x)/f''(x)
let newtonStep f x =
    let g = diff f x    // First derivative
    let h = diff2 f x   // Second derivative
    x - g / h
```

## 7.6 Implementing Custom Derivatives

### Registering Custom Functions

```fsharp
// Define custom function with manual derivatives
let softplus x =
    let primal = log(1.0 + exp x)
    // Derivative: 1 / (1 + exp(-x)) = sigmoid(x)
    let tangent = 1.0 / (1.0 + exp(-x))
    make_dual primal tangent

// Or use existing composition
let softplus_composed x = log (pack_flt 1.0 + exp x)
```

### Handling Control Flow

AD works through conditionals and loops:

```fsharp
// ReLU function
let relu (x: dual) =
    if unpack_flt x > 0.0 then x else pack_flt 0.0

// Derivative is:
// 1 if x > 0
// 0 if x < 0
// undefined at x = 0 (we choose 0 or 1)
```

## 7.7 Neural Network Application

### Gradient Descent with AD

```fsharp
// Automatic backpropagation
open Fowl.Neural
open Fowl.AD

// Define network
let input = Neural.Graph.input "x" [||]
let weights = Neural.Graph.parameter "W" [||] [|0.5|]
let bias = Neural.Graph.constant 0.1

let output = Neural.Graph.add (Neural.Graph.mul input weights) bias
let loss = Neural.Graph.pow (Neural.Graph.sub output target) (Neural.Graph.constant 2.0)

// Compute gradients automatically
let gradients = Neural.Backward.run [loss]

// Update parameters
let learningRate = 0.01
weights.Value.Value <- Some (weights.Value.Value.Value - learningRate * weights.Grad.Value.Value)
```

### Training Loop

```fsharp
let trainStep model lossFn (x, y) learningRate =
    // Forward pass
    let prediction = model x
    let loss = lossFn prediction y
    
    // Backward pass (automatic differentiation)
    let gradients = grad loss parameters
    
    // Update parameters
    for (param, grad) in List.zip parameters gradients do
        param <- param - learningRate * grad
```

## 7.8 Advanced Topics

### Checkpointing

For memory-efficient reverse mode:

```fsharp
// Store intermediate values only at checkpoints
// Trade computation for memory
let checkpointedFunction x =
    // Compute in segments
    let segment1 = checkpoint computeSegment1 x
    let segment2 = checkpoint computeSegment2 segment1
    let segment3 = checkpoint computeSegment3 segment2
    segment3
```

### Mixed Mode AD

Combine forward and reverse mode for efficiency:

```fsharp
// For f: ℝⁿ → ℝᵐ where both n and m are large
// Use forward for some dimensions, reverse for others
```

### Custom Primitives

For performance-critical operations:

```fsharp
// Define custom forward/backward for complex operations
let customConv2d input weights =
    // Manual forward pass
    let output = manualConvolve input weights
    
    // Manual backward function
    let backward gradOutput =
        let gradInput = manualConvolveGradInput gradOutput weights
        let gradWeights = manualConvolveGradWeights gradOutput input
        (gradInput, gradWeights)
    
    output, backward
```

## 7.9 Exercises

### Exercise 7.1: Verify AD Correctness

```fsharp
// Compare AD with analytical derivatives
let verifyDerivative f df_analytical x =
    let df_ad = diff f (pack_flt x) |> unpack_flt
    let df_exact = df_analytical x
    abs(df_ad - df_exact) < 1e-10

// Test on various functions
verifyDerivative sin cos 1.0
verifyDerivative (fun x -> x * x) (fun x -> 2.0 * x) 3.0
verifyDerivative exp exp 0.0
```

### Exercise 7.2: Implement Jacobian

```fsharp
// Compute Jacobian matrix of vector-valued function
let jacobian (f: dual[] -> dual[]) (x: float[]) : float[,] =
    let n = x.Length
    let m = f (Array.map pack_flt x).Length
    let J = Array2D.zeroCreate m n
    
    for j = 0 to n - 1 do
        // Seed j-th input
        let x_dual = Array.mapi (fun i xi ->
            if i = j then make_dual (xi, 1.0) else pack_flt xi) x
        
        let y = f x_dual
        for i = 0 to m - 1 do
            J.[i,j] <- unpack_tangent y.[i]
    
    J
```

### Exercise 7.3: Gradient Checking

```fsharp
// Verify gradients numerically
let checkGradients f x epsilon =
    let n = x.Length
    let analytical = grad f (make_dual x)
    let numerical = Array.zeroCreate n
    
    for i = 0 to n - 1 do
        let x_plus = Array.copy x
        let x_minus = Array.copy x
        x_plus.[i] <- x_plus.[i] + epsilon
        x_minus.[i] <- x_minus.[i] - epsilon
        
        numerical.[i] <- (f (make_dual x_plus) - f (make_dual x_minus)) / (2.0 * epsilon)
    
    Array.map2 (-) analytical numerical
    |> Array.map abs
    |> Array.max

// Should be < 1e-6 for well-behaved functions
```

## 7.10 Performance Considerations

### Memory vs Computation Trade-off

```
Forward mode: O(1) memory, O(n) computation for n inputs
Reverse mode: O(graph size) memory, O(1) computation per gradient
```

### When to Use Each Mode

```fsharp
// Forward mode - few inputs, many outputs
let f (x: float) = [|sin x; cos x; exp x|]  // 1 input, 3 outputs

// Reverse mode - many inputs, few outputs  ← Neural networks!
let loss (weights: float[]) = computeLoss weights  // n inputs, 1 output
```

## 7.11 Summary

Key concepts:
- Forward mode: propagate derivatives with values (dual numbers)
- Reverse mode: build computation graph, propagate backwards (backprop)
- Both give exact derivatives (up to floating-point precision)
- AD enables gradient-based optimization of complex functions
- Foundation of modern deep learning

---

*Next: [Chapter 8: Signal Processing](chapter08.md)*
