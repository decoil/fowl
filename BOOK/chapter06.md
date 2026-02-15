# Chapter 6: Optimization

## 6.1 Introduction

Optimization is the process of finding the best solution from all feasible solutions. Fowl provides various optimization algorithms for both unconstrained and constrained problems.

## 6.2 Gradient-Based Optimization

### Gradient Descent

```fsharp
open Fowl.Optimization

// Define objective function
let f (x: float[]) =
    x.[0] * x.[0] + x.[1] * x.[1]  // f(x,y) = x² + y²

// Define gradient
let gradF (x: float[]) =
    [| 2.0 * x.[0]; 2.0 * x.[1] |]

// Initial guess
let x0 = [| 5.0; 5.0 |]

// Optimize
let options = { defaultOptions with 
    MaxIter = 1000
    Tol = 1e-6
    LearningRate = 0.1
}

let result = GradientDescent.minimize f gradF x0 options

printfn "Minimum at: %A" result.X
printfn "Function value: %f" result.Fun
printfn "Iterations: %d" result.Nit
printfn "Converged: %b" result.Success
```

### Adaptive Learning Rate

```fsharp
// Automatically adjusts learning rate
let result = GradientDescent.minimizeAdaptive f gradF x0 options
```

### RMSprop

```fsharp
// RMSprop: maintains moving average of squared gradients
let result = RMSprop.minimize f gradF x0 options
```

### Adam

```fsharp
// Adam: combines momentum and RMSprop
let result = Adam.minimize f gradF x0 options
```

### AdamW

```fsharp
// AdamW: Adam with decoupled weight decay
let result = AdamW.minimize f gradF x0 (weightDecay=0.01) options
```

## 6.3 Second-Order Methods

### Newton's Method

```fsharp
// Requires gradient and Hessian
let gradF (x: float[]) = [| 2.0 * x.[0]; 2.0 * x.[1] |]

let hessianF (x: float[]) =
    array2D [|
        [| 2.0; 0.0 |]
        [| 0.0; 2.0 |]
    |]

// Solve H * d = -g for direction
// Then update: x = x + learningRate * d
```

### Quasi-Newton Methods (BFGS)

```fsharp
// BFGS: approximates inverse Hessian
// More efficient than Newton for high dimensions
```

## 6.4 Line Search

```fsharp
// Backtracking line search
let backtrackingLineSearch f grad x direction =
    let alpha = ref 1.0
    let c = 0.5  // Armijo constant
    let rho = 0.5  // Decay rate
    
    let fx = f x
    let gradDotDir = Array.map2 (*) grad direction |> Array.sum
    
    while f (Array.map2 (+) x (Array.map (fun d -> !alpha * d) direction)) >
          fx + c * !alpha * gradDotDir do
        alpha := !alpha * rho
    
    !alpha
```

## 6.5 Constrained Optimization

### Box Constraints

```fsharp
// Projected gradient descent
let projectedGradientDescent f grad x0 (bounds: (float * float)[]) =
    let mutable x = Array.copy x0
    
    for iter = 1 to 1000 do
        let g = grad x
        let step = 0.01
        
        // Gradient step
        for i = 0 to x.Length - 1 do
            x.[i] <- x.[i] - step * g.[i]
        
        // Project onto bounds
        for i = 0 to x.Length - 1 do
            let (lower, upper) = bounds.[i]
            x.[i] <- max lower (min upper x.[i])
    
    x
```

### Lagrange Multipliers

```fsharp
// For equality constraints: minimize f(x) subject to g(x) = 0
// Lagrangian: L(x, λ) = f(x) + λ * g(x)
// Solve: ∇L = 0
```

## 6.6 Global Optimization

### Simulated Annealing

```fsharp
// Good for non-convex problems
let result = SimulatedAnnealing.minimize f x0 (initialTemp=100.0) options
```

### Differential Evolution

```fsharp
// Population-based global optimization
let differentialEvolution f bounds populationSize maxIter =
    let rng = Random()
    let dim = bounds.Length
    
    // Initialize population
    let population = 
        Array.init populationSize (fun _ -
            Array.init dim (fun i -
                let (lo, hi) = bounds.[i]
                lo + rng.NextDouble() * (hi - lo)))
    
    // Evolution loop
    for iter = 1 to maxIter do
        for i = 0 to populationSize - 1 do
            // Mutation and crossover
            // ...
            ()
    
    population
```

## 6.7 Test Functions

### Rosenbrock Function

```fsharp
// Classic optimization test: f(x,y) = (a-x)² + b(y-x²)²
// Global minimum at (a, a²)
let rosenbrock (a: float) (b: float) (x: float[]) =
    pown (a - x.[0]) 2 + b * pown (x.[1] - pown x.[0] 2) 2

// Gradient
let rosenbrockGrad (a: float) (b: float) (x: float[]) =
    [|
        -2.0 * (a - x.[0]) - 4.0 * b * x.[0] * (x.[1] - pown x.[0] 2)
        2.0 * b * (x.[1] - pown x.[0] 2)
    |]
```

### Other Test Functions

```fsharp
// Sphere function: f(x) = Σ xᵢ²
let sphere (x: float[]) = Array.sumBy (fun xi -> xi * xi) x

// Rastrigin function: many local minima
let rastrigin (x: float[]) =
    let A = 10.0
    A * float x.Length + 
    Array.sumBy (fun xi -> xi * xi - A * cos(2.0 * PI * xi)) x

// Ackley function
let ackley (x: float[]) =
    let a = 20.0
    let b = 0.2
    let c = 2.0 * PI
    let n = float x.Length
    let sum1 = Array.sumBy (fun xi -> xi * xi) x
    let sum2 = Array.sumBy (fun xi -> cos(c * xi)) x
    -a * exp(-b * sqrt(sum1 / n)) - exp(sum2 / n) + a + exp(1.0)
```

## 6.8 Applications

### Linear Regression

```fsharp
// Fit y = Xβ by minimizing ||y - Xβ||²
let linearRegression (X: float[,]) (y: float[]) =
    let n = X.GetLength(0)
    let p = X.GetLength(1)
    
    // Objective: f(β) = ||y - Xβ||²
    let f (beta: float[]) =
        let residuals = 
            Array.init n (fun i -
                let pred = Array.sumBy (fun j -> beta.[j] * X.[i,j]) [|0..p-1|]
                y.[i] - pred)
        Array.sumBy (fun r -> r * r) residuals
    
    // Gradient: ∇f = -2Xᵀ(y - Xβ)
    let grad (beta: float[]) =
        let residuals = 
            Array.init n (fun i -
                let pred = Array.sumBy (fun j -> beta.[j] * X.[i,j]) [|0..p-1|]
                y.[i] - pred)
        Array.init p (fun j -
            -2.0 * Array.sumBy (fun i -> X.[i,j] * residuals.[i]) [|0..n-1|])
    
    let beta0 = Array.zeroCreate p
    GradientDescent.minimize f grad beta0 defaultOptions
```

### Logistic Regression

```fsharp
// Binary classification via optimization
let logisticRegression (X: float[,]) (y: float[]) =
    let n = X.GetLength(0)
    let p = X.GetLength(1)
    
    let sigmoid z = 1.0 / (1.0 + exp(-z))
    
    // Negative log-likelihood
    let f (beta: float[]) =
        Array.init n (fun i -
            let z = Array.sumBy (fun j -> beta.[j] * X.[i,j]) [|0..p-1|]
            let p = sigmoid z
            if y.[i] = 1.0 then -log p else -log(1.0 - p))
        |> Array.sum
    
    let grad (beta: float[]) =
        // ... gradient computation ...
        Array.zeroCreate p
    
    let beta0 = Array.zeroCreate p
    GradientDescent.minimize f grad beta0 defaultOptions
```

### Neural Network Training

```fsharp
// Training as optimization
let trainNetwork (model: Model) (data: (float[] * float[])[]) =
    // Parameters
    let parameters = getParameters model
    
    // Loss function
    let loss () =
        data
        |> Array.sumBy (fun (x, y) -
            let pred = forward model x
            mseLoss pred y)
    
    // Gradient via backpropagation
    let grad () =
        resetGradients model
        let totalLoss = ref 0.0
        for (x, y) in data do
            let pred = forward model x
            let loss = mseLoss pred y
            totalLoss := !totalLoss + loss
            backward model loss
        getGradients parameters
    
    // Optimize
    Adam.minimize loss grad parameters defaultOptions
```

## 6.9 Exercises

### Exercise 6.1: Gradient Checking

```fsharp
let checkGradient f grad x (epsilon: float) =
    let n = x.Length
    let analytical = grad x
    let numerical = Array.zeroCreate n
    
    for i = 0 to n - 1 do
        let xPlus = Array.copy x
        let xMinus = Array.copy x
        xPlus.[i] <- xPlus.[i] + epsilon
        xMinus.[i] <- xMinus.[i] - epsilon
        numerical.[i] <- (f xPlus - f xMinus) / (2.0 * epsilon)
    
    // Compare
    let diff = Array.map2 (fun a n -> abs(a - n)) analytical numerical
    Array.max diff

// Usage
let error = checkGradient f grad x0 1e-5
printfn "Gradient check error: %e" error
```

### Exercise 6.2: Momentum Visualization

```fsharp
// Visualize optimization trajectory
let traceOptimization f grad x0 optimizer =
    let mutable x = Array.copy x0
    let trajectory = ResizeArray<float[]>()
    trajectory.Add(x)
    
    for iter = 1 to 100 do
        let g = grad x
        // Update with momentum
        // ...
        trajectory.Add(Array.copy x)
    
    trajectory.ToArray()
```

### Exercise 6.3: Constraint Handling

```fsharp
// Project onto probability simplex
let projectOntoSimplex (y: float[]) =
    let n = y.Length
    let sorted = Array.sortDescending y
    let mutable cumsum = 0.0
    let mutable rho = 0
    
    for i = 0 to n - 1 do
        cumsum <- cumsum + sorted.[i]
        let t = (cumsum - 1.0) / float (i + 1)
        if sorted.[i] - t > 0.0 then
            rho <- i + 1
    
    let lambda = (Array.sum sorted.[..rho-1] - 1.0) / float rho
    y |> Array.map (fun yi -> max 0.0 (yi - lambda))
```

## 6.10 Best Practices

1. **Start with Adam**: Good default choice for most problems
2. **Learning rate scheduling**: Decay over time
3. **Gradient clipping**: Prevent exploding gradients
4. **Early stopping**: Monitor validation loss
5. **Random restarts**: For non-convex problems

## 6.11 Summary

Key concepts:
- Gradient descent: follow negative gradient
- Momentum: accelerate in consistent directions
- Adaptive methods: per-parameter learning rates
- Line search: find optimal step size
- Constraints: projection or Lagrange multipliers

---

*Next: [Chapter 7: Algorithmic Differentiation](chapter07.md)*
