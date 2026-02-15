# Tutorial 6: Optimization

Minimize functions and solve optimization problems with Fowl's optimization algorithms.

## Overview

Optimization is essential for machine learning, scientific computing, and many other fields. Fowl provides gradient-based and global optimization algorithms.

## Learning Objectives

- Use gradient descent methods
- Apply automatic differentiation for gradients
- Perform global optimization
- Solve constrained optimization problems
- Tune hyperparameters

## Setup

```fsharp
open Fowl
open Fowl.Core.Types
open Fowl.Optimization
open Fowl.AD

let unwrap = function Ok x -> x | Error e -> failwith e.Message
```

## Gradient-Based Optimization

### Simple Gradient Descent

```fsharp
// Define objective function: f(x) = x²
let f (x: float) = x * x

// Define gradient: f'(x) = 2x
let df (x: float) = 2.0 * x

// Minimize starting from x = 10
let options = { 
    MaxIter = 100 
    Tolerance = 1e-6
    LearningRate = 0.1
    Verbose = true
}

let result = GradientDescent.minimize1D f df 10.0 options

printfn "Minimum at x = %.6f" result.X
printfn "Function value: %.6f" result.Fun
printfn "Iterations: %d" result.Nit
// Should converge to x = 0
```

### Multivariate Optimization

```fsharp
// Rosenbrock function (classic test function)
// f(x,y) = (1-x)² + 100(y-x²)²
let rosenbrock (x: float[]) =
    (1.0 - x.[0]) ** 2.0 + 100.0 * (x.[1] - x.[0] ** 2.0) ** 2.0

// Analytical gradient
let rosenbrockGrad (x: float[]) =
    let dx0 = -2.0 * (1.0 - x.[0]) - 400.0 * x.[0] * (x.[1] - x.[0] ** 2.0)
    let dx1 = 200.0 * (x.[1] - x.[0] ** 2.0)
    [|dx0; dx1|]

// Minimize starting from (0,0)
let x0 = [|0.0; 0.0|]
let options = { 
    MaxIter = 1000
    Tolerance = 1e-6
    LearningRate = 0.001
    Verbose = true
}

let result = GradientDescent.minimize rosenbrock rosenbrockGrad x0 options

printfn "Optimum: x = (%.6f, %.6f)" result.X.[0] result.X.[1]
printfn "Function value: %.10f" result.Fun
// Should converge to (1, 1) with f = 0
```

### Using Automatic Differentiation

```fsharp
// Define function using Dual numbers for automatic differentiation
let fDual (x: Dual) (y: Dual) = (Dual.fromFloat 1.0 - x) ** Dual.fromFloat 2.0 + 
                                Dual.fromFloat 100.0 * (y - x ** Dual.fromFloat 2.0) ** Dual.fromFloat 2.0

// Compute gradient automatically
let x0Dual = Dual.init [|0.0; 0.0|]
let gradient = grad fDual x0Dual

printfn "Automatic gradient at (0,0): %A" gradient
// Compare with analytical gradient:
// ∇f = [-2(1-x) - 400x(y-x²), 200(y-x²)]
// At (0,0): [-2, 0]
```

### Adam Optimizer

```fsharp
// Beale function
let beale (x: float[]) =
    (1.5 - x.[0] + x.[0] * x.[1]) ** 2.0 +
    (2.25 - x.[0] + x.[0] * x.[1] ** 2.0) ** 2.0 +
    (2.625 - x.[0] + x.[0] * x.[1] ** 3.0) ** 2.0

// Use AD for gradient
let bealeDual (x: Dual) (y: Dual) = 
    let x0 = x
    let x1 = y
    (Dual.fromFloat 1.5 - x0 + x0 * x1) ** Dual.fromFloat 2.0 +
    (Dual.fromFloat 2.25 - x0 + x0 * x1 ** Dual.fromFloat 2.0) ** Dual.fromFloat 2.0 +
    (Dual.fromFloat 2.625 - x0 + x0 * x1 ** Dual.fromFloat 3.0) ** Dual.fromFloat 2.0

// Use Adam optimizer
let x0 = [|0.0; 0.0|]
let options = { 
    MaxIter = 2000
    Tolerance = 1e-6
    LearningRate = 0.01
    Verbose = true
}

let result = Adam.minimize beale bealeDual x0 options

printfn "Adam optimum: x = (%.6f, %.6f)" result.X.[0] result.X.[1]
printfn "Function value: %.10f" result.Fun
```

### Line Search

```fsharp
// Use line search to find optimal step size
let options = { 
    MaxIter = 100
    Tolerance = 1e-6
    LearningRate = 0.1  // Initial guess
    Verbose = true
    UseLineSearch = true  // Enable line search
}

let result = GradientDescent.minimize rosenbrock rosenbrockGrad x0 options

printfn "With line search: %d iterations" result.Nit
```

## Momentum Methods

### SGD with Momentum

```fsharp
// Create a path visualization function
let optimizePath f df x0 learningRate momentum maxIter =
    let mutable x = x0
    let mutable velocity = Array.map (fun _ - 0.0) x0
    let path = [x]
    
    for i = 1 to maxIter do
        let grad = df x
        
        // Update velocity
        velocity <- Array.map2 (fun v g - momentum * v - learningRate * g) velocity grad
        
        // Update position
        x <- Array.map2 (+) x velocity
        path <- path @ [x]
        
        if Array.sqrt (Array.sumBy (fun g - g * g) grad) < 1e-6 then
            break
    
    Array.ofList path

// Compare: without momentum vs with momentum
let pathNoMomentum = optimizePath rosenbrock rosenbrockGrad [|0.0; 0.0|] 0.001 0.0 100
let pathWithMomentum = optimizePath rosenbrock rosenbrockGrad [|0.0; 0.0|] 0.001 0.9 100

printfn "No momentum: %d iterations, final f = %.6f" 
    pathNoMomentum.Length (rosenbrock pathNoMomentum.[pathNoMomentum.Length - 1])
printfn "With momentum: %d iterations, final f = %.6f" 
    pathWithMomentum.Length (rosenbrock pathWithMomentum.[pathWithMomentum.Length - 1])
```

## Global Optimization

### Simulated Annealing

```fsharp
// Rastrigin function (many local minima)
let rastrigin (x: float[]) =
    let n = float x.Length
    let sum = Array.sumBy (fun xi - 
        xi ** 2.0 - 10.0 * cos (2.0 * System.Math.PI * xi) + 10.0) x
    10.0 * n + sum

// Define search bounds
let bounds = [|(-5.12, 5.12); (-5.12, 5.12)|]

// Simulated annealing
let options = { 
    MaxIter = 10000
    Tolerance = 1e-6
    LearningRate = 0.01
    Verbose = true
    InitialTemp = 100.0
    CoolingRate = 0.995
}

let saResult = SimulatedAnnealing.minimize rastrigin [|0.0; 0.0|] bounds options

printfn "SA minimum: x = (%.6f, %.6f)" saResult.X.[0] saResult.X.[1]
printfn "Function value: %.6f" saResult.Fun
// Should find global minimum near (0, 0)
```

### Basin Hopping

```fsharp
// Basin hopping: combine local and global search
let basinHopping f df x0 bounds iterations = 
    let mutable bestX = x0
    let mutable bestFun = f x0
    
    for i = 1 to iterations do
        // Local optimization
        let localResult = GradientDescent.minimize f df bestX { 
            MaxIter = 100
            Tolerance = 1e-6
            LearningRate = 0.01
            Verbose = false
        }
        
        // Random jump within bounds
        let jumpX = 
            bestX
        |> Array.mapi (fun i x - 
            let (lb, ub) = bounds.[i]
            x + (Random.Shared.NextDouble() - 0.5) * (ub - lb) * 0.5)
        
        // Clamp to bounds
        let clampedX = 
            jumpX
        |> Array.mapi (fun i x - 
            let (lb, ub) = bounds.[i]
            max lb (min ub x))
        
        // Evaluate jump
        let jumpFun = f clampedX
        
        // Accept if better
        if jumpFun < bestFun then
            bestX <- clampedX
            bestFun <- jumpFun
    
    { X = bestX; Fun = bestFun; Nit = iterations; Success = true; Message = "Basin hopping complete" }

let bhResult = basinHopping rastrigin 
    (fun x - [|0.0; 0.0|] |> Array.map2 (*) (rastriginDual [|0.0; 0.0|] |> grad)  // Placeholder)
    [|0.0; 0.0|] bounds 100

printfn "Basin hopping minimum: x = (%.6f, %.6f)" bhResult.X.[0] bhResult.X.[1]
```

## Constrained Optimization

### Simple Bounds

```fsharp
// Minimize with bounds using projection
let minimizeWithBounds f df x0 bounds options =
    let mutable x = x0
    
    for i = 1 to options.MaxIter do
        let grad = df x
        
        // Gradient step
        let step = Array.map (fun g - -options.LearningRate * g) grad
        x <- Array.map2 (+) x step
        
        // Project onto bounds
        x <- Array.mapi (fun i xi - 
            let (lb, ub) = bounds.[i]
            max lb (min ub xi)) x
        
        // Check convergence
        if Array.sqrt (Array.sumBy (fun g - g * g) grad) < options.Tolerance then
            break
    
    { X = x; Fun = f x; Nit = i; Success = true; Message = "Converged" }

// Constrained minimization
let bounds = [|(-2.0, 2.0); (-2.0, 2.0)|]
let constrainedResult = minimizeWithBounds rosenbrock rosenbrockGrad [|0.0; 0.0|] bounds options

printfn "Constrained optimum: x = (%.6f, %.6f)" constrainedResult.X.[0] constrainedResult.X.[1]
```

### Penalty Methods

```fsharp
// Add penalty term for constraint violation
let minimizeWithPenalty f df x0 constraintFun constraintGrad penaltyCoeff options =
    let penalizedFun x =
        f x + penaltyCoeff * max 0.0 (constraintFun x) ** 2.0
    
    let penalizedGrad x =
        let grad = df x
        let constraintViolation = max 0.0 (constraintFun x)
        let penaltyGrad = 2.0 * penaltyCoeff * constraintViolation
        let cGrad = constraintGrad x
        Array.map2 (fun g cg - g + penaltyGrad * cg) grad cGrad
    
    GradientDescent.minimize penalizedFun penalizedGrad x0 options

// Constraint: x[0] + x[1] >= 1.5
let constraintFun (x: float[]) = 1.5 - (x.[0] + x.[1])
let constraintGrad (x: float[]) - [| -1.0; -1.0 |]

let penaltyResult = minimizeWithPenalty rosenbrock rosenbrockGrad 
    [|0.0; 0.0|] constraintFun constraintGrad 1000.0 options

printfn "Penalty optimum: x = (%.6f, %.6f)" penaltyResult.X.[0] penaltyResult.X.[1]
printfn "Constraint satisfaction: %.3f >= 1.5" (penaltyResult.X.[0] + penaltyResult.X.[1])
```

## Practical Examples

### Example 1: Linear Regression

```fsharp
// Generate data
let X = Array.init 100 (fun _ - Random.Shared.NextDouble() * 10.0)
let y = Array.map2 (fun xi _ - 
    3.0 + 2.0 * xi + Random.Shared.NextNormal() * 2.0) X X

// Define model: y = w0 + w1 * x
let model (w: float[]) (x: float[]) =
    Array.map (fun xi - w.[0] + w.[1] * xi) x

// Define loss: MSE
let lossFun w =
    let yPred = model w X
    Array.zip y yPred
    |> Array.sumBy (fun (yi, ypi) - (yi - ypi) ** 2.0)
    |> (*) (1.0 / float X.Length)

// Gradient (analytical)
let lossGrad w =
    let yPred = model w X
    let residual = Array.map2 (-) y yPred
    let dw0 = Array.sum residual / float X.Length * -2.0
    let dw1 = Array.sum (Array.map2 (*) residual X) / float X.Length * -2.0
    [|dw0; dw1|]

// Optimize
let w0 = [|0.0; 0.0|]
let options = { 
    MaxIter = 1000
    Tolerance = 1e-6
    LearningRate = 0.001
    Verbose = false
}

let result = GradientDescent.minimize lossFun lossGrad w0 options

printfn "Estimated: w0 = %.3f, w1 = %.3f" result.X.[0] result.X.[1]
printfn "Expected: w0 = 3.0, w1 = 2.0"
```

### Example 2: Logistic Regression

```fsharp
// Binary classification
let X = [|[|1.0; 2.0|]; [|2.0; 3.0|]; [|3.0; 1.0|]; [|4.0; 5.0|]|]
let y = [|0.0; 0.0; 1.0; 1.0|]

// Sigmoid function
let sigmoid z = 1.0 / (1.0 + exp (-z))

// Logistic loss
let logisticLoss w =
    Array.mapi (fun i xi - 
        let zi = w.[0] + w.[1] * xi.[0] + w.[2] * xi.[1]
        if y.[i] = 1.0 then 
            -log (sigmoid zi)
        else 
            -log (1.0 - sigmoid zi)
    ) X
    |> Array.sum

// Gradient
let logisticGrad w =
    [|0.0; 0.0; 0.0|]
    |> Array.mapi (fun j dw - 
        X
        |> Array.mapi (fun i xi - 
            let zi = w.[0] + w.[1] * xi.[0] + w.[2] * xi.[1]
            let xij = if j = 0 then 1.0 elif j = 1 then xi.[0] else xi.[1]
            (sigmoid zi - y.[i]) * xij)
        |> Array.sum
    )

// Optimize
let w0 = [|0.0; 0.0; 0.0|]
let options = { 
    MaxIter = 1000
    Tolerance = 1e-6
    LearningRate = 0.01
    Verbose = false
}

let result = GradientDescent.minimize logisticLoss logisticGrad w0 options

printfn "Logistic weights: %A" result.X
```

### Example 3: Portfolio Optimization

```fsharp
// Mean-Variance Portfolio Optimization
let mu = [|0.1; 0.15; 0.2|]  // Expected returns
let sigma = [|0.2; 0.3; 0.4|]  // Volatility
let corr = [|1.0; 0.5; 0.3; 0.5; 1.0; 0.4; 0.3; 0.4; 1.0|]  // Correlation matrix (flattened)

// Build covariance matrix
let cov = 
    Array2D.init 3 3 (fun i j - 
        let ci = sigma.[i]
        let cj = sigma.[j]
        let rij = corr.[i * 3 + j]
        ci * cj * rij)

// Portfolio variance
let portfolioVariance w = 
    let wArray = Array.ofList (Array.toList w)
    let wCol = Array2D.init 3 1 (fun i _ - wArray.[i])
    let wRow = Array2D.init 1 3 (fun _ i - wArray.[i])
    let wSigmaW = Matrix.matmul (Matrix.matmul wRow cov |> Result.get) wCol |> Result.get
    wSigmaW.[0, 0]

// Portfolio return
let portfolioReturn w = 
    Array.sum2 mu (Array.ofList (Array.toList w))

// Efficient frontier: minimize variance for given return
let targetReturn = 0.15

let efficientFrontierLoss w =
    let variance = portfolioVariance w
    let ret = portfolioReturn w
    variance + 1000.0 * (ret - targetReturn) ** 2.0  // Penalty for wrong return

// Optimize
let w0 = [0.33; 0.33; 0.34]
let options = { 
    MaxIter = 1000
    Tolerance = 1e-6
    LearningRate = 0.01
    Verbose = false
}

let result = GradientDescent.minimize efficientFrontierLoss (fun _ - [|0.0; 0.0; 0.0|]) w0 options

printfn "Optimal weights: %A" result.X
printfn "Portfolio return: %.3f" (portfolioReturn result.X)
printfn "Portfolio variance: %.6f" (portfolioVariance result.X)
```

## Exercises

1. Minimize the Himmelblau function with multiple methods
2. Implement a mini-batch gradient descent algorithm
3. Use optimization to fit a Gaussian to data
4. Solve a constrained optimization problem with inequalities
5. Implement a simple genetic algorithm for global optimization

## Solutions

```fsharp
// Exercise 1: Himmelblau function (4 minima)
let himmelblau (x: float[]) =
    (x.[0] ** 2.0 + x.[1] - 11.0) ** 2.0 +
    (x.[0] + x.[1] ** 2.0 - 7.0) ** 2.0

// Try different starting points
let startingPoints = [|[-2.0; 0.0]; [0.0; -2.0]; [2.0; 2.0]; [-1.0; -1.0]|]

startingPoints |> Array.iter (fun start - 
    let result = GradientDescent.minimize himmelblau (fun _ - [|0.0; 0.0|]) start options
    printfn "Start %A -> Min: (%.3f, %.3f)" start result.X.[0] result.X.[1]
)

// Exercise 3: Fit Gaussian
// Data points
let dataX = [|1.0; 2.0; 3.0; 4.0; 5.0|]
let dataY = [|0.8; 2.1; 4.0; 2.2; 0.9|]

// Gaussian model: A * exp(-(x-mu)²/(2*sigma²))
let gaussian w x =
    w.[0] * exp (-((x - w.[1]) ** 2.0) / (2.0 * w.[2] ** 2.0))

// Loss
let gaussianLoss w =
    dataX
    |> Array.map (fun x - gaussian w x)
    |> Array.zip dataY
    |> Array.sumBy (fun (y, yPred) - (y - yPred) ** 2.0)

// Optimize
let w0 = [|1.0; 3.0; 1.0|]
let result = GradientDescent.minimize gaussianLoss (fun _ - [|0.0; 0.0; 0.0|]) w0 options

printfn "Gaussian parameters: A=%.3f, mu=%.3f, sigma=%.3f" result.X.[0] result.X.[1] result.X.[2]
```

## Next Steps

- [Tutorial 7: Algorithmic Differentiation](Tutorial7_AlgorithmicDifferentiation.md)
- [User Guide](../USER_GUIDE.md#optimization)

---

*Estimated time: 60 minutes*