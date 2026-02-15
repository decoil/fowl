namespace Fowl.Optimization

open System
open Fowl
open Fowl.Core.Types

/// <summary>Result from optimization.
/// </summary>
type OptimizationResult = {
    /// Optimal parameters
    X: float[]
    /// Function value at optimum
    Fun: float
    /// Number of iterations
    Nit: int
    /// Whether convergence was achieved
    Success: bool
    /// Message describing termination
    Message: string
}

/// <summary>Options for optimization algorithms.
/// </summary>
type OptimizationOptions = {
    /// Maximum iterations
    MaxIter: int
    /// Convergence tolerance
    Tol: float
    /// Learning rate (for gradient-based)
    LearningRate: float
    /// Verbosity level
    Verbose: bool
}

/// <summary>Default optimization options.
/// </summary>
let defaultOptions = {
    MaxIter = 1000
    Tol = 1e-6
    LearningRate = 0.01
    Verbose = false
}

/// <summary>Gradient descent optimizer.
/// </summary>
module GradientDescent =
    
    /// <summary>Minimize function using gradient descent.
    /// </summary>
    let minimize (f: float[] -> float) 
                 (gradF: float[] -> float[]) 
                 (x0: float[])
                 (options: OptimizationOptions) : OptimizationResult =
        
        let mutable x = Array.copy x0
        let mutable fx = f x
        let mutable converged = false
        let mutable iter = 0
        let mutable message = "Maximum iterations reached"
        
        while not converged && iter < options.MaxIter do
            let grad = gradF x
            let xNew = Array.map2 (fun xi gi -
                xi - options.LearningRate * gi) x grad
            let fxNew = f xNew
            
            if abs (fxNew - fx) < options.Tol then
                converged <- true
                message <- "Converged (function value)"
            elif Array.map2 (fun xi xni -> abs (xni - xi)) x xNew |> Array.max < options.Tol then
                converged <- true
                message <- "Converged (parameter change)"
            
            x <- xNew
            fx <- fxNew
            iter <- iter + 1
            
            if options.Verbose && iter % 100 = 0 then
                printfn "Iter %d: f(x) = %.10f" iter fx
        
        {
            X = x
            Fun = fx
            Nit = iter
            Success = converged
            Message = message
        }
    
    /// <summary>Minimize with adaptive learning rate.
    /// </summary>
    let minimizeAdaptive (f: float[] -> float) 
                         (gradF: float[] -> float[]) 
                         (x0: float[])
                         (options: OptimizationOptions) : OptimizationResult =
        
        let mutable x = Array.copy x0
        let mutable fx = f x
        let mutable lr = options.LearningRate
        let mutable converged = false
        let mutable iter = 0
        let mutable message = "Maximum iterations reached"
        
        while not converged && iter < options.MaxIter do
            let grad = gradF x
            
            // Line search: try current LR, reduce if needed
            let mutable stepAccepted = false
            let mutable attempts = 0
            let mutable xNew = x
            let mutable fxNew = fx
            
            while not stepAccepted && attempts < 10 do
                xNew <- Array.map2 (fun xi gi -
                    xi - lr * gi) x grad
                fxNew <- f xNew
                
                if fxNew < fx then
                    stepAccepted <- true
                    lr <- lr * 1.1  // Increase LR slightly
                else
                    lr <- lr * 0.5  // Reduce LR
                    attempts <- attempts + 1
            
            if abs (fxNew - fx) < options.Tol then
                converged <- true
                message <- "Converged (function value)"
            
            x <- xNew
            fx <- fxNew
            iter <- iter + 1
        
        {
            X = x
            Fun = fx
            Nit = iter
            Success = converged
            Message = message
        }

/// <summary>RMSprop optimizer.
/// </summary>
module RMSprop =
    
    /// <summary>Minimize using RMSprop.
    /// </summary>
    let minimize (f: float[] -> float) 
                 (gradF: float[] -> float[]) 
                 (x0: float[])
                 ?(rho: float) ?(epsilon: float)
                 (options: OptimizationOptions) : OptimizationResult =
        
        let rho = defaultArg rho 0.9
        let epsilon = defaultArg epsilon 1e-8
        let n = x0.Length
        
        let mutable x = Array.copy x0
        let mutable fx = f x
        let mutable v = Array.zeroCreate n  // Moving average of squared gradients
        let mutable converged = false
        let mutable iter = 0
        let mutable message = "Maximum iterations reached"
        
        while not converged && iter < options.MaxIter do
            let grad = gradF x
            
            // Update moving average
            for i = 0 to n - 1 do
                v.[i] <- rho * v.[i] + (1.0 - rho) * grad.[i] * grad.[i]
            
            // Update parameters
            let xNew = Array.init n (fun i -
                x.[i] - options.LearningRate * grad.[i] / (sqrt v.[i] + epsilon))
            
            let fxNew = f xNew
            
            if abs (fxNew - fx) < options.Tol then
                converged <- true
                message <- "Converged"
            
            x <- xNew
            fx <- fxNew
            iter <- iter + 1
        
        {
            X = x
            Fun = fx
            Nit = iter
            Success = converged
            Message = message
        }

/// <summary>Adam optimizer.
/// </summary>
module Adam =
    
    /// <summary>Minimize using Adam (Adaptive Moment Estimation).
    /// </summary>
    let minimize (f: float[] -> float) 
                 (gradF: float[] -> float[]) 
                 (x0: float[])
                 ?(beta1: float) ?(beta2: float) ?(epsilon: float)
                 (options: OptimizationOptions) : OptimizationResult =
        
        let beta1 = defaultArg beta1 0.9
        let beta2 = defaultArg beta2 0.999
        let epsilon = defaultArg epsilon 1e-8
        let n = x0.Length
        
        let mutable x = Array.copy x0
        let mutable fx = f x
        let mutable m = Array.zeroCreate n  // First moment
        let mutable v = Array.zeroCreate n  // Second moment
        let mutable converged = false
        let mutable iter = 0
        let mutable message = "Maximum iterations reached"
        
        while not converged && iter < options.MaxIter do
            let grad = gradF x
            iter <- iter + 1
            
            // Update biased first moment
            for i = 0 to n - 1 do
                m.[i] <- beta1 * m.[i] + (1.0 - beta1) * grad.[i]
            
            // Update biased second moment
            for i = 0 to n - 1 do
                v.[i] <- beta2 * v.[i] + (1.0 - beta2) * grad.[i] * grad.[i]
            
            // Bias correction
            let mHat = Array.init n (fun i -> m.[i] / (1.0 - pown beta1 iter))
            let vHat = Array.init n (fun i -> v.[i] / (1.0 - pown beta2 iter))
            
            // Update parameters
            let xNew = Array.init n (fun i -
                x.[i] - options.LearningRate * mHat.[i] / (sqrt vHat.[i] + epsilon))
            
            let fxNew = f xNew
            
            if abs (fxNew - fx) < options.Tol then
                converged <- true
                message <- "Converged"
            
            x <- xNew
            fx <- fxNew
        
        {
            X = x
            Fun = fx
            Nit = iter
            Success = converged
            Message = message
        }

/// <summary>AdamW optimizer (Adam with weight decay).
/// </summary>
module AdamW =
    
    /// <summary>Minimize using AdamW.
    /// </summary>
    let minimize (f: float[] -> float) 
                 (gradF: float[] -> float[]) 
                 (x0: float[])
                 ?(beta1: float) ?(beta2: float) ?(epsilon: float) ?(weightDecay: float)
                 (options: OptimizationOptions) : OptimizationResult =
        
        let beta1 = defaultArg beta1 0.9
        let beta2 = defaultArg beta2 0.999
        let epsilon = defaultArg epsilon 1e-8
        let weightDecay = defaultArg weightDecay 0.01
        let n = x0.Length
        
        let mutable x = Array.copy x0
        let mutable fx = f x
        let mutable m = Array.zeroCreate n
        let mutable v = Array.zeroCreate n
        let mutable converged = false
        let mutable iter = 0
        let mutable message = "Maximum iterations reached"
        
        while not converged && iter < options.MaxIter do
            let grad = gradF x
            iter <- iter + 1
            
            // Decoupled weight decay
            for i = 0 to n - 1 do
                x.[i] <- x.[i] * (1.0 - options.LearningRate * weightDecay)
            
            // Adam updates
            for i = 0 to n - 1 do
                m.[i] <- beta1 * m.[i] + (1.0 - beta1) * grad.[i]
                v.[i] <- beta2 * v.[i] + (1.0 - beta2) * grad.[i] * grad.[i]
            
            let mHat = Array.init n (fun i -> m.[i] / (1.0 - pown beta1 iter))
            let vHat = Array.init n (fun i -> v.[i] / (1.0 - pown beta2 iter))
            
            let xNew = Array.init n (fun i -
                x.[i] - options.LearningRate * mHat.[i] / (sqrt vHat.[i] + epsilon))
            
            let fxNew = f xNew
            
            if abs (fxNew - fx) < options.Tol then
                converged <- true
                message <- "Converged"
            
            x <- xNew
            fx <- fxNew
        
        {
            X = x
            Fun = fx
            Nit = iter
            Success = converged
            Message = message
        }

/// <summary>Simulated annealing optimizer.
/// For non-convex optimization problems.
/// </summary>
module SimulatedAnnealing =
    
    /// <summary>Minimize using simulated annealing.
    /// </summary>
    let minimize (f: float[] -> float) 
                 (x0: float[])
                 ?(initialTemp: float) ?(coolingRate: float)
                 (options: OptimizationOptions) : OptimizationResult =
        
        let initialTemp = defaultArg initialTemp 100.0
        let coolingRate = defaultArg coolingRate 0.95
        let n = x0.Length
        let rng = System.Random()
        
        let mutable x = Array.copy x0
        let mutable fx = f x
        let mutable bestX = Array.copy x
        let mutable bestFx = fx
        let mutable temp = initialTemp
        let mutable iter = 0
        let mutable message = "Maximum iterations reached"
        
        while iter < options.MaxIter do
            // Generate neighbor
            let xNew = Array.init n (fun i -
                x.[i] + (rng.NextDouble() - 0.5) * 2.0 * temp / initialTemp)
            
            let fxNew = f xNew
            let delta = fxNew - fx
            
            // Accept if better, or with probability based on temperature
            if delta < 0.0 || rng.NextDouble() < exp (-delta / temp) then
                x <- xNew
                fx <- fxNew
                
                if fx < bestFx then
                    bestX <- Array.copy x
                    bestFx <- fx
            
            // Cool down
            temp <- temp * coolingRate
            iter <- iter + 1
        
        {
            X = bestX
            Fun = bestFx
            Nit = iter
            Success = true
            Message = "Completed cooling schedule"
        }

/// <summary>Common test functions for optimization.
/// </summary>
module TestFunctions =
    
    /// <summary>Rosenbrock function (banana function).
    /// Global minimum at (1, 1, ..., 1) with value 0.
    /// </summary>
    let rosenbrock (x: float[]) : float =
        let mutable sum = 0.0
        for i = 0 to x.Length - 2 do
            sum <- sum + 100.0 * pown (x.[i+1] - x.[i]*x.[i]) 2 + pown (1.0 - x.[i]) 2
        sum
    
    /// <summary>Gradient of Rosenbrock function.
    /// </summary>
    let rosenbrockGrad (x: float[]) : float[] =
        let n = x.Length
        let grad = Array.zeroCreate n
        
        for i = 0 to n - 1 do
            if i = 0 then
                grad.[i] <- -400.0 * x.[i] * (x.[i+1] - x.[i]*x.[i]) - 2.0 * (1.0 - x.[i])
            elif i = n - 1 then
                grad.[i] <- 200.0 * (x.[i] - x.[i-1]*x.[i-1])
            else
                grad.[i] <- 200.0 * (x.[i] - x.[i-1]*x.[i-1]) - 400.0 * x.[i] * (x.[i+1] - x.[i]*x.[i]) - 2.0 * (1.0 - x.[i])
        
        grad
    
    /// <summary>Sphere function.
    /// Global minimum at origin with value 0.
    /// </summary>
    let sphere (x: float[]) : float =
        x |> Array.sumBy (fun xi -> xi * xi)
    
    /// <summary>Gradient of sphere function.
    /// </summary>
    let sphereGrad (x: float[]) : float[] =
        x |> Array.map (fun xi -> 2.0 * xi)
    
    /// <summary>Rastrigin function.
    /// Many local minima. Global minimum at origin with value 0.
    /// </summary>
    let rastrigin (x: float[]) : float =
        let n = float x.Length
        let A = 10.0
        A * n + (x |> Array.sumBy (fun xi -
            xi * xi - A * cos (2.0 * System.Math.PI * xi)))
