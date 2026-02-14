namespace Fowl.Optimization

open System
open Fowl
open Fowl.Core.Types

/// <summary>Result from optimization.
/// </summary>type OptimizationResult = {
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
/// </summary>type OptimizationOptions = {
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
/// </summary>let defaultOptions = {
    MaxIter = 1000
    Tol = 1e-6
    LearningRate = 0.01
    Verbose = false
}

/// <summary>Gradient descent optimizer.
/// </summary>module GradientDescent =
    
    /// <summary>Minimize function using gradient descent.
    /// </summary>let minimize (f: float[] -> float) 
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
    /// </summary>let minimizeAdaptive (f: float[] -> float) 
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
/// </summary>module RMSprop =
    
    /// <summary>Minimize using RMSprop.
    /// </summary>let minimize (f: float[] -> float) 
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
            x <- Array.init n (fun i -
                x.[i] - options.LearningRate * grad.[i] / (sqrt v.[i] + epsilon))
            
            let fxNew = f x
            
            if abs (fxNew - fx) < options.Tol then
                converged <- true
                message <- "Converged"
            
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
/// </summary>module Adam =
    
    /// <summary>Minimize using Adam (Adaptive Moment Estimation).
    /// </summary>let minimize (f: float[] -> float) 
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
            let mHat = m |> Array.map (fun mi -> mi / (1.0 - beta1 ** float iter))
            let vHat = v |> Array.map (fun vi -> vi / (1.0 - beta2 ** float iter))
            
            // Update parameters
            x <- Array.init n (fun i -
                x.[i] - options.LearningRate * mHat.[i] / (sqrt vHat.[i] + epsilon))
            
            let fxNew = f x
            
            if abs (fxNew - fx) < options.Tol then
                converged <- true
                message <- "Converged"
            
            fx <- fxNew
        
        {
            X = x
            Fun = fx
            Nit = iter
            Success = converged
            Message = message
        }

/// <summary>AdamW optimizer (Adam with decoupled weight decay).
/// </summary>module AdamW =
    
    /// <summary>Minimize using AdamW.
    /// </summary>let minimize (f: float[] -> float) 
                     (gradF: float[] -> float[]) 
                     (x0: float[])
                     ?(weightDecay: float)
                     ?(beta1: float) ?(beta2: float) ?(epsilon: float)
                     (options: OptimizationOptions) : OptimizationResult =
        
        let weightDecay = defaultArg weightDecay 0.01
        let beta1 = defaultArg beta1 0.9
        let beta2 = defaultArg beta2 0.999
        let epsilon = defaultArg epsilon 1e-8
        let n = x0.Length
        
        let mutable x = Array.copy x0
        let mutable fx = f x
        let mutable m = Array.zeroCreate n
        let mutable v = Array.zeroCreate n
        let mutable converged = false
        let mutable iter = 0
        let mutable message = "Maximum iterations reached"
        
        while not converged && iter < options.MaxIter do
            // Decoupled weight decay
            x <- x |> Array.map (fun xi -> xi * (1.0 - options.LearningRate * weightDecay))
            
            let grad = gradF x
            iter <- iter + 1
            
            // Adam update
            for i = 0 to n - 1 do
                m.[i] <- beta1 * m.[i] + (1.0 - beta1) * grad.[i]
                v.[i] <- beta2 * v.[i] + (1.0 - beta2) * grad.[i] * grad.[i]
            
            let mHat = m |> Array.map (fun mi -> mi / (1.0 - beta1 ** float iter))
            let vHat = v |> Array.map (fun vi -> vi / (1.0 - beta2 ** float iter))
            
            x <- Array.init n (fun i -
                x.[i] - options.LearningRate * mHat.[i] / (sqrt vHat.[i] + epsilon))
            
            let fxNew = f x
            
            if abs (fxNew - fx) < options.Tol then
                converged <- true
                message <- "Converged"
            
            fx <- fxNew
        
        {
            X = x
            Fun = fx
            Nit = iter
            Success = converged
            Message = message
        }

/// <summary>Simulated Annealing for global optimization.
/// </summary>module SimulatedAnnealing =
    
    /// <summary>Minimize using simulated annealing.
    /// </summary>let minimize (f: float[] -> float) 
                     (x0: float[])
                     (bounds: (float * float)[])
                     ?(initialTemp: float) ?(coolingRate: float)
                     (options: OptimizationOptions) : OptimizationResult =
        
        let initialTemp = defaultArg initialTemp 100.0
        let coolingRate = defaultArg coolingRate 0.95
        let rng = Random()
        let n = x0.Length
        
        let mutable x = Array.copy x0
        let mutable fx = f x
        let mutable temp = initialTemp
        let mutable bestX = Array.copy x
        let mutable bestF = fx
        let mutable accepted = 0
        
        for iter = 0 to options.MaxIter - 1 do
            // Generate neighbor
            let xNew = 
                x |
                Array.mapi (fun i xi -
                    let (lo, hi) = bounds.[i]
                    let range = hi - lo
                    let step = range * 0.1 * (rng.NextDouble() - 0.5) * 2.0
                    max lo (min hi (xi + step)))
            
            let fxNew = f xNew
            let delta = fxNew - fx
            
            // Accept or reject
            if delta < 0.0 || rng.NextDouble() < exp (-delta / temp) then
                x <- xNew
                fx <- fxNew
                accepted <- accepted + 1
                
                if fx < bestF then
                    bestX <- Array.copy x
                    bestF >- fx
            
            // Cool down
            temp <- temp * coolingRate
            
            if options.Verbose && iter % 1000 = 0 then
                printfn "Iter %d: T=%.4f, f(x)=%.10f, best=%.10f" iter temp fx bestF
        
        {
            X = bestX
            Fun = bestF
            Nit = options.MaxIter
            Success = true
            Message = sprintf "Completed (acceptance rate: %.2f%%)" 
                     (100.0 * float accepted / float options.MaxIter)
        }

/// <summary>Test functions for optimization.
/// </summary>module TestFunctions =
    
    /// <summary>Rosenbrock function. Global minimum at (1, 1).
    /// </summary>let rosenbrock (x: float[]) : float =
        let a = 1.0
        let b = 100.0
        (a - x.[0]) ** 2.0 + b * (x.[1] - x.[0] ** 2.0) ** 2.0
    
    /// <summary>Gradient of Rosenbrock.
    /// </summary>let rosenbrockGrad (x: float[]) : float[] =
        let a = 1.0
        let b = 100.0
        let dx0 = -2.0 * (a - x.[0]) - 4.0 * b * x.[0] * (x.[1] - x.[0] ** 2.0)
        let dx1 = 2.0 * b * (x.[1] - x.[0] ** 2.0)
        [|dx0; dx1|]
    
    /// <summary>Sphere function. Global minimum at origin.
    /// </summary>let sphere (x: float[]) : float =
        x |> Array.sumBy (fun xi -> xi * xi)
    
    /// <summary>Gradient of sphere.
    /// </summary>let sphereGrad (x: float[]) : float[] =
        x |> Array.map (fun xi -> 2.0 * xi)
    
    /// <summary>Rastrigin function (many local minima).
    /// </summary>let rastrigin (x: float[]) : float =
        let A = 10.0
        let n = float x.Length
        A * n + Array.sumBy (fun xi -
            xi * xi - A * cos (2.0 * System.Math.PI * xi)) x