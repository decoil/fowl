/// Optimization Example
/// Gradient descent and simulated annealing

module OptimizationExample

open System
open Fowl
open Fowl.Core
open Fowl.Stats

/// Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
/// Global minimum at (a, a²)
let rosenbrock (a: float) (b: float) (x: float[]) : float =
    (a - x.[0]) ** 2.0 + b * (x.[1] - x.[0] ** 2.0) ** 2.0

/// Gradient of Rosenbrock function
let rosenbrockGradient (a: float) (b: float) (x: float[]) : float[] =
    let dx0 = -2.0 * (a - x.[0]) - 4.0 * b * x.[0] * (x.[1] - x.[0] ** 2.0)
    let dx1 = 2.0 * b * (x.[1] - x.[0] ** 2.0)
    [|dx0; dx1|]

/// Gradient descent optimizer
let gradientDescent (f: float[] -> float) 
                     (gradF: float[] -> float[]) 
                     (x0: float[]) 
                     (learningRate: float) 
                     (tolerance: float) 
                     (maxIter: int) : float[] * float * int =
    let mutable x = Array.copy x0
    let mutable fx = f x
    let mutable converged = false
    let mutable iter = 0
    
    while not converged && iter < maxIter do
        let grad = gradF x
        let xNew = Array.map2 (fun xi gi -> xi - learningRate * gi) x grad
        let fxNew = f xNew
        
        if abs (fxNew - fx) < tolerance then
            converged <- true
        
        x <- xNew
        fx <- fxNew
        iter <- iter + 1
    
    x, fx, iter

/// Simulated annealing for global optimization
let simulatedAnnealing (f: float[] -> float) 
                        (x0: float[]) 
                        (bounds: (float * float)[]) 
                        (initialTemp: float) 
                        (coolingRate: float) 
                        (maxIter: int) : float[] * float =
    let rng = Random()
    let mutable x = Array.copy x0
    let mutable fx = f x
    let mutable temp = initialTemp
    let mutable bestX = Array.copy x
    let mutable bestF = fx
    
    for iter = 0 to maxIter - 1 do
        // Generate neighbor
        let xNew = 
            x |
            Array.mapi (fun i xi -
                let (lo, hi) = bounds.[i]
                let step = (hi - lo) * 0.1 * (rng.NextDouble() - 0.5) * 2.0
                max lo (min hi (xi + step)))
        
        let fxNew = f xNew
        let delta = fxNew - fx
        
        // Accept or reject
        if delta < 0.0 || rng.NextDouble() < exp (-delta / temp) then
            x <- xNew
            fx <- fxNew
            
            if fx < bestF then
                bestX <- Array.copy x
                bestF >- fx
        
        // Cool down
        temp <- temp * coolingRate
    
    bestX, bestF

/// Beale function: another optimization test
let bealeFunction (x: float[]) : float =
    let x1, x2 = x.[0], x.[1]
    (1.5 - x1 + x1 * x2) ** 2.0 +
    (2.25 - x1 + x1 * x2 ** 2.0) ** 2.0 +
    (2.625 - x1 + x1 * x2 ** 3.0) ** 2.0

/// Run optimization examples
let runOptimization() : unit =
    printfn "=== Optimization Examples ==="
    printfn ""
    
    // Example 1: Rosenbrock function
    printfn "1. Rosenbrock Function Optimization"
    printfn "   f(x,y) = (1-x)² + 100(y-x²)²"
    printfn "   Global minimum: (1, 1)"
    printfn ""
    
    let x0 = [|-1.0; 2.0|]
    printfn "   Starting point: (%f, %f)" x0.[0] x0.[1]
    printfn "   Initial value: %f" (rosenbrock 1.0 100.0 x0)
    printfn ""
    
    // Gradient descent
    let gdSolution, gdValue, gdIters = gradientDescent 
        (rosenbrock 1.0 100.0) 
        (rosenbrockGradient 1.0 100.0)
        x0 0.001 1e-8 100000
    
    printfn "   Gradient Descent:"
    printfn "     Solution: (%.6f, %.6f)" gdSolution.[0] gdSolution.[1]
    printfn "     Value: %.10f" gdValue
    printfn "     Iterations: %d" gdIters
    printfn "     Distance to optimum: %.6f" 
            (sqrt ((gdSolution.[0] - 1.0) ** 2.0 + (gdSolution.[1] - 1.0) ** 2.0))
    printfn ""
    
    // Simulated annealing
    let bounds = [|-2.0, 2.0; -1.0, 3.0|]
    let saSolution, saValue = simulatedAnnealing
        (rosenbrock 1.0 100.0)
        x0 bounds 10.0 0.99 50000
    
    printfn "   Simulated Annealing:"
    printfn "     Solution: (%.6f, %.6f)" saSolution.[0] saSolution.[1]
    printfn "     Value: %.10f" saValue
    printfn "     Distance to optimum: %.6f"
            (sqrt ((saSolution.[0] - 1.0) ** 2.0 + (saSolution.[1] - 1.0) ** 2.0))
    printfn ""
    
    // Example 2: Beale function
    printfn "2. Beale Function Optimization"
    printfn "   Global minimum: (3, 0.5)"
    printfn ""
    
    let bealeX0 = [|0.0; 0.0|]
    let bealeBounds = [|-4.5, 4.5; -4.5, 4.5|]
    
    let bealeSA, bealeVal = simulatedAnnealing
        bealeFunction
        bealeX0 bealeBounds 5.0 0.995 30000
    
    printfn "   Simulated Annealing:"
    printfn "     Solution: (%.6f, %.6f)" bealeSA.[0] bealeSA.[1]
    printfn "     Value: %.10f" bealeVal
    printfn "     Distance to optimum: %.6f"
            (sqrt ((bealeSA.[0] - 3.0) ** 2.0 + (bealeSA.[1] - 0.5) ** 2.0))
    printfn ""
    
    // Example 3: Simple quadratic
    printfn "3. Quadratic Function: f(x) = x² + 2y²"
    printfn "   Global minimum: (0, 0)"
    printfn ""
    
    let quadratic (x: float[]) = x.[0] ** 2.0 + 2.0 * x.[1] ** 2.0
    let quadGrad (x: float[]) = [|2.0 * x.[0]; 4.0 * x.[1]|]
    
    let quadSolution, quadValue, quadIters = gradientDescent
        quadratic quadGrad [|-3.0; 2.0|] 0.1 1e-10 10000
    
    printfn "   Gradient Descent:"
    printfn "     Solution: (%.10f, %.10f)" quadSolution.[0] quadSolution.[1]
    printfn "     Value: %.2e" quadValue
    printfn "     Iterations: %d" quadIters
    printfn ""
    
    // Comparison table
    printfn "4. Algorithm Comparison"
    printfn ""
    printfn "   ┌─────────────────────┬───────────┬─────────────┬──────────────┐"
    printfn "   │ Function            │ Method    │ Iterations  │ Final Value  │"
    printfn "   ├─────────────────────┼───────────┼─────────────┼──────────────┤"
    printfn "   │ Rosenbrock          │ GD        │ %6d      │ %.6e │" gdIters gdValue
    printfn "   │ Rosenbrock          │ SA        │    50000    │ %.6e │" saValue
    printfn "   │ Quadratic           │ GD        │ %6d      │ %.6e │" quadIters quadValue
    printfn "   └─────────────────────┴───────────┴─────────────┴──────────────┘"
    printfn ""
    
    printfn "Notes:"
    printfn "  • Gradient Descent: Fast for convex, gets stuck in local minima"
    printfn "  • Simulated Annealing: Slower but finds global minimum"
    printfn "  • Rosenbrock has narrow valley - challenging for optimization"
    printfn ""
    
    printfn "=== Optimization Complete ==="

[<EntryPoint>]
let main argv =
    runOptimization()
    0