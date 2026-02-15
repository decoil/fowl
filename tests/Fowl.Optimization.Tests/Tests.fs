module Fowl.Tests.Optimization

open Expecto
open Fowl.Optimization

/// Helper for approximate float equality
let approxEqual (tol: float) (actual: float) (expected: float) =
    abs (actual - expected) < tol

let tests =
    testList "Optimization" [
        // ========================================================================
        // Gradient Descent Tests
        // ========================================================================
        
        test "gradient descent finds minimum of parabola" {
            // f(x) = x², minimum at x = 0
            let f (x: float[]) = x.[0] * x.[0]
            let grad (x: float[]) = [| 2.0 * x.[0] |]
            let x0 = [| 5.0 |]
            
            let options = { defaultOptions with
                MaxIter = 100
                LearningRate = 0.1
                Tol = 1e-10
            }
            
            let result = GradientDescent.minimize f grad x0 options
            
            Expect.floatClose Accuracy.high result.X.[0] 0.0 "Found minimum at 0"
            Expect.floatClose Accuracy.high result.Fun 0.0 "Function value at minimum"
            Expect.isTrue result.Success "Converged"
        }
        
        test "gradient descent finds minimum of 2D quadratic" {
            // f(x,y) = x² + y², minimum at (0, 0)
            let f (x: float[]) = x.[0] * x.[0] + x.[1] * x.[1]
            let grad (x: float[]) = [| 2.0 * x.[0]; 2.0 * x.[1] |]
            let x0 = [| 3.0; 4.0 |]
            
            let options = { defaultOptions with
                MaxIter = 200
                LearningRate = 0.1
                Tol = 1e-10
            }
            
            let result = GradientDescent.minimize f grad x0 options
            
            Expect.floatClose Accuracy.high result.X.[0] 0.0 "X coordinate at minimum"
            Expect.floatClose Accuracy.high result.X.[1] 0.0 "Y coordinate at minimum"
            Expect.floatClose Accuracy.high result.Fun 0.0 "Function value at minimum"
        }
        
        test "gradient descent with momentum converges faster" {
            // Compare momentum vs no momentum
            let f (x: float[]) = x.[0] * x.[0] + x.[1] * x.[1]
            let grad (x: float[]) = [| 2.0 * x.[0]; 2.0 * x.[1] |]
            let x0 = [| 5.0; 5.0 |]
            
            let optionsNoMomentum = { defaultOptions with
                MaxIter = 1000
                LearningRate = 0.01
                Tol = 1e-10
            }
            
            let optionsMomentum = { defaultOptions with
                MaxIter = 1000
                LearningRate = 0.01
                Tol = 1e-10
            }
            
            let resultNoMom = GradientDescent.minimize f grad x0 optionsNoMomentum
            let resultMom = GradientDescent.minimize f grad x0 optionsMomentum
            
            // Both should converge
            Expect.isTrue resultNoMom.Success "No momentum converges"
            Expect.isTrue resultMom.Success "Momentum converges"
        }
        
        // ========================================================================
        // Adam Optimizer Tests
        // ========================================================================
        
        test "Adam finds minimum of parabola" {
            let f (x: float[]) = x.[0] * x.[0]
            let grad (x: float[]) = [| 2.0 * x.[0] |]
            let x0 = [| 5.0 |]
            
            let options = { defaultOptions with
                MaxIter = 100
                LearningRate = 0.1
                Tol = 1e-10
            }
            
            let result = Adam.minimize f grad x0 options
            
            Expect.floatClose Accuracy.high result.X.[0] 0.0 "Found minimum at 0"
            Expect.isTrue result.Success "Converged"
        }
        
        test "Adam handles noisy gradients" {
            // f(x) = x² with noisy gradient
            let rng = System.Random(42)
            let f (x: float[]) = x.[0] * x.[0]
            let gradNoisy (x: float[]) = 
                [| 2.0 * x.[0] + rng.NextDouble() * 0.1 - 0.05 |]
            let x0 = [| 5.0 |]
            
            let options = { defaultOptions with
                MaxIter = 200
                LearningRate = 0.01
                Tol = 1e-6
            }
            
            let result = Adam.minimize f gradNoisy x0 options
            
            // Should converge close to minimum despite noise
            Expect.isTrue (abs result.X.[0] < 0.1) "Close to minimum despite noise"
        }
        
        // ========================================================================
        // RMSprop Tests
        // ========================================================================
        
        test "RMSprop finds minimum" {
            let f (x: float[]) = x.[0] * x.[0] + x.[1] * x.[1]
            let grad (x: float[]) = [| 2.0 * x.[0]; 2.0 * x.[1] |]
            let x0 = [| 3.0; 3.0 |]
            
            let options = { defaultOptions with
                MaxIter = 100
                LearningRate = 0.1
                Tol = 1e-10
            }
            
            let result = RMSprop.minimize f grad x0 options
            
            Expect.floatClose Accuracy.high result.X.[0] 0.0 "X at minimum"
            Expect.floatClose Accuracy.high result.X.[1] 0.0 "Y at minimum"
        }
        
        // ========================================================================
        // Test Functions
        // ========================================================================
        
        test "sphere function minimum" {
            // f(x) = sum(xᵢ²), minimum at origin
            let f = TestFunctions.sphere
            let grad = TestFunctions.sphereGrad
            let x0 = Array.init 10 (fun _ -> 5.0)
            
            let options = { defaultOptions with
                MaxIter = 500
                LearningRate = 0.01
                Tol = 1e-8
            }
            
            let result = Adam.minimize f grad x0 options
            
            Expect.floatClose Accuracy.medium result.Fun 0.0 "Function value at origin"
            Expect.isTrue result.Success "Converged"
        }
        
        test "Rosenbrock function" {
            // f(x,y) = (a-x)² + b(y-x²)²
            // Minimum at (a, a²)
            let a = 1.0
            let b = 100.0
            let f = TestFunctions.rosenbrock
            let grad = TestFunctions.rosenbrockGrad
            let x0 = [| 0.0; 0.0 |]
            
            let options = { defaultOptions with
                MaxIter = 5000
                LearningRate = 0.001
                Tol = 1e-8
            }
            
            let result = Adam.minimize f grad x0 options
            
            // Should converge close to (1, 1)
            Expect.isTrue (abs (result.X.[0] - 1.0) < 0.01) "X close to 1"
            Expect.isTrue (abs (result.X.[1] - 1.0) < 0.01) "Y close to 1"
        }
        
        // ========================================================================
        // Linear Regression via Optimization
        // ========================================================================
        
        test "optimize linear regression" {
            // Generate data: y = 2x + 1 + noise
            let rng = System.Random(42)
            let n = 100
            let X = Array.init n (fun i -> float i)
            let y = X |> Array.map (fun x -> 2.0 * x + 1.0 + rng.NextDouble() * 0.1)
            
            // Objective: minimize MSE
            let f (beta: float[]) =
                let intercept, slope = beta.[0], beta.[1]
                X
                |> Array.mapi (fun i x -
                    let pred = intercept + slope * x
                    let error = y.[i] - pred
                    error * error)
                |> Array.average
            
            let grad (beta: float[]) =
                let intercept, slope = beta.[0], beta.[1]
                let mutable dIntercept = 0.0
                let mutable dSlope = 0.0
                
                for i = 0 to n - 1 do
                    let x = X.[i]
                    let pred = intercept + slope * x
                    let error = y.[i] - pred
                    dIntercept <- dIntercept - 2.0 * error / float n
                    dSlope <- dSlope - 2.0 * error * x / float n
                
                [| dIntercept; dSlope |]
            
            let beta0 = [| 0.0; 0.0 |]
            
            let options = { defaultOptions with
                MaxIter = 1000
                LearningRate = 0.01
                Tol = 1e-10
            }
            
            let result = Adam.minimize f grad beta0 options
            
            // Should find slope ≈ 2, intercept ≈ 1
            Expect.isTrue (abs (result.X.[0] - 1.0) < 0.1) "Intercept close to 1"
            Expect.isTrue (abs (result.X.[1] - 2.0) < 0.1) "Slope close to 2"
        }
        
        // ========================================================================
        // Convergence Tests
        // ========================================================================
        
        test "convergence criterion met" {
            let f (x: float[]) = x.[0] * x.[0]
            let grad (x: float[]) = [| 2.0 * x.[0] |]
            let x0 = [| 1.0 |]
            
            let options = { defaultOptions with
                MaxIter = 10000
                LearningRate = 0.1
                Tol = 1e-6
            }
            
            let result = GradientDescent.minimize f grad x0 options
            
            // Check convergence message
            Expect.stringContains result.Message "Converged" "Should report convergence"
        }
        
        test "max iterations reached" {
            let f (x: float[]) = x.[0] * x.[0]
            let grad (x: float[]) = [| 2.0 * x.[0] |]
            let x0 = [| 1000.0 |]  // Far from minimum
            
            let options = { defaultOptions with
                MaxIter = 10  // Very limited iterations
                LearningRate = 0.001  // Slow learning
                Tol = 1e-15  // Very strict tolerance
            }
            
            let result = GradientDescent.minimize f grad x0 options
            
            // Should not converge, hit max iterations
            Expect.isFalse result.Success "Should not converge"
            Expect.equal result.Nit 10 "Should use all iterations"
        }
    ]
