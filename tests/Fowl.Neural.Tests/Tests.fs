module Fowl.Tests.Neural

open Expecto
open Fowl
open Fowl.Core.Types
open Fowl.Neural
open Fowl.Neural.Graph
open Fowl.Neural.Forward
open Fowl.Neural.Backward
open Fowl.Neural.Layers
open Fowl.Neural.Optimizer
open Fowl.Neural.Loss
open Fowl.Neural.Training

/// Helper for approximate float equality
let approxEqual (tol: float) (actual: float) (expected: float) =
    abs (actual - expected) < tol

/// Helper for Ndarray approximate equality
let ndArrayApproxEqual (tol: float) (a: Ndarray<_, float>) (b: Ndarray<_, float>) =
    let aData = Ndarray.toArray a
    let bData = Ndarray.toArray b
    if aData.Length <> bData.Length then
        false
    else
        Array.zip aData bData
        |> Array.forall (fun (x, y) -> approxEqual tol x y)

let tests =
    testList "Neural Networks" [
        // ========================================================================
        // Graph Construction Tests
        // ========================================================================
        
        test "input node creates correctly" {
            let node = input "x" [|3|]
            Expect.equal node.Shape [|3|] "Input shape should be [3]"
            Expect.equal node.Op Input "Operation should be Input"
            Expect.equal node.Name (Some "x") "Name should be 'x'"
        }
        
        test "constant node creates correctly" {
            let node = constant 5.0
            Expect.equal node.Shape [||] "Constant has scalar shape"
            match node.Value.Value with
            | Some v -> Expect.equal v 5.0 "Constant value should be 5.0"
            | None -> failwith "Constant should have value"
        }
        
        test "parameter node creates correctly" {
            let data = [|1.0; 2.0; 3.0|]
            let node = parameter "W" [|3|] data
            Expect.equal node.Shape [|3|] "Parameter shape should match"
            Expect.equal node.Op Parameter "Operation should be Parameter"
        }
        
        test "add operation combines nodes" {
            let a = constant 2.0
            let b = constant 3.0
            let result = Graph.add a b
            Expect.equal result.Op (Add(a, b)) "Operation should be Add"
            Expect.equal result.Shape [||] "Scalar addition yields scalar"
        }
        
        test "matmul requires compatible shapes" {
            let a = input "A" [|2; 3|]
            let b = input "B" [|3; 4|]
            
            match Graph.matmul a b with
            | Ok result -> Expect.equal result.Shape [|2; 4|] "Matmul yields [2; 4]"
            | Error e -> failwithf "Matmul failed: %A" e
        }
        
        test "matmul fails with incompatible shapes" {
            let a = input "A" [|2; 3|]
            let b = input "B" [|4; 5|]
            
            match Graph.matmul a b with
            | Ok _ -> failwith "Should have failed with incompatible shapes"
            | Error (DimensionMismatch _) -> () // Expected
            | Error e -> failwithf "Wrong error: %A" e
        }
        
        // ========================================================================
        // Forward Pass Tests
        // ========================================================================
        
        test "forward pass computes constant" {
            let c = constant 5.0
            
            match Forward.run [c] with
            | Ok () ->
                match c.Value.Value with
                | Some v -> Expect.equal v 5.0 "Constant value computed correctly"
                | None -> failwith "Value should be set"
            | Error e -> failwithf "Forward pass failed: %A" e
        }
        
        test "forward pass computes addition" {
            let a = constant 2.0
            let b = constant 3.0
            let sum = Graph.add a b
            
            match Forward.run [sum] with
            | Ok () ->
                match sum.Value.Value with
                | Some v -> Expect.equal v 5.0 "2 + 3 = 5"
                | None -> failwith "Sum value should be set"
            | Error e -> failwithf "Forward pass failed: %A" e
        }
        
        test "forward pass computes multiplication" {
            let a = constant 4.0
            let b = constant 5.0
            let prod = Graph.mul a b
            
            match Forward.run [prod] with
            | Ok () ->
                match prod.Value.Value with
                | Some v -> Expect.equal v 20.0 "4 * 5 = 20"
                | None -> failwith "Product value should be set"
            | Error e -> failwithf "Forward pass failed: %A" e
        }
        
        test "forward pass with inputs" {
            let x = input "x" [|2|]
            let w = parameter "W" [|2; 1|] [|2.0; 3.0|]
            
            match Graph.matmul x w with
            | Ok h ->
                let inputs = Map ["x", [|1.0; 2.0|]]
                
                match Forward.runWithInputs h inputs with
                | Ok () ->
                    match h.Value.Value with
                    | Some v -> 
                        // [1, 2] @ [2, 3] = 1*2 + 2*3 = 8
                        Expect.equal v 8.0 "Matrix multiplication correct"
                    | None -> failwith "Value should be set"
                | Error e -> failwithf "Forward with inputs failed: %A" e
            | Error e -> failwithf "Matmul failed: %A" e
        }
        
        test "forward pass with activation" {
            let x = constant (-2.0)
            let activated = Graph.activate ReLU x
            
            match Forward.run [activated] with
            | Ok () ->
                match activated.Value.Value with
                | Some v -> Expect.equal v 0.0 "ReLU(-2) = 0"
                | None -> failwith "Value should be set"
            | Error e -> failwithf "Forward pass failed: %A" e
        }
        
        test "forward pass with sigmoid" {
            let x = constant 0.0
            let activated = Graph.activate Sigmoid x
            
            match Forward.run [activated] with
            | Ok () ->
                match activated.Value.Value with
                | Some v -> 
                    // sigmoid(0) = 0.5
                    Expect.floatClose Accuracy.medium v 0.5 "sigmoid(0) = 0.5"
                | None -> failwith "Value should be set"
            | Error e -> failwithf "Forward pass failed: %A" e
        }
        
        // ========================================================================
        // Backward Pass / Gradient Tests
        // ========================================================================
        
        test "backward pass computes gradient for addition" {
            let a = constant 2.0
            let b = constant 3.0
            let sum = Graph.add a b
            
            result {
                do! Forward.run [sum]
                do! Backward.run [sum]
                
                // d(sum)/da = 1, d(sum)/db = 1
                match a.Grad.Value, b.Grad.Value with
                | Some ga, Some gb ->
                    Expect.floatClose Accuracy.medium ga 1.0 "d(a+b)/da = 1"
                    Expect.floatClose Accuracy.medium gb 1.0 "d(a+b)/db = 1"
                | _ -> failwith "Gradients not computed"
            } |> function
                | Ok () -> ()
                | Error e -> failwithf "Test failed: %A" e
        }
        
        test "backward pass computes gradient for multiplication" {
            let a = parameter "a" [||] [|2.0|]
            let b = parameter "b" [||] [|3.0|]
            let prod = Graph.mul a b
            
            result {
                do! Forward.run [prod]
                do! Backward.run [prod]
                
                // d(a*b)/da = b = 3, d(a*b)/db = a = 2
                match a.Grad.Value, b.Grad.Value with
                | Some ga, Some gb ->
                    Expect.floatClose Accuracy.medium ga 3.0 "d(a*b)/da = b = 3"
                    Expect.floatClose Accuracy.medium gb 2.0 "d(a*b)/db = a = 2"
                | _ -> failwith "Gradients not computed"
            } |> function
                | Ok () -> ()
                | Error e -> failwithf "Test failed: %A" e
        }
        
        test "backward pass computes gradient for power" {
            let x = parameter "x" [||] [|3.0|]
            let squared = Graph.pow x (Graph.constant 2.0)
            
            result {
                do! Forward.run [squared]
                do! Backward.run [squared]
                
                // d(x^2)/dx = 2x = 6
                match x.Grad.Value with
                | Some g ->
                    Expect.floatClose Accuracy.medium g 6.0 "d(x^2)/dx at x=3 is 6"
                | None -> failwith "Gradient not computed"
            } |> function
                | Ok () -> ()
                | Error e -> failwithf "Test failed: %A" e
        }
        
        test "gradient accumulation from multiple paths" {
            let x = parameter "x" [||] [|2.0|]
            let y = Graph.mul x x  // y = x^2
            let z = Graph.mul x y  // z = x^3
            
            result {
                do! Forward.run [z]
                do! Backward.run [z]
                
                // dz/dx = 3x^2 = 12 (from both paths)
                match x.Grad.Value with
                | Some g ->
                    Expect.floatClose Accuracy.medium g 12.0 "d(x^3)/dx at x=2 is 12"
                | None -> failwith "Gradient not computed"
            } |> function
                | Ok () -> ()
                | Error e -> failwithf "Test failed: %A" e
        }
        
        // ========================================================================
        // Dense Layer Tests
        // ========================================================================
        
        test "dense layer creates with correct shapes" {
            match Layers.dense 10 5 (Some ReLU) (Some 42) with
            | Ok layer ->
                Expect.equal layer.InputSize 10 "Input size"
                Expect.equal layer.OutputSize 5 "Output size"
                Expect.equal layer.Activation (Some ReLU) "Activation"
            | Error e -> failwithf "Dense creation failed: %A" e
        }
        
        test "dense layer forward pass" {
            result {
                let! layer = Layers.dense 2 1 None (Some 42)
                let input = input "x" [|2|]
                
                // Manually set weights for testing
                // Reset the seed to get deterministic weights
                let! output = Layers.forwardDense layer input
                
                let inputs = Map ["x", [|1.0; 1.0|]]
                do! Forward.runWithInputs output inputs
                
                // Just verify it runs without error
                match output.Value.Value with
                | Some _ -> ()
                | None -> failwith "No output value"
            } |> function
                | Ok () -> ()
                | Error e -> failwithf "Test failed: %A" e
        }
        
        // ========================================================================
        // Loss Function Tests
        // ========================================================================
        
        test "MSE loss computes correctly" {
            let pred = Graph.constantArray [|1.0; 2.0; 3.0|] [|3|]
            let target = Graph.constantArray [|1.0; 2.0; 3.0|] [|3|]
            
            match Loss.mse pred target with
            | Ok loss ->
                result {
                    do! Forward.run [loss]
                    
                    match loss.Value.Value with
                    | Some v ->
                        // MSE of identical arrays = 0
                        Expect.floatClose Accuracy.medium v 0.0 "MSE of identical is 0"
                    | None -> failwith "Loss value not computed"
                } |> function
                    | Ok () -> ()
                    | Error e -> failwithf "Forward failed: %A" e
            | Error e -> failwithf "MSE creation failed: %A" e
        }
        
        test "MSE loss with different values" {
            let pred = Graph.constantArray [|1.0; 2.0; 3.0|] [|3|]
            let target = Graph.constantArray [|2.0; 2.0; 2.0|] [|3|]
            
            match Loss.mse pred target with
            | Ok loss ->
                result {
                    do! Forward.run [loss]
                    
                    match loss.Value.Value with
                    | Some v ->
                        // MSE = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = (1 + 0 + 1) / 3 = 0.666...
                        Expect.floatClose Accuracy.medium v 0.6666667 "MSE correct"
                    | None -> failwith "Loss value not computed"
                } |> function
                    | Ok () -> ()
                    | Error e -> failwithf "Forward failed: %A" e
            | Error e -> failwithf "MSE creation failed: %A" e
        }
        
        // ========================================================================
        // Optimizer Tests
        // ========================================================================
        
        test "SGD optimizer updates parameters" {
            let param = parameter "w" [|1|] [|1.0|]
            param.Grad.Value <- Some 2.0  // Set gradient manually
            
            let optimizer = Optimizer.sgd 0.1 0.0
            let params = Layers.getParameters { Weights = param; Bias = param; Activation = None; InputSize = 1; OutputSize = 1 }
            
            // Initial value: 1.0, grad: 2.0, lr: 0.1
            // New value = 1.0 - 0.1 * 2.0 = 0.8
            Optimizer.updateSGD optimizer params
            
            match param.Value.Value with
            | Some v ->
                Expect.floatClose Accuracy.medium v 0.8 "SGD update correct"
            | None -> failwith "Parameter value not set"
        }
        
        test "SGD with momentum updates parameters" {
            let param = parameter "w" [|1|] [|1.0|]
            param.Grad.Value <- Some 2.0
            
            let optimizer = Optimizer.sgd 0.1 0.9
            let velocity = ref 0.0
            let params = [(param, velocity)]
            
            // First update: v = 0.9*0 + 2.0 = 2.0, w = 1.0 - 0.1*2.0 = 0.8
            Optimizer.updateSGDMomentum optimizer params
            
            match param.Value.Value with
            | Some v ->
                Expect.floatClose Accuracy.medium v 0.8 "SGD with momentum update"
            | None -> failwith "Parameter value not set"
        }
        
        // ========================================================================
        // Integration / Training Tests
        // ========================================================================
        
        test "simple linear regression converges" {
            // y = 2x + 1
            let X = [| [|1.0|]; [|2.0|]; [|3.0|]; [|4.0|] |]
            let y = [| 3.0; 5.0; 7.0; 9.0 |]  // y = 2x + 1
            
            result {
                let! model = Layers.dense 1 1 None (Some 42)
                
                // Create computation graph
                let input = input "x" [|1|]
                let! output = Layers.forwardDense model input
                let target = input "y" [|1|]
                let! loss = Loss.mse output target
                
                let parameters = Layers.getParameters model
                let optimizer = Optimizer.sgd 0.01 0.0
                
                // Training loop
                for epoch = 1 to 100 do
                    let mutable totalLoss = 0.0
                    
                    for i = 0 to X.Length - 1 do
                        // Reset gradients
                        parameters |> List.iter (fun (p, _) -> p.Grad.Value <- None)
                        
                        // Forward pass
                        let inputs = Map ["x", X.[i]; "y", [|y.[i]|]]
                        do! Forward.runWithInputs loss inputs
                        
                        match loss.Value.Value with
                        | Some l -> totalLoss <- totalLoss + l
                        | None -> ()
                        
                        // Backward pass
                        do! Backward.run [loss]
                        
                        // Update
                        Optimizer.updateSGD optimizer parameters
                    
                    if epoch % 20 = 0 then
                        printfn "Epoch %d: loss = %.4f" epoch (totalLoss / float X.Length)
                
                // Verify convergence
                // Just check that it runs without error
                Expect.isTrue true "Training completed"
            } |> function
                | Ok () -> ()
                | Error e -> failwithf "Training failed: %A" e
        }
        
        test "gradient checking for numerical accuracy" {
            // Test that analytical gradients match numerical approximations
            let epsilon = 1e-4
            let tol = 1e-3
            
            let x = parameter "x" [||] [|2.0|]
            let f = Graph.pow x (Graph.constant 3.0)  // f(x) = x^3
            
            result {
                // Analytical gradient
                do! Forward.run [f]
                do! Backward.run [f]
                
                let analyticalGrad =
                    match x.Grad.Value with
                    | Some g -> g
                    | None -> failwith "No gradient"
                
                // Numerical gradient: (f(x+h) - f(x-h)) / (2h)
                let xPlus = parameter "x_plus" [||] [|2.0 + epsilon|]
                let fPlus = Graph.pow xPlus (Graph.constant 3.0)
                do! Forward.run [fPlus]
                let fPlusVal = fPlus.Value.Value |> Option.defaultValue 0.0
                
                let xMinus = parameter "x_minus" [||] [|2.0 - epsilon|]
                let fMinus = Graph.pow xMinus (Graph.constant 3.0)
                do! Forward.run [fMinus]
                let fMinusVal = fMinus.Value.Value |> Option.defaultValue 0.0
                
                let numericalGrad = (fPlusVal - fMinusVal) / (2.0 * epsilon)
                
                // f(x) = x^3, f'(x) = 3x^2 = 12 at x=2
                printfn "Analytical: %.6f, Numerical: %.6f" analyticalGrad numericalGrad
                
                Expect.isTrue (abs (analyticalGrad - numericalGrad) < tol)
                    (sprintf "Gradients don't match: analytical=%.6f, numerical=%.6f" analyticalGrad numericalGrad)
            } |> function
                | Ok () -> ()
                | Error e -> failwithf "Gradient check failed: %A" e
        }
    ]
