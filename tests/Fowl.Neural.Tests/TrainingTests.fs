module Fowl.Tests.Neural.Training

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

let tests =
    testList "Neural Network - Training" [
        // ========================================================================
        // Gradient Computation Tests
        // ========================================================================
        
        test "gradients computed for dense layer" {
            result {
                // Create simple network: input -> dense -> loss
                let! layer = Layers.dense 2 1 None (Some 42)
                
                let input = input "x" [|2|]
                let target = input "y" [|1|]
                
                let! output = Layers.forwardDense layer input
                let! loss = Loss.mse output target
                
                // Forward pass
                let inputs = Map ["x", [|1.0; 2.0|]; "y", [|3.0|]]
                do! Forward.runWithInputs loss inputs
                
                // Backward pass
                do! Backward.run [loss]
                
                // Check gradients exist
                let params = Layers.getParameters layer
                for (param, _) in params do
                    match param.Grad.Value with
                    | Some grad ->
                        Expect.isTrue (Array.length grad > 0) "Gradient has values"
                    | None ->
                        failtest "Gradient not computed"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        test "gradient descent reduces loss" {
            result {
                // Simple linear regression: y = 2x
                let! layer = Layers.dense 1 1 None (Some 42)
                
                let input = input "x" [|1|]
                let target = input "y" [|1|]
                
                let! output = Layers.forwardDense layer input
                let! loss = Loss.mse output target
                
                let params = Layers.getParameters layer
                let optimizer = Optimizer.sgd 0.1 0.0
                
                // Initial loss
                let inputs = Map ["x", [|1.0|]; "y", [|2.0|]]  // Target is 2
                do! Forward.runWithInputs loss inputs
                let initialLoss = loss.Value.Value |> Option.defaultValue 999.0
                
                // Train for 10 steps
                for _ in 1..10 do
                    // Reset gradients
                    params |> List.iter (fun (p, _) -> p.Grad.Value <- None)
                    
                    // Forward
                    do! Forward.runWithInputs loss inputs
                    
                    // Backward
                    do! Backward.run [loss]
                    
                    // Update
                    Optimizer.updateSGD optimizer params
                
                // Final loss
                params |> List.iter (fun (p, _) -> p.Grad.Value <- None)
                do! Forward.runWithInputs loss inputs
                let finalLoss = loss.Value.Value |> Option.defaultValue 999.0
                
                Expect.isTrue (finalLoss < initialLoss) "Loss decreased"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        // ========================================================================
        // Optimizer Tests
        // ========================================================================
        
        test "Adam optimizer updates parameters" {
            result {
                let! layer = Layers.dense 1 1 None (Some 42)
                
                let input = input "x" [|1|]
                let! output = Layers.forwardDense layer input
                let! loss = Loss.mse output (input "y" [|1|])
                
                let params = Layers.getParameters layer
                let adam = Optimizer.adam 0.01 0.9 0.999 1e-8
                
                // Get initial weight
                let initialWeight =
                    match layer.Weights.Value.Value with
                    | Some w -> w.[0]
                    | None -> 0.0
                
                // One training step
                let inputs = Map ["x", [|2.0|]; "y", [|4.0|]]
                do! Forward.runWithInputs loss inputs
                do! Backward.run [loss]
                Optimizer.updateAdam adam params
                
                // Weight should have changed
                let updatedWeight =
                    match layer.Weights.Value.Value with
                    | Some w -> w.[0]
                    | None -> 0.0
                
                Expect.notEqual updatedWeight initialWeight "Weight updated"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        test "SGD with momentum accumulates velocity" {
            result {
                let! layer = Layers.dense 1 1 None (Some 42)
                
                let input = input "x" [|1|]
                let! output = Layers.forwardDense layer input
                let! loss = Loss.mse output (input "y" [|1|])
                
                let params = Layers.getParameters layer
                let optimizer = Optimizer.sgd 0.01 0.9
                let velocities = params |> List.map (fun _ -> ref 0.0)
                
                // First step
                let inputs = Map ["x", [|1.0|]; "y", [|2.0|]]
                do! Forward.runWithInputs loss inputs
                do! Backward.run [loss]
                Optimizer.updateSGDMomentum optimizer (List.zip params velocities)
                
                // Velocity should be non-zero
                let hasVelocity = velocities |> List.exists (fun v -> !v <> 0.0)
                Expect.isTrue hasVelocity "Velocity accumulated"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        // ========================================================================
        // Loss Function Tests
        // ========================================================================
        
        test "MSE loss is zero when prediction equals target" {
            result {
                let pred = constantArray [|1.0; 2.0; 3.0|] [|3|]
                let target = constantArray [|1.0; 2.0; 3.0|] [|3|]
                
                let! loss = Loss.mse pred target
                
                do! Forward.run [loss]
                
                match loss.Value.Value with
                | Some l -> Expect.floatClose Accuracy.high l 0.0 "MSE is zero"
                | None -> failtest "No loss value"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        test "MSE loss increases with error" {
            result {
                let pred1 = constantArray [|1.0|] [|1|]
                let target1 = constantArray [|1.0|] [|1|]
                let! loss1 = Loss.mse pred1 target1
                
                let pred2 = constantArray [|2.0|] [|1|]
                let target2 = constantArray [|1.0|] [|1|]
                let! loss2 = Loss.mse pred2 target2
                
                do! Forward.run [loss1]
                do! Forward.run [loss2]
                
                let l1 = loss1.Value.Value |> Option.defaultValue 0.0
                let l2 = loss2.Value.Value |> Option.defaultValue 0.0
                
                Expect.isTrue (l2 > l1) "Larger error has larger loss"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        test "cross entropy loss for correct prediction is low" {
            result {
                // Perfect prediction: [1, 0, 0] for class 0
                let pred = constantArray [|1.0; 0.0; 0.0|] [|3|]
                let target = constantArray [|1.0; 0.0; 0.0|] [|3|]
                
                let! loss = Loss.crossEntropy pred target
                
                do! Forward.run [loss]
                
                match loss.Value.Value with
                | Some l ->
                    // Cross entropy should be near zero
                    Expect.isTrue (l < 0.01) "Loss is small for correct prediction"
                | None -> failtest "No loss value"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        // ========================================================================
        // End-to-End Training Tests
        // ========================================================================
        
        test "learns XOR function" {
            result {
                // XOR is not linearly separable, needs hidden layer
                let! hidden = Layers.dense 2 4 (Some Sigmoid) (Some 42)
                let! output = Layers.dense 4 1 (Some Sigmoid) (Some 43)
                
                let input = input "x" [|2|]
                let target = input "y" [|1|]
                
                let! h = Layers.forwardDense hidden input
                let! out = Layers.forwardDense output h
                let! loss = Loss.mse out target
                
                let params = 
                    Layers.getParameters hidden @ 
                    Layers.getParameters output
                
                let optimizer = Optimizer.adam 0.5 0.9 0.999 1e-8
                
                // XOR training data
                let data = [|
                    ([|0.0; 0.0|], [|0.0|])
                    ([|0.0; 1.0|], [|1.0|])
                    ([|1.0; 0.0|], [|1.0|])
                    ([|1.0; 1.0|], [|0.0|])
                |]
                
                // Train for 500 epochs
                for epoch = 1 to 500 do
                    let mutable totalLoss = 0.0
                    
                    for (x, y) in data do
                        params |> List.iter (fun (p, _) -> p.Grad.Value <- None)
                        
                        let inputs = Map ["x", x; "y", y]
                        do! Forward.runWithInputs loss inputs
                        do! Backward.run [loss]
                        Optimizer.updateAdam optimizer params
                        
                        totalLoss <- totalLoss + (loss.Value.Value |> Option.defaultValue 0.0)
                    
                    if epoch % 100 = 0 then
                        printfn "Epoch %d: loss = %.4f" epoch (totalLoss / 4.0)
                
                // Test predictions
                let mutable correct = 0
                for (x, y) in data do
                    params |> List.iter (fun (p, _) -> p.Grad.Value <- None)
                    let inputs = Map ["x", x]
                    do! Forward.runWithInputs out inputs
                    
                    let pred = out.Value.Value |> Option.map (fun v -> if v.[0] > 0.5 then 1.0 else 0.0) |> Option.defaultValue -1.0
                    if abs(pred - y.[0]) < 0.1 then
                        correct <- correct + 1
                
                printfn "XOR accuracy: %d/4" correct
                Expect.isTrue (correct >= 3) "Learned XOR (at least 3/4 correct)"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        test "batch training reduces loss" {
            result {
                let! layer = Layers.dense 1 1 None (Some 42)
                
                let input = input "x" [|1|]
                let target = input "y" [|1|]
                let! output = Layers.forwardDense layer input
                let! loss = Loss.mse output target
                
                let params = Layers.getParameters layer
                let optimizer = Optimizer.sgd 0.01 0.0
                
                // Batch of data: y = 2x
                let batch = [|
                    for i in 1..10 ->
                        ([|float i|], [|float (2 * i)|])
                |]
                
                // Compute initial average loss
                let mutable initialLoss = 0.0
                for (x, y) in batch do
                    params |> List.iter (fun (p, _) -> p.Grad.Value <- None)
                    let inputs = Map ["x", x; "y", y]
                    do! Forward.runWithInputs loss inputs
                    initialLoss <- initialLoss + (loss.Value.Value |> Option.defaultValue 0.0)
                initialLoss <- initialLoss / float batch.Length
                
                // Train for 50 epochs with batch
                for epoch = 1 to 50 do
                    let mutable epochLoss = 0.0
                    
                    // Accumulate gradients over batch
                    params |> List.iter (fun (p, _) -> p.Grad.Value <- None)
                    
                    for (x, y) in batch do
                        let inputs = Map ["x", x; "y", y]
                        do! Forward.runWithInputs loss inputs
                        do! Backward.run [loss]
                        epochLoss <- epochLoss + (loss.Value.Value |> Option.defaultValue 0.0)
                    
                    // Update once per batch
                    Optimizer.updateSGD optimizer params
                
                // Compute final average loss
                let mutable finalLoss = 0.0
                for (x, y) in batch do
                    params |> List.iter (fun (p, _) -> p.Grad.Value <- None)
                    let inputs = Map ["x", x; "y", y]
                    do! Forward.runWithInputs loss inputs
                    finalLoss <- finalLoss + (loss.Value.Value |> Option.defaultValue 0.0)
                finalLoss <- finalLoss / float batch.Length
                
                printfn "Initial loss: %.4f, Final loss: %.4f" initialLoss finalLoss
                Expect.isTrue (finalLoss < initialLoss) "Batch training reduced loss"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
    ]
