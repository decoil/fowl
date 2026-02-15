module Fowl.Tests.Neural.Convolution

open Expecto
open Fowl
open Fowl.Core.Types
open Fowl.Neural
open Fowl.Neural.ConvLayers

let tests =
    testList "Neural Network - Convolution" [
        // ========================================================================
        // Conv2D Tests
        // ========================================================================
        
        test "conv2d creates with correct dimensions" {
            let inChannels = 3
            let outChannels = 16
            let kernelSize = 3
            
            match Conv2D.create inChannels outChannels kernelSize Padding.Same (Some 42) with
            | Ok layer ->
                Expect.equal layer.InChannels 3 "Input channels"
                Expect.equal layer.OutChannels 16 "Output channels"
                Expect.equal layer.KernelSize 3 "Kernel size"
            | Error e -> failtestf "Failed to create layer: %A" e
        }
        
        test "conv2d forward pass preserves spatial dimensions with Same padding" {
            result {
                let! layer = Conv2D.create 3 16 3 Padding.Same (Some 42)
                
                // Input: batch=1, channels=3, height=32, width=32
                let input = Graph.input "input" [|1; 32; 32; 3|]
                
                let! output = Conv2D.forward layer input
                
                // Output should have same spatial dimensions
                Expect.equal output.Shape.[0] 1 "Batch size preserved"
                Expect.equal output.Shape.[1] 32 "Height preserved"
                Expect.equal output.Shape.[2] 32 "Width preserved"
                Expect.equal output.Shape.[3] 16 "Output channels"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        test "conv2d forward pass reduces dimensions with Valid padding" {
            result {
                let! layer = Conv2D.create 3 16 3 Padding.Valid (Some 42)
                
                // Input: batch=1, height=32, width=32, channels=3
                let input = Graph.input "input" [|1; 32; 32; 3|]
                
                let! output = Conv2D.forward layer input
                
                // Output: 32 - 3 + 1 = 30
                Expect.equal output.Shape.[1] 30 "Height reduced"
                Expect.equal output.Shape.[2] 30 "Width reduced"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        test "conv2d with stride reduces output size" {
            result {
                // Create layer with stride 2
                let! layer = Conv2D.create 3 16 3 Padding.Same (Some 42)
                let layerWithStride = { layer with Stride = 2 }
                
                let input = Graph.input "input" [|1; 32; 32; 3|]
                let! output = Conv2D.forward layerWithStride input
                
                // With stride 2, output should be half size
                Expect.equal output.Shape.[1] 16 "Height halved"
                Expect.equal output.Shape.[2] 16 "Width halved"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        // ========================================================================
        // Pooling Tests
        // ========================================================================
        
        test "max pooling reduces dimensions" {
            result {
                let pool = Pool2D.create MaxPool 2 2 Padding.Valid
                
                let input = Graph.input "input" [|1; 32; 32; 16|]
                let! output = Pool2D.forward pool input
                
                // 32 / 2 = 16
                Expect.equal output.Shape.[1] 16 "Height halved"
                Expect.equal output.Shape.[2] 16 "Width halved"
                Expect.equal output.Shape.[3] 16 "Channels preserved"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        test "average pooling reduces dimensions" {
            result {
                let pool = Pool2D.create AvgPool 2 2 Padding.Valid
                
                let input = Graph.input "input" [|1; 32; 32; 16|]
                let! output = Pool2D.forward pool input
                
                Expect.equal output.Shape.[1] 16 "Height halved"
                Expect.equal output.Shape.[2] 16 "Width halved"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        // ========================================================================
        // Batch Normalization Tests
        // ========================================================================
        
        test "batch norm creates with correct dimensions" {
            let numFeatures = 64
            let bn = BatchNorm2D.create numFeatures
            
            Expect.equal bn.NumFeatures 64 "Number of features"
            Expect.equal bn.Gamma.Shape.[0] 64 "Gamma shape"
            Expect.equal bn.Beta.Shape.[0] 64 "Beta shape"
        }
        
        test "batch norm normalizes during training" {
            result {
                let bn = BatchNorm2D.create 16
                
                // Input with mean 5, std 2
                let input = Graph.input "input" [|4; 8; 8; 16|]
                
                // Set running statistics
                bn.RunningMean <- Array.init 16 (fun _ -> 5.0)
                bn.RunningVar <- Array.init 16 (fun _ -> 4.0)
                
                let! output = BatchNorm2D.forward bn input (isTraining=false)
                
                // Output should be normalized
                Expect.equal output.Shape.[3] 16 "Channels preserved"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        // ========================================================================
        // Flatten Tests
        // ========================================================================
        
        test "flatten reshapes correctly" {
            result {
                let flatten = { StartDim = 1; EndDim = -1 }
                
                // Input: [batch, channels, height, width]
                let input = Graph.input "input" [|4; 16; 8; 8|]
                
                let! output = Flatten.forward flatten input
                
                // Output: [batch, channels*height*width]
                Expect.equal output.Shape.[0] 4 "Batch preserved"
                Expect.equal output.Shape.[1] 1024 "Flattened dimensions"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        // ========================================================================
        // Integration Tests
        // ========================================================================
        
        test "simple CNN forward pass" {
            result {
                // Conv -> ReLU -> Pool -> Flatten
                let! conv1 = Conv2D.create 3 32 3 Padding.Same (Some 42)
                let pool1 = Pool2D.create MaxPool 2 2 Padding.Valid
                let! conv2 = Conv2D.create 32 64 3 Padding.Same (Some 43)
                let pool2 = Pool2D.create MaxPool 2 2 Padding.Valid
                let flatten = { StartDim = 1; EndDim = -1 }
                
                let input = Graph.input "input" [|1; 32; 32; 3|]
                
                // Forward pass
                let! h1 = Conv2D.forward conv1 input
                let h1_activated = Graph.activate ReLU h1
                let! h2 = Pool2D.forward pool1 h1_activated
                
                let! h3 = Conv2D.forward conv2 h2
                let h3_activated = Graph.activate ReLU h3
                let! h4 = Pool2D.forward pool2 h3_activated
                
                let! output = Flatten.forward flatten h4
                
                // 32 -> 16 -> 8, 64 channels
                // 8 * 8 * 64 = 4096
                Expect.equal output.Shape.[1] 4096 "Flattened output"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        test "convolution output values are reasonable" {
            result {
                let! layer = Conv2D.create 1 1 3 Padding.Same (Some 42)
                
                // Create constant input of ones
                let ones = Graph.constantArray (Array.create 64 1.0) [|1; 8; 8; 1|]
                
                let! output = Conv2D.forward layer ones
                
                // Output should be finite numbers
                match Forward.run [output] with
                | Ok () ->
                    match output.Value.Value with
                    | Some values ->
                        Expect.isTrue (Array.forall (fun x -> not (Double.IsNaN x)) values) "No NaN values"
                        Expect.isTrue (Array.forall (fun x -> not (Double.IsInfinity x)) values) "No Inf values"
                    | None -> failtest "No output value"
                | Error e -> failtestf "Forward pass failed: %A" e
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
    ]
