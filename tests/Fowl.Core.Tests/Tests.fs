module Fowl.Tests.Core

open Expecto
open Fowl
open Fowl.Core.Types

let tests =
    testList "Ndarray Core" [
        test "zeros creates array of zeros" {
            match Ndarray.zeros<Float64> [|3; 4|] with
            | Ok arr ->
                Expect.equal (Ndarray.shape arr) [|3; 4|] "Shape should be [3; 4]"
                Expect.equal (Ndarray.numel arr) 12 "Should have 12 elements"
                
                // All elements should be 0
                for i = 0 to 2 do
                    for j = 0 to 3 do
                        match Ndarray.get arr [|i; j|] with
                        | Ok v -> Expect.equal v 0.0 (sprintf "Element [%d; %d] should be 0" i j)
                        | Error e -> failwithf "Get failed: %A" e
            | Error e -> failwithf "zeros failed: %A" e
        }
        
        test "ones creates array of ones" {
            match Ndarray.ones<Float64> [|2; 3|] with
            | Ok arr ->
                Expect.equal (Ndarray.shape arr) [|2; 3|] "Shape should be [2; 3]"
                
                for i = 0 to 1 do
                    for j = 0 to 2 do
                        match Ndarray.get arr [|i; j|] with
                        | Ok v -> Expect.equal v 1.0 (sprintf "Element [%d; %d] should be 1" i j)
                        | Error e -> failwithf "Get failed: %A" e
            | Error e -> failwithf "ones failed: %A" e
        }
        
        test "create fills array with value" {
            match Ndarray.create<Float64> [|2; 2|] 5.0 with
            | Ok arr ->
                for i = 0 to 1 do
                    for j = 0 to 1 do
                        match Ndarray.get arr [|i; j|] with
                        | Ok v -> Expect.equal v 5.0 (sprintf "Element [%d; %d] should be 5" i j)
                        | Error e -> failwithf "Get failed: %A" e
            | Error e -> failwithf "create failed: %A" e
        }
        
        test "get and set work correctly" {
            match Ndarray.zeros<Float64> [|3; 3|] with
            | Ok arr ->
                match Ndarray.set arr [|1; 2|] 42.0 with
                | Ok () ->
                    match Ndarray.get arr [|1; 2|] with
                    | Ok v ->
                        Expect.equal v 42.0 "Set value should be retrievable"
                        match Ndarray.get arr [|0; 0|] with
                        | Ok v2 -> Expect.equal v2 0.0 "Other values should be unchanged"
                        | Error e -> failwithf "Get failed: %A" e
                    | Error e -> failwithf "Get failed: %A" e
                | Error e -> failwithf "Set failed: %A" e
            | Error e -> failwithf "zeros failed: %A" e
        }
        
        test "linspace creates evenly spaced values" {
            match Ndarray.linspace 0.0 10.0 5 with
            | Ok arr ->
                let data = Ndarray.toArray arr
                Expect.equal data.[0] 0.0 "First element should be start"
                Expect.equal data.[4] 10.0 "Last element should be stop"
                Expect.equal data.[2] 5.0 "Middle element should be average"
            | Error e -> failwithf "linspace failed: %A" e
        }
        
        test "linspace validates num parameter" {
            match Ndarray.linspace 0.0 10.0 1 with
            | Ok _ -> failwith "Should have failed with num < 2"
            | Error (InvalidArgument _) -> ()  // Expected
            | Error e -> failwithf "Wrong error type: %A" e
        }
        
        test "arange creates values with step" {
            match Ndarray.arange 0.0 10.0 2.0 with
            | Ok arr ->
                let data = Ndarray.toArray arr
                Expect.equal data [|0.0; 2.0; 4.0; 6.0; 8.0|] "Should create stepped array"
            | Error e -> failwithf "arange failed: %A" e
        }
        
        test "reshape changes shape" {
            match Ndarray.zeros<Float64> [|2; 3|] with
            | Ok arr ->
                match Ndarray.reshape [|6|] arr with
                | Ok reshaped ->
                    Expect.equal (Ndarray.shape reshaped) [|6|] "Shape should be [6]"
                    Expect.equal (Ndarray.numel reshaped) 6 "Number of elements unchanged"
                | Error e -> failwithf "reshape failed: %A" e
            | Error e -> failwithf "zeros failed: %A" e
        }
        
        test "reshape validates element count" {
            match Ndarray.zeros<Float64> [|2; 3|] with
            | Ok arr ->
                match Ndarray.reshape [|5|] arr with
                | Ok _ -> failwith "Should have failed with mismatched element count"
                | Error (InvalidShape _) -> ()  // Expected
                | Error e -> failwithf "Wrong error type: %A" e
            | Error e -> failwithf "zeros failed: %A" e
        }
        
        test "map applies function to all elements" {
            match Ndarray.ones<Float64> [|2; 2|] with
            | Ok arr ->
                let doubled = Ndarray.map (fun x -> x * 2.0) arr
                let data = Ndarray.toArray doubled
                Expect.equal data [|2.0; 2.0; 2.0; 2.0|] "All elements should be doubled"
            | Error e -> failwithf "ones failed: %A" e
        }
        
        test "fold aggregates values" {
            match Ndarray.ones<Float64> [|3; 3|] with
            | Ok arr ->
                let sum = Ndarray.fold (+) 0.0 arr
                Expect.equal sum 9.0 "Sum of 9 ones should be 9"
            | Error e -> failwithf "ones failed: %A" e
        }
        
        test "element-wise add" {
            match Ndarray.ones<Float64> [|2; 2|], Ndarray.ones<Float64> [|2; 2|] with
            | Ok a, Ok b ->
                match Ndarray.add a b with
                | Ok c ->
                    let data = Ndarray.toArray c
                    Expect.equal data [|2.0; 2.0; 2.0; 2.0|] "Element-wise addition"
                | Error e -> failwithf "add failed: %A" e
            | Error e, _ | _, Error e -> failwithf "ones failed: %A" e
        }
        
        test "element-wise mul" {
            match Ndarray.create<Float64> [|2; 2|] 3.0, Ndarray.create<Float64> [|2; 2|] 4.0 with
            | Ok a, Ok b ->
                match Ndarray.mul a b with
                | Ok c ->
                    let data = Ndarray.toArray c
                    Expect.equal data [|12.0; 12.0; 12.0; 12.0|] "Element-wise multiplication"
                | Error e -> failwithf "mul failed: %A" e
            | Error e, _ | _, Error e -> failwithf "create failed: %A" e
        }
        
        test "shape mismatch returns error" {
            match Ndarray.ones<Float64> [|2; 2|], Ndarray.ones<Float64> [|3; 3|] with
            | Ok a, Ok b ->
                match Ndarray.add a b with
                | Ok _ -> failwith "Should have failed with shape mismatch"
                | Error (DimensionMismatch _) -> ()  // Expected
                | Error e -> failwithf "Wrong error type: %A" e
            | Error e, _ | _, Error e -> failwithf "ones failed: %A" e
        }
        
        test "division by zero returns error" {
            match Ndarray.ones<Float64> [|2; 2|], Ndarray.zeros<Float64> [|2; 2|] with
            | Ok a, Ok b ->
                match Ndarray.div a b with
                | Ok _ -> failwith "Should have failed with division by zero"
                | Error (InvalidArgument _) -> ()  // Expected
                | Error e -> failwithf "Wrong error type: %A" e
            | Error e, _ | _, Error e -> failwithf "create failed: %A" e
        }
    ]
