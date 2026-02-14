module Fowl.Tests.Core

open Expecto
open Fowl

let tests =
    testList "Ndarray Core" [
        test "zeros creates array of zeros" {
            let arr = Ndarray.zeros<Float64> [|3; 4|]
            Expect.equal (Ndarray.shape arr) [|3; 4|] "Shape should be [3; 4]"
            Expect.equal (Ndarray.numel arr) 12 "Should have 12 elements"
            
            // All elements should be 0
            for i = 0 to 2 do
                for j = 0 to 3 do
                    Expect.equal (Ndarray.get arr [|i; j|]) 0.0 (sprintf "Element [%d; %d] should be 0" i j)
        }
        
        test "ones creates array of ones" {
            let arr = Ndarray.ones<Float64> [|2; 3|]
            Expect.equal (Ndarray.shape arr) [|2; 3|] "Shape should be [2; 3]"
            
            for i = 0 to 1 do
                for j = 0 to 2 do
                    Expect.equal (Ndarray.get arr [|i; j|]) 1.0 (sprintf "Element [%d; %d] should be 1" i j)
        }
        
        test "create fills array with value" {
            let arr = Ndarray.create<Float64> [|2; 2|] 5.0
            for i = 0 to 1 do
                for j = 0 to 1 do
                    Expect.equal (Ndarray.get arr [|i; j|]) 5.0 (sprintf "Element [%d; %d] should be 5" i j)
        }
        
        test "get and set work correctly" {
            let arr = Ndarray.zeros<Float64> [|3; 3|]
            Ndarray.set arr [|1; 2|] 42.0
            Expect.equal (Ndarray.get arr [|1; 2|]) 42.0 "Set value should be retrievable"
            Expect.equal (Ndarray.get arr [|0; 0|]) 0.0 "Other values should be unchanged"
        }
        
        test "linspace creates evenly spaced values" {
            let arr = Ndarray.linspace 0.0 10.0 5
            let data = Ndarray.toArray arr
            Expect.equal data.[0] 0.0 "First element should be start"
            Expect.equal data.[4] 10.0 "Last element should be stop"
            Expect.equal data.[2] 5.0 "Middle element should be average"
        }
        
        test "arange creates values with step" {
            let arr = Ndarray.arange 0.0 10.0 2.0
            let data = Ndarray.toArray arr
            Expect.equal data [|0.0; 2.0; 4.0; 6.0; 8.0|] "Should create stepped array"
        }
        
        test "reshape changes shape" {
            let arr = Ndarray.zeros<Float64> [|2; 3|]
            let reshaped = Ndarray.reshape [|6|] arr
            Expect.equal (Ndarray.shape reshaped) [|6|] "Shape should be [6]"
            Expect.equal (Ndarray.numel reshaped) 6 "Number of elements unchanged"
        }
        
        test "map applies function to all elements" {
            let arr = Ndarray.ones<Float64> [|2; 2|]
            let doubled = Ndarray.map (fun x -> x * 2.0) arr
            let data = Ndarray.toArray doubled
            Expect.equal data [|2.0; 2.0; 2.0; 2.0|] "All elements should be doubled"
        }
        
        test "fold aggregates values" {
            let arr = Ndarray.ones<Float64> [|3; 3|]
            let sum = Ndarray.fold (+) 0.0 arr
            Expect.equal sum 9.0 "Sum of 9 ones should be 9"
        }
        
        test "element-wise add" {
            let a = Ndarray.ones<Float64> [|2; 2|]
            let b = Ndarray.ones<Float64> [|2; 2|]
            let c = Ndarray.add a b
            let data = Ndarray.toArray c
            Expect.equal data [|2.0; 2.0; 2.0; 2.0|] "Element-wise addition"
        }
        
        test "element-wise mul" {
            let a = Ndarray.create<Float64> [|2; 2|] 3.0
            let b = Ndarray.create<Float64> [|2; 2|] 4.0
            let c = Ndarray.mul a b
            let data = Ndarray.toArray c
            Expect.equal data [|12.0; 12.0; 12.0; 12.0|] "Element-wise multiplication"
        }
        
        test "scalar operations" {
            let a = Ndarray.ones<Float64> [|2; 2|]
            let b = Ndarray.addScalar a 5.0
            let data = Ndarray.toArray b
            Expect.equal data [|6.0; 6.0; 6.0; 6.0|] "Scalar addition"
        }
        
        test "shape mismatch throws" {
            let a = Ndarray.ones<Float64> [|2; 2|]
            let b = Ndarray.ones<Float64> [|3; 3|]
            Expect.throws (fun _ -> Ndarray.add a b |> ignore) "Should throw on shape mismatch"
        }
    ]
