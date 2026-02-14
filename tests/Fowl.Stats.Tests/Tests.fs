module Fowl.Tests.Stats

open Expecto
open Fowl
open Fowl.Core.Types
open Fowl.Stats

let tests =
    testList "Statistics" [
        test "mean calculates average" {
            match Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0; 5.0|] [|5|] with
            | Ok arr ->
                let m = Descriptive.mean arr
                Expect.floatClose Accuracy.medium m 3.0 "Mean should be 3.0"
            | Error e -> failwithf "ofArray failed: %A" e
        }
        
        test "var calculates variance" {
            match Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0; 5.0|] [|5|] with
            | Ok arr ->
                let v = Descriptive.var arr
                // Population variance: 2.0
                Expect.floatClose Accuracy.medium v 2.0 "Variance should be 2.0"
            | Error e -> failwithf "ofArray failed: %A" e
        }
        
        test "std calculates standard deviation" {
            match Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0; 5.0|] [|5|] with
            | Ok arr ->
                let s = Descriptive.std arr
                // sqrt(2.0) ≈ 1.414
                Expect.floatClose Accuracy.medium s (sqrt 2.0) "Std should be sqrt(2)"
            | Error e -> failwithf "ofArray failed: %A" e
        }
        
        test "median finds middle value (odd count)" {
            match Ndarray.ofArray [|1.0; 2.0; 6.0; 4.0; 3.0|] [|5|] with
            | Ok arr ->
                let m = Descriptive.median arr
                Expect.equal m 3.0 "Median should be 3.0"
            | Error e -> failwithf "ofArray failed: %A" e
        }
        
        test "median averages middle values (even count)" {
            match Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0|] [|4|] with
            | Ok arr ->
                let m = Descriptive.median arr
                Expect.equal m 2.5 "Median should be 2.5"
            | Error e -> failwithf "ofArray failed: %A" e
        }
        
        test "Gaussian pdf calculates correctly" {
            match Distributions.Gaussian.pdf 0.0 1.0 0.0 with
            | Ok v ->
                // PDF at mean: 1/sqrt(2*pi) ≈ 0.399
                Expect.floatClose Accuracy.medium v 0.39894228 "PDF at mean"
            | Error e -> failwithf "pdf failed: %A" e
        }
        
        test "Gaussian pdf validates sigma" {
            match Distributions.Gaussian.pdf 0.0 -1.0 0.0 with
            | Ok _ -> failwith "Should have failed with negative sigma"
            | Error (InvalidArgument _) -> ()  // Expected
            | Error e -> failwithf "Wrong error type: %A" e
        }
        
        test "Uniform pdf within bounds" {
            match Distributions.Uniform.pdf 0.0 10.0 5.0 with
            | Ok v ->
                Expect.equal v 0.1 "PDF within bounds should be 0.1"
            | Error e -> failwithf "pdf failed: %A" e
        }
        
        test "Uniform pdf outside bounds" {
            match Distributions.Uniform.pdf 0.0 10.0 15.0 with
            | Ok v ->
                Expect.equal v 0.0 "PDF outside bounds should be 0"
            | Error e -> failwithf "pdf failed: %A" e
        }
        
        test "covariance calculation" {
            match Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0; 5.0|] [|5|], 
                  Ndarray.ofArray [|2.0; 4.0; 6.0; 8.0; 10.0|] [|5|] with
            | Ok x, Ok y ->
                let cov = Correlation.covariance x y
                // Perfect linear relationship
                Expect.isGreaterThan cov 0.0 "Covariance should be positive"
            | Error e, _ | _, Error e -> failwithf "ofArray failed: %A" e
        }
        
        test "pearson correlation for perfect correlation" {
            match Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0; 5.0|] [|5|],
                  Ndarray.ofArray [|2.0; 4.0; 6.0; 8.0; 10.0|] [|5|] with
            | Ok x, Ok y ->
                let corr = Correlation.pearsonCorrelation x y
                Expect.floatClose Accuracy.medium corr 1.0 "Perfect positive correlation"
            | Error e, _ | _, Error e -> failwithf "ofArray failed: %A" e
        }
        
        test "random generation with state" {
            let state1 = Random.init(42)
            let arr1, _ = Distributions.Gaussian.rvsWithState 0.0 1.0 [|100|] state1
            
            let state2 = Random.init(42)
            let arr2, _ = Distributions.Gaussian.rvsWithState 0.0 1.0 [|100|] state2
            
            // Same seed should produce same values
            let data1 = Ndarray.toArray arr1
            let data2 = Ndarray.toArray arr2
            
            Expect.sequenceEqual data1 data2 "Same seed should produce same values"
        }
    ]
