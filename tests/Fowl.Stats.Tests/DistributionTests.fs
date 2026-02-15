module Fowl.Tests.Stats.Distributions

open Expecto
open Fowl
open Fowl.Core.Types
open Fowl.Stats

/// Helper for approximate float equality
let approxEqual (tol: float) (actual: float) (expected: float) =
    abs (actual - expected) < tol

let tests =
    testList "Distribution Tests" [
        // ========================================================================
        // Normal Distribution
        // ========================================================================
        
        test "normal pdf at mean is max" {
            match Distributions.Gaussian.pdf 0.0 1.0 0.0 with
            | Ok v ->
                // PDF at mean should be 1/sqrt(2*pi) ≈ 0.3989
                Expect.floatClose Accuracy.medium v 0.39894228 "PDF at mean"
            | Error e -> failtestf "PDF failed: %A" e
        }
        
        test "normal pdf decreases away from mean" {
            result {
                let! at0 = Distributions.Gaussian.pdf 0.0 1.0 0.0
                let! at1 = Distributions.Gaussian.pdf 0.0 1.0 1.0
                let! at2 = Distributions.Gaussian.pdf 0.0 1.0 2.0
                
                Expect.isTrue (at0 > at1) "PDF decreases at x=1"
                Expect.isTrue (at1 > at2) "PDF decreases at x=2"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        test "normal cdf at infinity approaches 1" {
            match Distributions.Gaussian.cdf 0.0 1.0 10.0 with
            | Ok v ->
                Expect.isTrue (v > 0.9999) "CDF(10) ≈ 1"
            | Error e -> failtestf "CDF failed: %A" e
        }
        
        test "normal cdf at -infinity approaches 0" {
            match Distributions.Gaussian.cdf 0.0 1.0 (-10.0) with
            | Ok v ->
                Expect.isTrue (v < 0.0001) "CDF(-10) ≈ 0"
            | Error e -> failtestf "CDF failed: %A" e
        }
        
        test "normal cdf at mean is 0.5" {
            match Distributions.Gaussian.cdf 0.0 1.0 0.0 with
            | Ok v ->
                Expect.floatClose Accuracy.medium v 0.5 "CDF at mean"
            | Error e -> failtestf "CDF failed: %A" e
        }
        
        test "normal ppf is inverse of cdf" {
            result {
                for p in [0.1; 0.25; 0.5; 0.75; 0.9] do
                    let! x = Distributions.Gaussian.ppf 0.0 1.0 p
                    let! pBack = Distributions.Gaussian.cdf 0.0 1.0 x
                    Expect.floatClose Accuracy.medium pBack p (sprintf "ppf/cdf roundtrip for p=%.2f" p)
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        test "normal mean and variance" {
            let mu = 5.0
            let sigma = 2.0
            
            result {
                let! mean = Distributions.Gaussian.mean mu sigma
                let! var = Distributions.Gaussian.variance mu sigma
                
                Expect.floatClose Accuracy.medium mean mu "Mean"
                Expect.floatClose Accuracy.medium var (sigma * sigma) "Variance"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        // ========================================================================
        // Gamma Distribution
        // ========================================================================
        
        test "gamma pdf is positive" {
            match Distributions.Gamma.pdf 2.0 1.0 1.0 with
            | Ok v ->
                Expect.isTrue (v > 0.0) "Gamma PDF is positive"
            | Error e -> failtestf "PDF failed: %A" e
        }
        
        test "gamma pdf at 0 is 0 for shape > 1" {
            match Distributions.Gamma.pdf 2.0 1.0 0.0 with
            | Ok v ->
                Expect.floatClose Accuracy.medium v 0.0 "PDF(0) = 0 for shape > 1"
            | Error e -> failtestf "PDF failed: %A" e
        }
        
        test "gamma mean is shape * scale" {
            let shape = 3.0
            let scale = 2.0
            
            match Distributions.Gamma.mean shape scale with
            | Ok mean ->
                Expect.floatClose Accuracy.medium mean (shape * scale) "Gamma mean"
            | Error e -> failtestf "Mean failed: %A" e
        }
        
        // ========================================================================
        // Beta Distribution
        // ========================================================================
        
        test "beta pdf integrates to 1" {
            // Approximate integral using trapezoid rule
            let a, b = 2.0, 3.0
            let n = 1000
            let dx = 1.0 / float n
            
            let integral =
                [0 .. n]
                |> List.map (fun i -
                    let x = float i * dx
                    match BetaDistribution.pdf a b x with
                    | Ok v -> v
                    | Error _ -> 0.0)
                |> List.windowed 2
                |> List.sumBy (fun [y1; y2] -> (y1 + y2) / 2.0 * dx)
            
            Expect.floatClose Accuracy.low integral 1.0 "Beta PDF integrates to 1"
        }
        
        test "beta mean is a / (a + b)" {
            let a, b = 2.0, 3.0
            
            match BetaDistribution.mean a b with
            | Ok mean ->
                Expect.floatClose Accuracy.medium mean (a / (a + b)) "Beta mean"
            | Error e -> failtestf "Mean failed: %A" e
        }
        
        // ========================================================================
        // Student's t-Distribution
        // ========================================================================
        
        test "t-distribution approaches normal as df increases" {
            result {
                // t with df=100 should be close to standard normal
                let! tPdf = StudentTDistribution.pdf 100.0 1.0
                let! normalPdf = Distributions.Gaussian.pdf 0.0 1.0 1.0
                
                // Should be close (within 5%)
                let diff = abs (tPdf - normalPdf)
                Expect.isTrue (diff < 0.05 * normalPdf) "t approaches normal"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        test "t-distribution has heavier tails than normal" {
            result {
                let! tPdf = StudentTDistribution.pdf 5.0 3.0
                let! normalPdf = Distributions.Gaussian.pdf 0.0 1.0 3.0
                
                // t-distribution should have higher PDF in tails
                Expect.isTrue (tPdf > normalPdf) "t has heavier tails"
            } |> function
                | Ok () -> ()
                | Error e -> failtestf "Test failed: %A" e
        }
        
        // ========================================================================
        // Chi-Square Distribution
        // ========================================================================
        
        test "chi-square mean equals df" {
            let df = 5.0
            
            match ChiSquareDistribution.mean df with
            | Ok mean ->
                Expect.floatClose Accuracy.medium mean df "Chi-square mean = df"
            | Error e -> failtestf "Mean failed: %A" e
        }
        
        test "chi-square variance equals 2*df" {
            let df = 5.0
            
            match ChiSquareDistribution.variance df with
            | Ok var ->
                Expect.floatClose Accuracy.medium var (2.0 * df) "Chi-square variance = 2*df"
            | Error e -> failtestf "Variance failed: %A" e
        }
        
        // ========================================================================
        // F-Distribution
        // ========================================================================
        
        test "f-distribution mean for d2 > 2" {
            let d1, d2 = 5.0, 10.0
            
            match FDistribution.mean d1 d2 with
            | Ok mean ->
                // Mean = d2 / (d2 - 2) for d2 > 2
                let expected = d2 / (d2 - 2.0)
                Expect.floatClose Accuracy.medium mean expected "F mean"
            | Error e -> failtestf "Mean failed: %A" e
        }
        
        // ========================================================================
        // Binomial Distribution
        // ========================================================================
        
        test "binomial pmf sums to 1" {
            let n, p = 10, 0.3
            
            let sum =
                [0 .. n]
                |> List.sumBy (fun k -
                    match BinomialDistribution.pmf n p k with
                    | Ok v -> v
                    | Error _ -> 0.0)
            
            Expect.floatClose Accuracy.medium sum 1.0 "Binomial PMF sums to 1"
        }
        
        test "binomial mean is n*p" {
            let n, p = 20, 0.4
            
            match BinomialDistribution.mean n p with
            | Ok mean ->
                Expect.floatClose Accuracy.medium mean (float n * p) "Binomial mean"
            | Error e -> failtestf "Mean failed: %A" e
        }
        
        // ========================================================================
        // Poisson Distribution
        // ========================================================================
        
        test "poisson pmf sums to 1" {
            let lambda = 2.5
            
            // Sum first 20 terms (should be sufficient for lambda=2.5)
            let sum =
                [0 .. 20]
                |> List.sumBy (fun k -
                    match PoissonDistribution.pmf lambda k with
                    | Ok v -> v
                    | Error _ -> 0.0)
            
            Expect.floatClose Accuracy.medium sum 1.0 "Poisson PMF sums to 1"
        }
        
        test "poisson mean equals lambda" {
            let lambda = 3.0
            
            match PoissonDistribution.mean lambda with
            | Ok mean ->
                Expect.floatClose Accuracy.medium mean lambda "Poisson mean"
            | Error e -> failtestf "Mean failed: %A" e
        }
        
        // ========================================================================
        // Geometric Distribution
        // ========================================================================
        
        test "geometric pmf sums to 1" {
            let p = 0.3
            
            // Sum first 100 terms
            let sum =
                [1 .. 100]
                |> List.sumBy (fun k -
                    match GeometricDistribution.pmf p k with
                    | Ok v -> v
                    | Error _ -> 0.0)
            
            // Should be close to 1 (geometric is infinite, but truncated)
            Expect.isTrue (sum > 0.99) "Geometric PMF sums close to 1"
        }
        
        test "geometric mean is 1/p" {
            let p = 0.4
            
            match GeometricDistribution.mean p with
            | Ok mean ->
                Expect.floatClose Accuracy.medium mean (1.0 / p) "Geometric mean"
            | Error e -> failtestf "Mean failed: %A" e
        }
        
        // ========================================================================
        // Random Sampling Tests
        // ========================================================================
        
        test "normal random sample has correct mean" {
            let mu, sigma = 5.0, 2.0
            let n = 10000
            
            // Generate samples
            let rng = RandomState.create 42
            let samples = 
                Array.init n (fun _ -
                    match Distributions.Gaussian.rvs mu sigma rng with
                    | Ok v -> v
                    | Error _ -> 0.0)
            
            let sampleMean = Array.average samples
            
            // Should be close to true mean (within 0.1 for n=10000)
            Expect.isTrue (abs (sampleMean - mu) < 0.1) 
                (sprintf "Sample mean %.3f close to true mean %.3f" sampleMean mu)
        }
        
        test "normal random sample has correct variance" {
            let mu, sigma = 0.0, 1.0
            let n = 10000
            
            let rng = RandomState.create 42
            let samples = 
                Array.init n (fun _ -
                    match Distributions.Gaussian.rvs mu sigma rng with
                    | Ok v -> v
                    | Error _ -> 0.0)
            
            let mean = Array.average samples
            let variance = samples |> Array.averageBy (fun x -> (x - mean) ** 2.0)
            
            // Should be close to true variance (within 0.1 for n=10000)
            let trueVar = sigma * sigma
            Expect.isTrue (abs (variance - trueVar) < 0.1)
                (sprintf "Sample variance %.3f close to true variance %.3f" variance trueVar)
        }
    ]
