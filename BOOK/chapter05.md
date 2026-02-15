# Chapter 5: Statistics and Probability

## 5.1 Descriptive Statistics

### Central Tendency

```fsharp
open Fowl
open Fowl.Stats

let data = [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0|]

// Mean
let mean = Descriptive.mean data  // 5.5

// Median
let median = Descriptive.median data  // 5.5

// Mode (for discrete data)
let discreteData = [|1; 2; 2; 3; 3; 3; 4|] |> Array.map float
let mode = Descriptive.mode discreteData  // 3.0
```

### Dispersion

```fsharp
// Variance (population)
let var = Descriptive.var data

// Standard deviation
let std = Descriptive.std data

// Range
let range = Descriptive.range data

// Interquartile range
let iqr = DescriptiveExtended.iqr data |> unwrap

// Median absolute deviation (robust)
let mad = DescriptiveExtended.mad data |> unwrap
```

### Shape

```fsharp
// Skewness (asymmetry)
let skew = Descriptive.skewness data

// Kurtosis (tail heaviness)
let kurt = Descriptive.kurtosis data

// Percentiles
let p25 = Descriptive.percentile data 0.25 |> unwrap
let p75 = Descriptive.percentile data 0.75 |> unwrap
```

## 5.2 Probability Distributions

### Continuous Distributions

#### Normal Distribution

```fsharp
open Fowl.Stats.Distributions

let mu = 0.0
let sigma = 1.0

// Probability density function
let pdf = Gaussian.pdf mu sigma 1.5 |> unwrap

// Cumulative distribution function
let cdf = Gaussian.cdf mu sigma 1.5 |> unwrap

// Percent point function (inverse CDF)
let ppf = Gaussian.ppf mu sigma 0.95 |> unwrap  // ~1.645

// Random sampling
let rng = RandomState.create 42
let sample = Gaussian.rvs mu sigma rng |> unwrap
let samples = RandomState.randn rng [|1000|]

// Statistical properties
let mean = Gaussian.mean mu sigma |> unwrap
let variance = Gaussian.variance mu sigma |> unwrap
```

#### Gamma Distribution

```fsharp
let shape = 2.0
let scale = 2.0

let pdf = Gamma.pdf shape scale 3.0 |> unwrap
let cdf = Gamma.cdf shape scale 3.0 |> unwrap
let sample = Gamma.rvs shape scale rng |> unwrap
```

#### Beta Distribution

```fsharp
let a = 2.0
let b = 5.0

// Useful for modeling proportions
let pdf = BetaDistribution.pdf a b 0.5 |> unwrap
```

#### Student's t-Distribution

```fsharp
let df = 10.0

// Heavier tails than normal
let pdf = StudentTDistribution.pdf df 2.0 |> unwrap

// Confidence intervals
let t95 = StudentTDistribution.ppf df 0.975 |> unwrap
```

### Discrete Distributions

#### Binomial Distribution

```fsharp
let n = 10
let p = 0.3

// Probability mass function
let pmf = BinomialDistribution.pmf n p 3 |> unwrap

// Cumulative probability
let cdf = BinomialDistribution.cdf n p 3 |> unwrap

// Random samples
let sample = BinomialDistribution.rvs n p rng |> unwrap
```

#### Poisson Distribution

```fsharp
let lambda = 3.0

// For count data
let pmf = PoissonDistribution.pmf lambda 2 |> unwrap
let sample = PoissonDistribution.rvs lambda rng |> unwrap
```

## 5.3 Hypothesis Testing

### One-Sample t-Test

```fsharp
open Fowl.Stats.HypothesisTests

// Test if sample mean differs from known value
let sample = [|5.2; 4.8; 5.5; 5.0; 5.3|]
let popMean = 5.0
let alpha = 0.05

let! result = ttestOneSample sample popMean alpha

// Interpret results
printfn "t-statistic: %f" result.Statistic
printfn "p-value: %f" result.PValue
printfn "Significant: %b" result.Significant
```

### Two-Sample t-Test

```fsharp
let sample1 = [|5.2; 4.8; 5.5; 5.0; 5.3|]
let sample2 = [|4.5; 4.2; 4.8; 4.3; 4.6|]

// Welch's t-test (doesn't assume equal variances)
let! result = ttestIndependent sample1 sample2 alpha
```

### Chi-Square Tests

```fsharp
// Goodness of fit
let observed = [|30; 20; 25; 25|]
let expected = [|25.0; 25.0; 25.0; 25.0|]
let! result = chiSquareGoodnessOfFit observed expected alpha

// Test of independence
let contingencyTable = array2D [|
    [|30; 20|]
    [|25; 25|]
|]
let! result = chiSquareIndependence contingencyTable alpha
```

### ANOVA

```fsharp
open Fowl.Stats.Anova

// Compare means across multiple groups
let group1 = [|5.2; 4.8; 5.5; 5.0; 5.3|]
let group2 = [|4.5; 4.2; 4.8; 4.3; 4.6|]
let group3 = [|6.0; 5.8; 6.2; 5.9; 6.1|]

let! result = oneWayANOVA [group1; group2; group3] alpha

printfn "F-statistic: %f" result.FStatistic
printfn "p-value: %f" result.PValue

// Post-hoc analysis
let! tukey = tukeyHSD [group1; group2; group3] result
```

### Non-Parametric Tests

```fsharp
open Fowl.Stats.NonParametricTests

// Mann-Whitney U (non-parametric alternative to t-test)
let! mwResult = mannWhitneyU sample1 sample2 alpha

// Wilcoxon signed-rank (paired samples)
let before = [|5.2; 4.8; 5.5; 5.0; 5.3|]
let after = [|5.5; 5.0; 5.8; 5.3; 5.6|]
let! wilcoxResult = wilcoxonSignedRank before after alpha
```

## 5.4 Correlation and Covariance

### Pearson Correlation

```fsharp
open Fowl.Stats.Correlation

let x = [|1.0; 2.0; 3.0; 4.0; 5.0|]
let y = [|2.0; 4.0; 6.0; 8.0; 10.0|]

// Perfect positive correlation
let corr = pearsonCorrelation x y  // 1.0

// Covariance
let cov = covariance x y

// Correlation matrix
let data = array2D [|
    [|1.0; 2.0; 3.0|]
    [|2.0; 4.0; 6.0|]
    [|3.0; 6.0; 9.0|]
|]
let! corrMatrix = correlationMatrix data
```

### Rank Correlation

```fsharp
open Fowl.Stats.RankCorrelation

// Spearman's rho (monotonic relationship)
let spearman = spearmanCorrelation x y

// Kendall's tau
let kendall = kendallTau x y
```

## 5.5 Regression Analysis

### Simple Linear Regression

```fsharp
open Fowl.Regression

let X = array2D [|
    [|1.0|]
    [|2.0|]
    [|3.0|]
    [|4.0|]
|]
let y = [|3.0; 5.0; 7.0; 9.0|]

let! result = OLS.fit X y

printfn "Coefficients: %A" result.Coefficients
printfn "R-squared: %f" result.RSquared
printfn "RMSE: %f" result.RMSE

// Make predictions
let Xnew = array2D [| [|5.0|] |]
let predictions = OLS.predict result Xnew
```

### Multiple Regression

```fsharp
let X = array2D [|
    [|1.0; 2.0|]
    [|2.0; 3.0|]
    [|3.0; 4.0|]
    [|4.0; 5.0|]
|]
let y = [|6.0; 9.0; 12.0; 15.0|]

let! result = OLS.fit X y
```

### Regularized Regression

```fsharp
// Ridge regression (L2 penalty)
let! ridgeResult = Ridge.fit X y 0.1

// Lasso regression (L1 penalty)
let! lassoResult = Lasso.fit X y 0.1
```

## 5.6 Exercises

### Exercise 5.1: Confidence Intervals

```fsharp
let confidenceInterval (data: float[]) (confidence: float) =
    result {
        let n = float data.Length
        let mean = Array.average data
        let std = Descriptive.std data
        
        // t-distribution critical value
        let df = n - 1.0
        let alpha = 1.0 - confidence
        let! tCrit = StudentTDistribution.ppf df (1.0 - alpha/2.0)
        
        let margin = tCrit * std / sqrt n
        return (mean - margin, mean + margin)
    }

// Usage
let data = [|5.2; 4.8; 5.5; 5.0; 5.3; 4.9; 5.1|]
match confidenceInterval data 0.95 with
| Ok (lower, upper) ->
    printfn "95%% CI: [%.3f, %.3f]" lower upper
```

### Exercise 5.2: A/B Testing

```fsharp
let abTest (control: float[]) (treatment: float[]) =
    result {
        // Descriptive statistics
        let controlMean = Descriptive.mean control
        let treatmentMean = Descriptive.mean treatment
        
        // Statistical test
        let! testResult = ttestIndependent control treatment 0.05
        
        // Effect size (Cohen's d)
        let pooledStd = sqrt ((Descriptive.var control + Descriptive.var treatment) / 2.0)
        let cohensD = (treatmentMean - controlMean) / pooledStd
        
        // Power analysis (simplified)
        let effectSize = abs cohensD
        
        return {
            ControlMean = controlMean
            TreatmentMean = treatmentMean
            Difference = treatmentMean - controlMean
            PValue = testResult.PValue
            Significant = testResult.Significant
            EffectSize = cohensD
        }
    }
```

### Exercise 5.3: Bootstrapping

```fsharp
let bootstrapCI (data: float[]) (nBootstrap: int) (confidence: float) =
    let rng = Random()
    let n = data.Length
    
    let bootstrapMeans =
        [|
            for _ in 1..nBootstrap do
                let sample = Array.init n (fun _ -> data.[rng.Next(n)])
                Array.average sample
        |]
    
    let sorted = Array.sort bootstrapMeans
    let alpha = 1.0 - confidence
    let lowerIdx = int (float nBootstrap * alpha / 2.0)
    let upperIdx = int (float nBootstrap * (1.0 - alpha / 2.0))
    
    (sorted.[lowerIdx], sorted.[upperIdx])
```

## 5.7 Summary

Key concepts:
- Descriptive statistics summarize data characteristics
- Distributions model random phenomena
- Hypothesis tests evaluate statistical claims
- Correlation measures association (not causation)
- Regression models relationships between variables

---

*Next: [Chapter 6: Optimization](chapter06.md)*
