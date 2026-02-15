# Tutorial 3: Statistics

Learn statistical analysis with Fowl's comprehensive statistics module.

## Overview

Fowl provides a complete statistics module including descriptive statistics, probability distributions, hypothesis testing, and correlation analysis.

## Learning Objectives

- Compute descriptive statistics
- Work with probability distributions
- Perform hypothesis tests
- Calculate correlations
- Generate random samples

## Setup

```fsharp
open Fowl
open Fowl.Core.Types
open Fowl.Stats

let unwrap = function Ok x -> x | Error e -> failwith e.Message
```

## Descriptive Statistics

### Central Tendency

```fsharp
let data = [|12.0; 15.0; 18.0; 21.0; 24.0; 27.0; 30.0|]

// Mean (average)
let mean = Descriptive.mean data |> unwrap  // 18.14...

// Median (middle value)
let median = Descriptive.median data |> unwrap  // 21.0

// Mode (most frequent value)
let mode = Descriptive.mode data  // None (all unique)

let dataWithMode = [|1.0; 2.0; 2.0; 3.0; 2.0|]
let modeResult = Descriptive.mode dataWithMode  // Some 2.0
```

### Dispersion

```fsharp
// Variance
let variance = Descriptive.var data |> unwrap

// Standard deviation
let stdDev = Descriptive.std data |> unwrap

// Range
let range = Descriptive.range data |> unwrap  // (12.0, 30.0)

// Interquartile range
let iqr = DescriptiveExtended.iqr data |> unwrap  // Q3 - Q1

// Median absolute deviation (robust measure)
let mad = DescriptiveExtended.mad data |> unwrap
```

### Shape

```fsharp
// Skewness (asymmetry)
let skew = Descriptive.skewness data |> unwrap
// Negative = left-skewed, Positive = right-skewed

// Kurtosis (tail heaviness)
let kurt = Descriptive.kurtosis data |> unwrap
// High kurtosis = heavy tails

// Standard error of mean
let sem = DescriptiveExtended.sem data |> unwrap
```

### Percentiles

```fsharp
// Quartiles
let q1 = Descriptive.percentile data 25.0 |> unwrap  // 25th percentile
let q2 = Descriptive.percentile data 50.0 |> unwrap  // Median
let q3 = Descriptive.percentile data 75.0 |> unwrap  // 75th percentile

// Five-number summary
let min, q1, median, q3, max = 
    DescriptiveExtended.fiveNumberSummary data |> unwrap

// Custom percentiles
let p10 = Descriptive.percentile data 10.0 |> unwrap
let p90 = Descriptive.percentile data 90.0 |> unwrap
```

## Probability Distributions

### Continuous Distributions

#### Normal (Gaussian)

```fsharp
// Parameters: mean, standard deviation
let mu = 0.0
let sigma = 1.0

// Probability density function (PDF)
let pdf = GaussianDistribution.pdf mu sigma 0.0 |> unwrap  // 0.399...

// Cumulative distribution function (CDF)
let cdf = GaussianDistribution.cdf mu sigma 1.96 |> unwrap  // 0.975

// Percent point function (inverse CDF)
let ppf = GaussianDistribution.ppf mu sigma 0.975 |> unwrap  // 1.96

// Random sampling
let sample = GaussianDistribution.rvs mu sigma (Some 42) |> unwrap
let samples = Array.init 1000 (fun _ -
    GaussianDistribution.rvs mu sigma (Some 42) |> unwrap)

// Moments
let mean = GaussianDistribution.mean mu sigma |> unwrap
let variance = GaussianDistribution.variance mu sigma |> unwrap
```

#### Other Continuous Distributions

```fsharp
// Gamma distribution
let gammaSample = GammaDistribution.rvs 2.0 1.0 (Some 42) |> unwrap

// Beta distribution
let betaSample = BetaDistribution.rvs 2.0 5.0 (Some 42) |> unwrap

// Student's t-distribution
let tSample = StudentTDistribution.rvs 10.0 (Some 42) |> unwrap

// Chi-square distribution
let chi2Sample = ChiSquareDistribution.rvs 3.0 (Some 42) |> unwrap

// F-distribution
let fSample = FDistribution.rvs 5.0 10.0 (Some 42) |> unwrap

// Cauchy distribution (heavy-tailed)
let cauchySample = CauchyDistribution.rvs 0.0 1.0 (Some 42) |> unwrap

// Weibull distribution (reliability)
let weibullSample = WeibullDistribution.rvs 2.0 1.0 (Some 42) |> unwrap

// Log-normal distribution (finance)
let lognormalSample = LogNormalDistribution.rvs 0.0 1.0 (Some 42) |> unwrap
```

### Discrete Distributions

```fsharp
// Binomial: n trials, p probability of success
let binomSample = BinomialDistribution.rvs 10 0.3 (Some 42) |> unwrap

// Poisson: λ average rate
let poissonSample = PoissonDistribution.rvs 2.5 (Some 42) |> unwrap

// Geometric: p probability of success
let geomSample = GeometricDistribution.rvs 0.3 (Some 42) |> unwrap

// Multinomial: multiple categories
let probs = [|0.2; 0.3; 0.5|]
let multinomSample = MultinomialDistribution.rvs probs 100 (Some 42) |> unwrap
```

### Multivariate Distributions

```fsharp
// Dirichlet distribution (for probabilities)
let alpha = [|1.0; 2.0; 3.0|]
let dirichletSample = DirichletDistribution.rvs alpha (Some 42) |> unwrap
// Returns [|p1; p2; p3|] where sum = 1
```

## Hypothesis Testing

### t-Tests

```fsharp
// One-sample t-test
// H0: mean = μ0
let data = [|1.2; 1.5; 1.3; 1.7; 1.4|]
let tResult = HypothesisTests.ttest_one_sample data 1.0 |> unwrap

printfn "t-statistic: %.4f" tResult.Statistic
printfn "p-value: %.4f" tResult.PValue
printfn "Degrees of freedom: %d" tResult.DegreesOfFreedom

if tResult.PValue < 0.05 then
    printfn "Reject H0: mean is significantly different from 1.0"
else
    printfn "Fail to reject H0"

// Independent two-sample t-test
// H0: mean1 = mean2
let group1 = [|23.0; 25.0; 28.0; 22.0; 26.0|]  // Treatment
let group2 = [|20.0; 21.0; 19.0; 22.0; 20.0|]  // Control
let tIndResult = HypothesisTests.ttest_independent group1 group2 |> unwrap

printfn "Difference in means: %.2f" (Descriptive.mean group1 |> unwrap - Descriptive.mean group2 |> unwrap)
printfn "p-value: %.4f" tIndResult.PValue
```

### ANOVA

```fsharp
// One-way ANOVA
// H0: All group means are equal
let treatmentA = [|23.0; 25.0; 28.0; 22.0; 26.0|]
let treatmentB = [|30.0; 32.0; 29.0; 31.0; 33.0|]
let treatmentC = [|19.0; 21.0; 20.0; 22.0; 18.0|]
let groups = [|treatmentA; treatmentB; treatmentC|]

let anovaResult = Anova.oneWay groups |> unwrap

printfn "F-statistic: %.4f" anovaResult.FStatistic
printfn "p-value: %.4f" anovaResult.PValue
printfn "R-squared: %.4f" anovaResult.RSquared

// Post-hoc analysis (if significant)
if anovaResult.PValue < 0.05 then
    printfn "Significant difference found between groups"
    // Perform pairwise comparisons or use Tukey HSD
```

### Non-Parametric Tests

```fsharp
// Mann-Whitney U test (non-parametric alternative to t-test)
let mwResult = NonParametricTests.mannWhitneyU group1 group2 |> unwrap
printfn "U-statistic: %.1f, p-value: %.4f" mwResult.UStatistic mwResult.PValue

// Kruskal-Wallis test (non-parametric ANOVA)
let kwResult = NonParametricTests.kruskalWallis groups |> unwrap
printfn "H-statistic: %.4f, p-value: %.4f" kwResult.HStatistic kwResult.PValue

// Wilcoxon signed-rank test (paired samples)
let before = [|10.0; 12.0; 14.0; 16.0; 18.0|]
let after = [|12.0; 13.0; 15.0; 16.0; 19.0|]
let wResult = NonParametricTests.wilcoxonSignedRank before after |> unwrap
```

### Chi-Square Tests

```fsharp
// Goodness of fit test
let observed = [|45.0; 55.0; 35.0; 65.0|]  // Observed frequencies
let expected = [|50.0; 50.0; 50.0; 50.0|]  // Expected under H0
let chi2Result = HypothesisTests.chi2_goodness observed expected |> unwrap

// Independence test (contingency table)
// Rows: Gender, Columns: Preference
let contingency = [|[|30.0; 20.0|]; [|25.0; 25.0|]|]
let chi2IndResult = HypothesisTests.chi2_independence contingency |> unwrap
```

### Variance Tests

```fsharp
// Levene's test (robust to non-normality)
let leveneResult = HypothesisTestsExtended.levene groups |> unwrap

// Bartlett's test (assumes normality)
let bartlettResult = HypothesisTestsExtended.bartlett groups |> unwrap

// F-test for equality of variances
let fTestResult = HypothesisTests.f_test treatmentA treatmentB |> unwrap
```

### Normality Tests

```fsharp
let data = GaussianDistribution.rvs 0.0 1.0 (Some 42) |> unwrap |> Array.init 100 (fun _ -> ...)

// Shapiro-Wilk test
let swResult = NormalityTests.shapiro_wilk data |> unwrap

// Anderson-Darling test
let adResult = NormalityTests.anderson_darling data |> unwrap

// Jarque-Bera test
let jbResult = NormalityTests.jarque_bera data |> unwrap

// Kolmogorov-Smirnov test
let ksResult = NormalityTests.kolmogorov_smirnov data 0.0 1.0 |> unwrap
```

## Correlation Analysis

### Pearson Correlation

```fsharp
let x = [|1.0; 2.0; 3.0; 4.0; 5.0|]
let y = [|2.0; 4.0; 6.0; 8.0; 10.0|]  // Perfect positive correlation

let r = Correlation.pearsonCorrelation x y |> unwrap
printfn "Pearson r: %.4f" r  // 1.0

// Test significance
let rWithP = Correlation.pearsonCorrelationWithPValue x y |> unwrap
let corr, pValue = rWithP
```

### Rank Correlations

```fsharp
// Spearman rank correlation (monotonic relationships)
let rho = RankCorrelation.spearman x y |> unwrap
let rhoWithP = RankCorrelation.spearmanWithPValue x y |> unwrap

// Kendall tau correlation (ordinal association)
let tau = RankCorrelation.kendall x y |> unwrap
let tauWithP = RankCorrelation.kendallWithPValue x y |> unwrap
```

### Correlation Matrix

```fsharp
// Multiple variables
let var1 = [|1.0; 2.0; 3.0; 4.0; 5.0|]
let var2 = [|2.0; 4.0; 6.0; 8.0; 10.0|]
let var3 = [|5.0; 4.0; 3.0; 2.0; 1.0|]  // Negative correlation with var1

let data = [|var1; var2; var3|]
let corrMatrix = Correlation.correlationMatrix data |> unwrap
// corrMatrix.[0,1] ≈ 1.0 (positive)
// corrMatrix.[0,2] ≈ -1.0 (negative)
```

## Practical Examples

### Example 1: A/B Testing

```fsharp
// Website conversion rates
let controlGroup = [|[|0.0|]; [|1.0|]; [|0.0|]; [|0.0|]; [|1.0|]; ...|]  // 0 = no convert, 1 = convert
let treatmentGroup = [|[|1.0|]; [|1.0|]; [|0.0|]; [|1.0|]; [|1.0|]; ...|]

// Two-proportion z-test (simplified as t-test here)
let controlRates = controlGroup |> Array.map (fun x -> x.[0])
let treatmentRates = treatmentGroup |> Array.map (fun x -> x.[0])

let abResult = HypothesisTests.ttest_independent controlRates treatmentRates |> unwrap

if abResult.PValue < 0.05 then
    let controlMean = Descriptive.mean controlRates |> unwrap
    let treatmentMean = Descriptive.mean treatmentRates |> unwrap
    printfn "Significant difference! Treatment: %.2f%% vs Control: %.2f%%" 
        (treatmentMean * 100.0) (controlMean * 100.0)
```

### Example 2: Quality Control

```fsharp
// Product measurements over time
let sample1 = [|10.1; 10.2; 10.0; 10.3; 10.1|]  // Batch 1
let sample2 = [|10.5; 10.6; 10.4; 10.7; 10.5|]  // Batch 2

// Check if variance has changed
let leveneResult = HypothesisTestsExtended.levene [|sample1; sample2|] |> unwrap

if leveneResult.PValue < 0.05 then
    printfn "WARNING: Variance has changed significantly!"
    printfn "Standard deviations: %.3f vs %.3f"
        (Descriptive.std sample1 |> unwrap)
        (Descriptive.std sample2 |> unwrap)
```

### Example 3: Correlation Analysis

```fsharp
// Study time vs exam scores
let studyHours = [|2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0|]
let examScores = [|65.0; 72.0; 78.0; 85.0; 88.0; 92.0; 95.0|]

let correlation = Correlation.pearsonCorrelation studyHours examScores |> unwrap
let corrTest = Correlation.pearsonCorrelationWithPValue studyHours examScores |> unwrap

printfn "Correlation: %.3f" correlation
printfn "P-value: %.4f" (snd corrTest)

if snd corrTest < 0.05 then
    printfn "Significant positive correlation between study time and scores!"
```

## Exercises

1. Generate 1000 samples from a normal distribution and verify mean ≈ 0, std ≈ 1
2. Perform a t-test to compare two groups of exam scores
3. Test if a dataset follows normal distribution using Shapiro-Wilk
4. Calculate Spearman correlation between two variables with outliers
5. Perform ANOVA on three different teaching methods

## Solutions

```fsharp
// Exercise 1
let samples = Array.init 1000 (fun _ -
    GaussianDistribution.rvs 0.0 1.0 (Some 42) |> unwrap)
let sampleMean = Descriptive.mean samples |> unwrap  // Close to 0
let sampleStd = Descriptive.std samples |> unwrap    // Close to 1

// Exercise 2
let classA = [|75.0; 82.0; 78.0; 85.0; 79.0|]
let classB = [|68.0; 72.0; 70.0; 74.0; 71.0|]
let tResult = HypothesisTests.ttest_independent classA classB |> unwrap

// Exercise 3
let data = [|1.0; 2.0; 3.0; 4.0; 5.0; 100.0|]  // Outlier makes non-normal
let swResult = NormalityTests.shapiro_wilk data |> unwrap

// Exercise 4 (robust to outliers)
let x = [|1.0; 2.0; 3.0; 4.0; 100.0|]  // Outlier
let y = [|2.0; 4.0; 6.0; 8.0; 200.0|]  // Corresponding outlier
let spearmanRho = RankCorrelation.spearman x y |> unwrap  // Still ~1.0

// Exercise 5
let method1 = [|85.0; 88.0; 82.0; 90.0; 87.0|]
let method2 = [|78.0; 75.0; 80.0; 76.0; 79.0|]
let method3 = [|92.0; 95.0; 89.0; 94.0; 91.0|]
let anovaResult = Anova.oneWay [|[|method1; method2; method3|]|] |> unwrap
```

## Next Steps

- [Tutorial 4: Neural Networks](Tutorial4_NeuralNetworks.md)
- [User Guide](../USER_GUIDE.md#statistics)

---

*Estimated time: 45 minutes*