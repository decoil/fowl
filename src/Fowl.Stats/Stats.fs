/// Fowl Statistics Module
/// Provides statistical functions including descriptive statistics, 
/// probability distributions, and correlation analysis
module Fowl.Stats

open Fowl

// Re-export descriptive statistics
let mean = Descriptive.mean
let var = Descriptive.var
let std = Descriptive.std
let median = Descriptive.median
let percentile = Descriptive.percentile
let skewness = Descriptive.skewness
let kurtosis = Descriptive.kurtosis
let moment = Descriptive.moment
let min = Descriptive.min
let max = Descriptive.max
let range = Descriptive.range
let sum = Descriptive.sum
let prod = Descriptive.prod
let cumsum = Descriptive.cumsum
let cumprod = Descriptive.cumprod

// Re-export distributions
let gaussianPdf = Distributions.Gaussian.pdf
let gaussianCdf = Distributions.Gaussian.cdf
let gaussianPpf = Distributions.Gaussian.ppf
let gaussianRvs = Distributions.Gaussian.rvs
let gaussianLogpdf = Distributions.Gaussian.logpdf

let uniformPdf = Distributions.Uniform.pdf
let uniformCdf = Distributions.Uniform.cdf
let uniformPpf = Distributions.Uniform.ppf
let uniformRvs = Distributions.Uniform.rvs

let exponentialPdf = Distributions.Exponential.pdf
let exponentialCdf = Distributions.Exponential.cdf
let exponentialPpf = Distributions.Exponential.ppf
let exponentialRvs = Distributions.Exponential.rvs

let gammaPdf = Distributions.Gamma.pdf
let gammaRvs = Distributions.Gamma.rvs

let betaPdf = Distributions.Beta.pdf
let betaRvs = Distributions.Beta.rvs

// Re-export correlation
let covariance = Correlation.covariance
let pearsonCorrelation = Correlation.pearsonCorrelation
let correlationMatrix = Correlation.correlationMatrix
