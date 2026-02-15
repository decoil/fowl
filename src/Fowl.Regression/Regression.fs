namespace Fowl.Regression

open System
open Fowl
open Fowl.Core.Types
open Fowl.Linq
open Fowl.Stats

/// <summary>Result from a regression fit.
/// </summary>
type RegressionResult = {
    /// Coefficients (including intercept as first element)
    Coefficients: float[]
    /// R-squared (coefficient of determination)
    RSquared: float
    /// Adjusted R-squared
    AdjustedRSquared: float
    /// Residual sum of squares
    RSS: float
    /// Total sum of squares
    TSS: float
    /// Root mean squared error
    RMSE: float
    /// Standard errors of coefficients
    StandardErrors: float[]
    /// T-statistics for coefficients
    TStatistics: float[]
    /// P-values for coefficients
    PValues: float[]
    /// Predicted values
    Predicted: float[]
    /// Residuals
    Residuals: float[]
}

/// <summary>Options for regularized regression.
/// </summary>
type RegularizationOptions = {
    /// L2 regularization strength (Ridge)
    Alpha: float
    /// L1 regularization strength (Lasso)
    L1Ratio: float
    /// Maximum iterations for iterative solvers
    MaxIter: int
    /// Convergence tolerance
    Tol: float
}

/// <summary>Ordinary Least Squares (OLS) regression.
/// </summary>
module OLS =
    
    /// <summary>Fit linear regression using normal equations.
    /// Solves: beta = (X^T X)^-1 X^T y
    /// </summary>
    let fit (X: float[,]) (y: float[]) : FowlResult<RegressionResult> =
        let n = X.GetLength(0)
        let p = X.GetLength(1)
        
        if n <> y.Length then
            Error.invalidArgument "X and y must have same number of rows"
        elif n <= p then
            Error.invalidArgument "Need more samples than features"
        else
            result {
                // Add intercept column
                let XWithIntercept = Array2D.init n (p + 1) (fun i j -
                    if j = 0 then 1.0 else X.[i, j-1])
                
                let! xArr = Ndarray.ofArray2D XWithIntercept
                let! yArr = Ndarray.ofArray y [||]
                
                // Normal equations: (X^T X)^-1 X^T y
                let! xt = Matrix.transpose xArr
                let! xtx = Matrix.matmul xt xArr
                let! xtxInv = Factorizations.inv xtx
                let! xty = Matrix.matmul xt yArr
                let! coeffsArr = Matrix.matmul xtxInv xty
                
                let coefficients = Ndarray.toArray coeffsArr
                
                // Predictions
                let! predictionsArr = Matrix.matmul xArr coeffsArr
                let predicted = Ndarray.toArray predictionsArr
                
                // Residuals
                let residuals = Array.map2 (-) y predicted
                
                // RSS and TSS
                let yMean = Array.average y
                let rss = residuals |> Array.sumBy (fun r -> r * r)
                let tss = y |> Array.sumBy (fun yi -> (yi - yMean) ** 2.0)
                
                // R-squared
                let r2 = 1.0 - rss / tss
                let adjR2 = 1.0 - (1.0 - r2) * float (n - 1) / float (n - p - 1)
                
                // RMSE
                let rmse = sqrt (rss / float n)
                
                // Standard errors (simplified - assumes homoscedasticity)
                let mse = rss / float (n - p - 1)
                let! xtxInvArr = Ndarray.toArray2D xtxInv
                let stdErrors = 
                    Array.init (p + 1) (fun i -
                        sqrt (mse * xtxInvArr.[i, i]))
                
                // T-statistics and p-values
                let tStats = Array.map2 (/) coefficients stdErrors
                
                // Approximate p-values using normal distribution
                let pValues = 
                    tStats |> Array.map (fun t -
                        2.0 * (1.0 - GaussianDistribution.cdf 0.0 1.0 (abs t) |> Result.defaultValue 0.5))
                
                return {
                    Coefficients = coefficients
                    RSquared = r2
                    AdjustedRSquared = adjR2
                    RSS = rss
                    TSS = tss
                    RMSE = rmse
                    StandardErrors = stdErrors
                    TStatistics = tStats
                    PValues = pValues
                    Predicted = predicted
                    Residuals = residuals
                }
            }
    
    /// <summary>Predict using fitted OLS model.
    /// </summary>
    let predict (result: RegressionResult) (X: float[,]) : float[] =
        let n = X.GetLength(0)
        let p = X.GetLength(1)
        let coeffs = result.Coefficients
        
        Array.init n (fun i -
            let mutable pred = coeffs.[0]  // Intercept
            for j = 0 to p - 1 do
                pred <- pred + coeffs.[j+1] * X.[i, j]
            pred)

/// <summary>Ridge regression (L2 regularization).
/// </summary>
module Ridge =
    
    /// <summary>Fit Ridge regression.
    /// Solves: beta = (X^T X + alpha*I)^-1 X^T y
    /// </summary>
    let fit (X: float[,]) (y: float[]) (alpha: float) : FowlResult<RegressionResult> =
        let n = X.GetLength(0)
        let p = X.GetLength(1)
        
        if n <> y.Length then
            Error.invalidArgument "X and y must have same number of rows"
        else
            result {
                // Add intercept column (not regularized)
                let XWithIntercept = Array2D.init n (p + 1) (fun i j -
                    if j = 0 then 1.0 else X.[i, j-1])
                
                let! xArr = Ndarray.ofArray2D XWithIntercept
                let! yArr = Ndarray.ofArray y [||]
                
                // X^T X + alpha*I (but don't regularize intercept)
                let! xt = Matrix.transpose xArr
                let! xtx = Matrix.matmul xt xArr
                
                let! xtxArr = Ndarray.toArray2D xtx
                for i = 1 to p do  // Skip intercept (index 0)
                    xtxArr.[i, i] <- xtxArr.[i, i] + alpha
                
                let! xtxReg = Ndarray.ofArray2D xtxArr
                let! xtxInv = Factorizations.inv xtxReg
                let! xty = Matrix.matmul xt yArr
                let! coeffsArr = Matrix.matmul xtxInv xty
                
                let coefficients = Ndarray.toArray coeffsArr
                
                // Predictions and metrics
                let! predictionsArr = Matrix.matmul xArr coeffsArr
                let predicted = Ndarray.toArray predictionsArr
                let residuals = Array.map2 (-) y predicted
                
                let yMean = Array.average y
                let rss = residuals |> Array.sumBy (fun r -> r * r)
                let tss = y |> Array.sumBy (fun yi -> (yi - yMean) ** 2.0)
                let r2 = 1.0 - rss / tss
                let rmse = sqrt (rss / float n)
                
                return {
                    Coefficients = coefficients
                    RSquared = r2
                    AdjustedRSquared = r2  // Simplified
                    RSS = rss
                    TSS = tss
                    RMSE = rmse
                    StandardErrors = Array.zeroCreate (p + 1)
                    TStatistics = Array.zeroCreate (p + 1)
                    PValues = Array.zeroCreate (p + 1)
                    Predicted = predicted
                    Residuals = residuals
                }
            }

/// <summary>Lasso regression (L1 regularization).
/// Uses iterative soft thresholding (ISTA).
/// </summary>
module Lasso =
    
    /// <summary>Soft thresholding operator.
    /// </summary>
    let private softThreshold (x: float) (lambda: float) : float =
        if x > lambda then
            x - lambda
        elif x < -lambda then
            x + lambda
        else
            0.0
    
    /// <summary>Fit Lasso regression using ISTA.
    /// </summary>
    let fit (X: float[,]) (y: float[]) (alpha: float) 
            ?(maxIter: int) ?(tol: float) : FowlResult<RegressionResult> =
        
        let maxIter = defaultArg maxIter 1000
        let tol = defaultArg tol 1e-4
        let n = X.GetLength(0)
        let p = X.GetLength(1)
        
        if n <> y.Length then
            Error.invalidArgument "X and y must have same number of rows"
        else
            // Normalize X and y
            let xMeans = Array.init p (fun j -
                Array.init n (fun i -> X.[i, j]) |> Array.average)
            let xStds = Array.init p (fun j -
                let col = Array.init n (fun i -> X.[i, j])
                match Descriptive.std col with
                | Ok s -> max s 1e-8
                | Error _ -> 1.0)
            
            let XNormalized = Array2D.init n p (fun i j -
                (X.[i, j] - xMeans.[j]) / xStds.[j])
            
            let yMean = Array.average y
            let yStd = match Descriptive.std y with Ok s -> s | Error _ -> 1.0
            let yNormalized = y |> Array.map (fun yi -> (yi - yMean) / yStd)
            
            // ISTA algorithm
            let mutable coeffs = Array.zeroCreate p
            let learningRate = 0.01
            let lambda = alpha * learningRate
            
            let mutable converged = false
            let mutable iter = 0
            
            while not converged && iter < maxIter do
                let oldCoeffs = Array.copy coeffs
                
                // Gradient step
                for j = 0 to p - 1 do
                    let mutable gradient = 0.0
                    for i = 0 to n - 1 do
                        let pred = Array.sumBy (fun k -> coeffs.[k] * XNormalized.[i, k]) [|0..p-1|]
                        let error = yNormalized.[i] - pred
                        gradient <- gradient - error * XNormalized.[i, j]
                    
                    let update = coeffs.[j] - learningRate * gradient / float n
                    coeffs.[j] <- softThreshold update lambda
                
                // Check convergence
                let change = Array.map2 (fun old neu -> abs (old - neu)) oldCoeffs coeffs |> Array.max
                if change < tol then
                    converged <- true
                
                iter <- iter + 1
            
            // Denormalize coefficients
            let finalCoeffs = Array.zeroCreate (p + 1)
            finalCoeffs.[0] <- yMean  // Intercept
            for j = 0 to p - 1 do
                finalCoeffs.[j+1] <- coeffs.[j] * yStd / xStds.[j]
                finalCoeffs.[0] <- finalCoeffs.[0] - finalCoeffs.[j+1] * xMeans.[j]
            
            // Calculate metrics
            let predicted = OLS.predict { Coefficients = finalCoeffs } X
            let residuals = Array.map2 (-) y predicted
            let yMean = Array.average y
            let rss = residuals |> Array.sumBy (fun r -> r * r)
            let tss = y |> Array.sumBy (fun yi -> (yi - yMean) ** 2.0)
            let r2 = 1.0 - rss / tss
            let rmse = sqrt (rss / float n)
            
            Ok {
                Coefficients = finalCoeffs
                RSquared = r2
                AdjustedRSquared = r2
                RSS = rss
                TSS = tss
                RMSE = rmse
                StandardErrors = Array.zeroCreate (p + 1)
                TStatistics = Array.zeroCreate (p + 1)
                PValues = Array.zeroCreate (p + 1)
                Predicted = predicted
                Residuals = residuals
            }

/// <summary>Logistic regression for classification.
/// </summary>
module Logistic =
    
    /// <summary>Sigmoid function.
    /// </summary>
    let private sigmoid (z: float) : float =
        1.0 / (1.0 + exp (-z))
    
    /// <summary>Fit logistic regression using gradient descent.
    /// </summary>
    let fit (X: float[,]) (y: float[]) 
            ?(maxIter: int) ?(learningRate: float) ?(tol: float) : FowlResult<float[]> =
        
        let maxIter = defaultArg maxIter 1000
        let learningRate = defaultArg learningRate 0.1
        let tol = defaultArg tol 1e-6
        let n = X.GetLength(0)
        let p = X.GetLength(1)
        
        if n <> y.Length then
            Error.invalidArgument "X and y must have same number of rows"
        elif y |> Array.exists (fun yi -> yi <> 0.0 && yi <> 1.0) then
            Error.invalidArgument "y must be binary (0 or 1)"
        else
            // Add intercept
            let XWithIntercept = Array2D.init n (p + 1) (fun i j -
                if j = 0 then 1.0 else X.[i, j-1])
            
            let mutable coeffs = Array.zeroCreate (p + 1)
            let mutable converged = false
            let mutable iter = 0
            
            while not converged && iter < maxIter do
                let oldCoeffs = Array.copy coeffs
                
                // Compute predictions
                let predictions = 
                    Array.init n (fun i -
                        let z = Array.sumBy (fun j -
                            coeffs.[j] * XWithIntercept.[i, j]) [|0..p|]
                        sigmoid z)
                
                // Compute gradients
                let gradients = 
                    Array.init (p + 1) (fun j -
                        let mutable grad = 0.0
                        for i = 0 to n - 1 do
                            grad <- grad + (predictions.[i] - y.[i]) * XWithIntercept.[i, j]
                        grad / float n)
                
                // Update coefficients
                for j = 0 to p do
                    coeffs.[j] <- coeffs.[j] - learningRate * gradients.[j]
                
                // Check convergence
                let change = Array.map2 (fun old neu -> abs (old - neu)) oldCoeffs coeffs |> Array.max
                if change < tol then
                    converged <- true
                
                iter <- iter + 1
            
            Ok coeffs
    
    /// <summary>Predict probabilities using logistic regression.
    /// </summary>
    let predictProba (coeffs: float[]) (X: float[,]) : float[] =
        let n = X.GetLength(0)
        let p = X.GetLength(1)
        
        Array.init n (fun i -
            let mutable z = coeffs.[0]  // Intercept
            for j = 0 to p - 1 do
                z <- z + coeffs.[j+1] * X.[i, j]
            sigmoid z)
    
    /// <summary>Predict class labels (0 or 1).
    /// </summary>
    let predict (coeffs: float[]) (X: float[,]) : float[] =
        predictProba coeffs X |> Array.map (fun p -> if p >= 0.5 then 1.0 else 0.0)
