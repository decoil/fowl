namespace Fowl.Neural

open System
open Fowl

/// <summary>
/// Additional optimizers beyond SGD and Adam.
/// Adagrad, Adadelta, RMSprop, Adamax, Nadam.
/// </summary>
module AdvancedOptimizers =
    
    /// <summary>
    /// Adagrad optimizer state.
    /// Accumulates squared gradients for adaptive learning rates.
    /// </summary>
type AdagradState = {
        /// Accumulated squared gradients
        mutable AccSquaredGradients: float[]
        /// Learning rate
        LearningRate: float
        /// Small constant for numerical stability
        Epsilon: float
    }
    
    /// <summary>
    /// Adadelta optimizer state.
    /// Adapts learning rate based on moving window of gradient updates.
    /// </summary>
type AdadeltaState = {
        /// Accumulated squared gradients
        mutable AccSquaredGradients: float[]
        /// Accumulated squared parameter updates
        mutable AccSquaredUpdates: float[]
        /// Decay rate (rho)
        Rho: float
        /// Small constant for numerical stability
        Epsilon: float
    }
    
    /// <summary>
    /// RMSprop optimizer state.
    /// Moving average of squared gradients.
    /// </summary>
type RMSpropState = {
        /// Moving average of squared gradients
        mutable MovingAvgSquaredGradients: float[]
        /// Learning rate
        LearningRate: float
        /// Decay rate (rho)
        Rho: float
        /// Small constant for numerical stability
        Epsilon: float
    }
    
    /// <summary>
    /// Adamax optimizer state.
    /// Variant of Adam based on infinity norm.
    /// </summary>
type AdamaxState = {
        /// First moment (mean)
        mutable M: float[]
        /// Infinity norm moment
        mutable U: float[]
        /// Learning rate
        LearningRate: float
        /// Beta1 (first moment decay)
        Beta1: float
        /// Beta2 (infinity norm decay)
        Beta2: float
        /// Small constant
        Epsilon: float
        /// Time step
        mutable T: int
    }
    
    /// <summary>
    /// Create Adagrad optimizer.
    /// </summary>
    let createAdagrad (learningRate: float) (epsilon: float) (numParameters: int) : AdagradState =
        {
            AccSquaredGradients = Array.zeroCreate numParameters
            LearningRate = learningRate
            Epsilon = epsilon
        }
    
    /// <summary>
    /// Adagrad parameter update.
    /// Parameter update: θ = θ - (η / √(G + ε)) * g
    /// where G is accumulated squared gradient.
    /// </summary>
    let updateAdagrad (state: AdagradState) (parameters: float[]) (gradients: float[]) : unit =
        let lr = state.LearningRate
        let eps = state.Epsilon
        
        for i = 0 to parameters.Length - 1 do
            // Accumulate squared gradient
            state.AccSquaredGradients.[i] <- state.AccSquaredGradients.[i] + gradients.[i] * gradients.[i]
            
            // Adaptive learning rate
            let adaptiveLR = lr / (sqrt state.AccSquaredGradients.[i] + eps)
            
            // Update parameter
            parameters.[i] <- parameters.[i] - adaptiveLR * gradients.[i]
    
    /// <summary>
    /// Create Adadelta optimizer.
    /// </summary>
    let createAdadelta (rho: float) (epsilon: float) (numParameters: int) : AdadeltaState =
        {
            AccSquaredGradients = Array.zeroCreate numParameters
            AccSquaredUpdates = Array.zeroCreate numParameters
            Rho = rho
            Epsilon = epsilon
        }
    
    /// <summary>
    /// Adadelta parameter update.
    /// No explicit learning rate needed!
    /// </summary>
    let updateAdadelta (state: AdadeltaState) (parameters: float[]) (gradients: float[]) : unit =
        let rho = state.Rho
        let eps = state.Epsilon
        
        for i = 0 to parameters.Length - 1 do
            // Accumulate squared gradient
            state.AccSquaredGradients.[i] <- rho * state.AccSquaredGradients.[i] + (1.0 - rho) * gradients.[i] * gradients.[i]
            
            // Compute update
            let rmsUpdate = sqrt (state.AccSquaredUpdates.[i] + eps)
            let rmsGradient = sqrt (state.AccSquaredGradients.[i] + eps)
            let update = (rmsUpdate / rmsGradient) * gradients.[i]
            
            // Update parameter
            parameters.[i] <- parameters.[i] - update
            
            // Accumulate squared update
            state.AccSquaredUpdates.[i] <- rho * state.AccSquaredUpdates.[i] + (1.0 - rho) * update * update
    
    /// <summary>
    /// Create RMSprop optimizer.
    /// </summary>
    let createRMSprop (learningRate: float) (rho: float) (epsilon: float) (numParameters: int) : RMSpropState =
        {
            MovingAvgSquaredGradients = Array.zeroCreate numParameters
            LearningRate = learningRate
            Rho = rho
            Epsilon = epsilon
        }
    
    /// <summary>
    /// RMSprop parameter update.
    /// Similar to Adadelta but maintains learning rate.
    /// </summary>
    let updateRMSprop (state: RMSpropState) (parameters: float[]) (gradients: float[]) : unit =
        let lr = state.LearningRate
        let rho = state.Rho
        let eps = state.Epsilon
        
        for i = 0 to parameters.Length - 1 do
            // Update moving average of squared gradients
            state.MovingAvgSquaredGradients.[i] <- 
                rho * state.MovingAvgSquaredGradients.[i] + 
                (1.0 - rho) * gradients.[i] * gradients.[i]
            
            // Adaptive learning rate
            let adaptiveLR = lr / (sqrt state.MovingAvgSquaredGradients.[i] + eps)
            
            // Update parameter
            parameters.[i] <- parameters.[i] - adaptiveLR * gradients.[i]
    
    /// <summary>
    /// Create Adamax optimizer.
    /// </summary>
    let createAdamax (learningRate: float) (beta1: float) (beta2: float) (epsilon: float) (numParameters: int) : AdamaxState =
        {
            M = Array.zeroCreate numParameters
            U = Array.zeroCreate numParameters
            LearningRate = learningRate
            Beta1 = beta1
            Beta2 = beta2
            Epsilon = epsilon
            T = 0
        }
    
    /// <summary>
    /// Adamax parameter update.
    /// Uses infinity norm instead of L2 norm.
    /// </summary>
    let updateAdamax (state: AdamaxState) (parameters: float[]) (gradients: float[]) : unit =
        state.T <- state.T + 1
        let lr = state.LearningRate
        let beta1 = state.Beta1
        let beta2 = state.Beta2
        let eps = state.Epsilon
        
        let biasCorrection1 = 1.0 - beta1 ** float state.T
        
        for i = 0 to parameters.Length - 1 do
            // Update biased first moment
            state.M.[i] <- beta1 * state.M.[i] + (1.0 - beta1) * gradients.[i]
            
            // Update infinity norm moment
            let absGrad = abs gradients.[i]
            state.U.[i] <- max (beta2 * state.U.[i]) (absGrad + eps)
            
            // Bias-corrected first moment
            let mHat = state.M.[i] / biasCorrection1
            
            // Update parameter
            parameters.[i] <- parameters.[i] - (lr / state.U.[i]) * mHat