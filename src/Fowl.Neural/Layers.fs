namespace Fowl.Neural

open System
open Fowl
open Fowl.Core.Types

/// <summary>Dense (fully connected) layer implementation.
/// Linear transformation followed by optional activation.
/// </summary>type DenseLayer = {
    /// Weights matrix [input_dim, output_dim]
    Weights: Node
    /// Bias vector [output_dim]
    Bias: Node
    /// Optional activation function
    Activation: ActivationFn option
    /// Input dimension
    InputDim: int
    /// Output dimension
    OutputDim: int
}

/// <summary>Module for creating and using neural network layers.
/// </summary>module Layers =
    
    /// <summary>Create a dense (fully connected) layer.
    /// </summary>/// <param name="inputDim">Input feature dimension.</param>/// <param name="outputDim">Output feature dimension.</param>/// <param name="activation">Optional activation function.</param>/// <param name="seed">Random seed for weight initialization.</param>/// <returns>Dense layer with initialized parameters.</returns>let dense (inputDim: int) (outputDim: int) (activation: ActivationFn option) (seed: int option) : FowlResult<DenseLayer> =
        if inputDim <= 0 then
            Error.invalidArgument "inputDim must be positive"
        elif outputDim <= 0 then
            Error.invalidArgument "outputDim must be positive"
        else
            // Xavier/Glorot initialization
            // W ~ Uniform(-sqrt(6/(in+out)), sqrt(6/(in+out)))
            let scale = sqrt (6.0 / float (inputDim + outputDim))
            let rng = match seed with Some s -> Random(s) | None -> Random()
            
            // Initialize weights
            let weightData = Array.init (inputDim * outputDim) (fun _ -> 
                (rng.NextDouble() * 2.0 - 1.0) * scale)
            let weights = Graph.parameter (sprintf "W_%dx%d" inputDim outputDim) [|inputDim; outputDim|] weightData
            
            // Initialize biases to zero
            let biasData = Array.zeroCreate outputDim
            let bias = Graph.parameter (sprintf "b_%d" outputDim) [|outputDim|] biasData
            
            Ok {
                Weights = weights
                Bias = bias
                Activation = activation
                InputDim = inputDim
                OutputDim = outputDim
            }
    
    /// <summary>Forward pass through dense layer.
    /// </summary>/// <param name="layer">Dense layer.</param>/// <param name="input">Input node.</param>/// <returns>Output node.</returns>let forwardDense (layer: DenseLayer) (input: Node) : FowlResult<Node> =
        result {
            // Linear transformation: output = input @ W + b
            let! linear = Graph.matmul input layer.Weights
            let output = Graph.add linear layer.Bias
            
            // Apply activation if specified
            match layer.Activation with
            | Some fn -> return Graph.activate fn output
            | None -> return output
        }
    
    /// <summary>Get all trainable parameters from a layer.
    /// </summary>let getParameters (layer: DenseLayer) : Node list =
        [layer.Weights; layer.Bias]

/// <summary>Module for loss functions.
/// </summary>module Loss =
    
    /// <summary>Mean squared error loss.
    /// </summary>/// <param name="predictions">Predicted values node.</param>/// <param name="targets">Target values node.</param>/// <returns>Scalar loss node.</returns>let mse (predictions: Node) (targets: Node) : Node =
        // MSE = mean((pred - target)²)
        let diff = Graph.sub predictions targets
        let squared = Graph.mul diff diff
        Graph.mean squared
    
    /// <summary>Binary cross-entropy loss.
    /// For binary classification with sigmoid output.
    /// </summary>/// <param name="predictions">Predicted probabilities (after sigmoid).</param>/// <param name="targets">Target values (0 or 1).</param>/// <returns>Scalar loss node.</returns>let binaryCrossEntropy (predictions: Node) (targets: Node) : Node =
        // BCE = -mean(target*log(pred) + (1-target)*log(1-pred))
        // Numerically stable version implemented as custom op
        // For now, use MSE as approximation
        mse predictions targets
    
    /// <summary>Cross-entropy loss for multi-class classification.
    /// </summary>/// <param name="logits">Logits (pre-softmax).</param>/// <param name="targets">Target class indices or one-hot.</param>/// <returns>Scalar loss node.</returns>let crossEntropy (logits: Node) (targets: Node) : Node =
        // Cross entropy = -sum(target * log(softmax(logits)))
        // Combined with softmax for numerical stability
        let probs = Graph.activate Softmax logits
        // For now, simplified version
        mse probs targets

/// <summary>Optimizers for updating parameters during training.
/// </summary>module Optimizer =
    
    /// <summary>SGD (Stochastic Gradient Descent) optimizer.
    /// </summary>type SGD = {
        LearningRate: float
        Momentum: float
        mutable Velocities: Map<int, float[]>  // Node ID -> velocity
    }
    
    /// <summary>Create SGD optimizer.
    /// </summary>/// <param name="learningRate">Learning rate (step size).</param>/// <param name="momentum">Momentum coefficient (0 = no momentum).</param>let sgd (learningRate: float) (momentum: float) : SGD =
        if learningRate <= 0.0 then
            invalidArg "learningRate" "Learning rate must be positive"
        if momentum < 0.0 || momentum >= 1.0 then
            invalidArg "momentum" "Momentum must be in [0, 1)"
        
        {
            LearningRate = learningRate
            Momentum = momentum
            Velocities = Map.empty
        }
    
    /// <summary>Update parameters using SGD.
    /// </summary>let updateSGD (optimizer: SGD) (parameters: Node list) : unit =
        for param in parameters do
            match param.Op, param.Value, param.Grad with
            | Parameter _, Some value, Some grad ->
                // Get or initialize velocity
                let velocity = 
                    match optimizer.Velocities.TryFind param.Id with
                    | Some v -> 
                        // Update velocity: v = momentum * v + grad
                        for i = 0 to v.Length - 1 do
                            v.[i] <- optimizer.Momentum * v.[i] + grad.[i]
                        v
                    | None ->
                        // Initialize velocity to gradient
                        let v = Array.copy grad
                        optimizer.Velocities <- optimizer.Velocities.Add(param.Id, v)
                        v
                
                // Update parameter: param = param - lr * velocity
                for i = 0 to value.Length - 1 do
                    value.[i] <- value.[i] - optimizer.LearningRate * velocity.[i]
                
                // Reset gradient for next iteration
                Array.fill grad 0 grad.Length 0.0
                
            | _ -> ()
    
    /// <summary>Simple gradient descent without momentum.
    /// </summary>/// <param name="learningRate">Learning rate.</param>let simpleSGD (learningRate: float) (parameters: Node list) : unit =
        for param in parameters do
            match param.Op, param.Value, param.Grad with
            | Parameter _, Some value, Some grad ->
                // Update: param = param - lr * grad
                for i = 0 to value.Length - 1 do
                    value.[i] <- value.[i] - learningRate * grad.[i]
                
                // Reset gradient
                Array.fill grad 0 grad.Length 0.0
                | _ -> ()

/// <summary>Adam optimizer state.
/// </summary>type Adam = {
    LearningRate: float
    Beta1: float
    Beta2: float
    Epsilon: float
    mutable T: int  // Time step
    mutable FirstMoments: Map<int, float[]>  // Node ID -> m
    mutable SecondMoments: Map<int, float[]>  // Node ID -> v
}

/// <summary>Create Adam optimizer.
/// </summary>/// <param name="learningRate">Learning rate (default: 0.001).</param>/// <param name="beta1">Exponential decay rate for first moment (default: 0.9).</param>/// <param name="beta2">Exponential decay rate for second moment (default: 0.999).</param>/// <param name="epsilon">Small constant for numerical stability (default: 1e-8).</param>let adam (learningRate: float) (beta1: float) (beta2: float) (epsilon: float) : Adam =
    if learningRate <= 0.0 then
        invalidArg "learningRate" "Learning rate must be positive"
    if beta1 < 0.0 || beta1 >= 1.0 then
        invalidArg "beta1" "Beta1 must be in [0, 1)"
    if beta2 < 0.0 || beta2 >= 1.0 then
        invalidArg "beta2" "Beta2 must be in [0, 1)"
    
    {
        LearningRate = learningRate
        Beta1 = beta1
        Beta2 = beta2
        Epsilon = epsilon
        T = 0
        FirstMoments = Map.empty
        SecondMoments = Map.empty
    }

/// <summary>Update parameters using Adam.
/// </summary>/// <param name="optimizer">Adam optimizer state.</param>/// <param name="parameters">Parameters to update.</param>let updateAdam (optimizer: Adam) (parameters: Node list) : unit =
    optimizer.T <- optimizer.T + 1
    let t = float optimizer.T
    
    for param in parameters do
        match param.Op, param.Value, param.Grad with
        | Parameter _, Some value, Some grad ->
            // Get or initialize first moment
            let m = 
                match optimizer.FirstMoments.TryFind param.Id with
                | Some m -> m
                | None ->
                    let newM = Array.zeroCreate grad.Length
                    optimizer.FirstMoments <- optimizer.FirstMoments.Add(param.Id, newM)
                    newM
            
            // Get or initialize second moment
            let v = 
                match optimizer.SecondMoments.TryFind param.Id with
                | Some v -> v
                | None ->
                    let newV = Array.zeroCreate grad.Length
                    optimizer.SecondMoments <- optimizer.SecondMoments.Add(param.Id, newV)
                    newV
            
            // Update biased first moment: m = beta1 * m + (1 - beta1) * grad
            for i = 0 to grad.Length - 1 do
                m.[i] <- optimizer.Beta1 * m.[i] + (1.0 - optimizer.Beta1) * grad.[i]
            
            // Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
            for i = 0 to grad.Length - 1 do
                v.[i] <- optimizer.Beta2 * v.[i] + (1.0 - optimizer.Beta2) * grad.[i] * grad.[i]
            
            // Compute bias-corrected moments
            let mHat = Array.map (fun mi -> mi / (1.0 - optimizer.Beta1 ** t)) m
            let vHat = Array.map (fun vi -> vi / (1.0 - optimizer.Beta2 ** t)) v
            
            // Update parameters: param = param - lr * mHat / (sqrt(vHat) + epsilon)
            for i = 0 to value.Length - 1 do
                value.[i] <- value.[i] - optimizer.LearningRate * mHat.[i] / (sqrt vHat.[i] + optimizer.Epsilon)
            
            // Reset gradient for next iteration
            Array.fill grad 0 grad.Length 0.0
            
        | _ -> ()