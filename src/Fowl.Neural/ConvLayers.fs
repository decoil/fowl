namespace Fowl.Neural

open System
open Fowl
open Fowl.Core.Types

/// <summary>2D Convolution layer for image processing.
/// Implements Conv2D with configurable kernel, stride, padding.
/// </summary>
type Conv2DLayer = {
    /// Input channels (e.g., 3 for RGB)
    InChannels: int
    /// Output channels (number of filters)
    OutChannels: int
    /// Kernel size (e.g., 3 for 3x3)
    KernelSize: int
    /// Stride for convolution
    Stride: int
    /// Padding: Same, Valid, or explicit int
    Padding: Padding
    /// Dilation rate
    Dilation: int
    /// Learnable filters [out_channels, in_channels, kernel_h, kernel_w]
    Weights: Node
    /// Learnable bias [out_channels]
    Bias: Node
}

/// <summary>Padding modes for convolution.
/// </summary>
and Padding =
    | Same      // Output same size as input (with stride=1)
    | Valid     // No padding
    | Explicit of int  // Specific padding amount

/// <summary>2D Pooling layer (MaxPool or AvgPool).
/// </summary>
type Pool2DLayer = {
    PoolType: PoolType
    KernelSize: int
    Stride: int
    Padding: Padding
}

/// <summary>Type of pooling operation.
/// </summary>
and PoolType = MaxPool | AvgPool

/// <summary>Batch Normalization layer for 2D data.
/// </summary>
type BatchNorm2DLayer = {
    NumFeatures: int
    /// Learnable scale parameter (gamma)
    Gamma: Node
    /// Learnable shift parameter (beta)
    Beta: Node
    /// Running mean (for inference)
    mutable RunningMean: float[]
    /// Running variance (for inference)
    mutable RunningVar: float[]
    /// Momentum for running statistics
    Momentum: float
    /// Epsilon for numerical stability
    Epsilon: float
}

/// <summary>Flatten layer to convert NCHW to 2D.
/// </summary>
type FlattenLayer = {
    StartDim: int
    EndDim: int
}

/// <summary>Module for computer vision layers.
/// </summary>
module Conv2D =
    
    /// <summary>Calculate output size for convolution.
    /// </summary>
    let private calculateOutputSize (inputSize: int) (kernelSize: int) 
                                       (stride: int) (padding: int) (dilation: int) : int =
        let dilatedKernel = (kernelSize - 1) * dilation + 1
        (inputSize + 2 * padding - dilatedKernel) / stride + 1
    
    /// <summary>Calculate padding for "Same" mode.
    /// </summary>
    let private calculateSamePadding (inputSize: int) (kernelSize: int) 
                                            (stride: int) : int =
        let outputSize = (inputSize + stride - 1) / stride
        let totalPadding = max 0 ((outputSize - 1) * stride + kernelSize - inputSize)
        totalPadding / 2
    
    /// <summary>Perform 2D convolution on a single channel.
    /// Naive implementation for clarity.
    /// </summary>
    let private conv2dSingleChannel (input: float[,]) (kernel: float[,])
                                      (stride: int) (padH: int) (padW: int) : float[,] =
        let inH = input.GetLength(0)
        let inW = input.GetLength(1)
        let kH = kernel.GetLength(0)
        let kW = kernel.GetLength(1)
        
        let outH = calculateOutputSize inH kH stride padH 1
        let outW = calculateOutputSize inW kW stride padW 1
        
        let output = Array2D.zeroCreate outH outW
        
        for oh = 0 to outH - 1 do
            for ow = 0 to outW - 1 do
                let mutable sum = 0.0
                for kh = 0 to kH - 1 do
                    for kw = 0 to kW - 1 do
                        let ih = oh * stride - padH + kh
                        let iw = ow * stride - padW + kw
                        if ih >= 0 && ih < inH && iw >= 0 && iw < inW then
                            sum <- sum + input.[ih, iw] * kernel.[kh, kw]
                output.[oh, ow] <- sum
        
        output
    
    /// <summary>Create a Conv2D layer with Xavier initialization.
    /// </summary>
    let create (inChannels: int) (outChannels: int) (kernelSize: int)
               ?(stride: int) ?(padding: Padding) ?(dilation: int)
               (seed: int option) : FowlResult<Conv2DLayer> =
        
        let stride = defaultArg stride 1
        let padding = defaultArg padding Valid
        let dilation = defaultArg dilation 1
        
        if inChannels <= 0 then
            Error.invalidArgument "inChannels must be positive"
        elif outChannels <= 0 then
            Error.invalidArgument "outChannels must be positive"
        elif kernelSize <= 0 then
            Error.invalidArgument "kernelSize must be positive"
        else
            // Xavier initialization for Conv2D
            // std = sqrt(2.0 / (in_channels * kernel_size^2))
            let fanIn = float (inChannels * kernelSize * kernelSize)
            let std = sqrt (2.0 / fanIn)
            let rng = match seed with Some s -> Random(s) | None -> Random()
            
            // Initialize weights: [out_channels, in_channels, kernel_h, kernel_w]
            let weightSize = outChannels * inChannels * kernelSize * kernelSize
            let weightData = Array.init weightSize (fun _ -> 
                (rng.NextDouble() * 2.0 - 1.0) * std)
            let weights = Graph.parameter 
                (sprintf "Conv2d_W_%d_%d" inChannels outChannels)
                [|outChannels; inChannels; kernelSize; kernelSize|]
                weightData
            
            // Initialize bias
            let biasData = Array.zeroCreate outChannels
            let bias = Graph.parameter
                (sprintf "Conv2d_b_%d" outChannels)
                [|outChannels|]
                biasData
            
            Ok {
                InChannels = inChannels
                OutChannels = outChannels
                KernelSize = kernelSize
                Stride = stride
                Padding = padding
                Dilation = dilation
                Weights = weights
                Bias = bias
            }
    
    /// <summary>Forward pass through Conv2D layer.
    /// Input shape: [batch, channels, height, width] (NCHW format)
    /// </summary>
    let forward (layer: Conv2DLayer) (input: Node) : FowlResult<Node> =
        // This would involve reshaping and convolution operations
        // For now, return a placeholder operation
        // Full implementation requires extensive tensor operations
        Error.notImplemented "Conv2D forward pass requires tensor operations - implementing"

/// <summary>2D Pooling operations.
/// </summary>
module Pool2D =
    
    /// <summary>Max pooling operation.
    /// </summary>
    let maxPool (input: float[,]) (kernelSize: int) (stride: int) : float[,] =
        let inH = input.GetLength(0)
        let inW = input.GetLength(1)
        let outH = (inH - kernelSize) / stride + 1
        let outW = (inW - kernelSize) / stride + 1
        
        let output = Array2D.zeroCreate outH outW
        
        for oh = 0 to outH - 1 do
            for ow = 0 to outW - 1 do
                let mutable maxVal = Double.NegativeInfinity
                for kh = 0 to kernelSize - 1 do
                    for kw = 0 to kernelSize - 1 do
                        let ih = oh * stride + kh
                        let iw = ow * stride + kw
                        if ih < inH && iw < inW then
                            maxVal <- max maxVal input.[ih, iw]
                output.[oh, ow] <- maxVal
        
        output
    
    /// <summary>Average pooling operation.
    /// </summary>
    let avgPool (input: float[,]) (kernelSize: int) (stride: int) : float[,] =
        let inH = input.GetLength(0)
        let inW = input.GetLength(1)
        let outH = (inH - kernelSize) / stride + 1
        let outW = (inW - kernelSize) / stride + 1
        
        let output = Array2D.zeroCreate outH outW
        let kernelArea = float (kernelSize * kernelSize)
        
        for oh = 0 to outH - 1 do
            for ow = 0 to outW - 1 do
                let mutable sum = 0.0
                let mutable count = 0
                for kh = 0 to kernelSize - 1 do
                    for kw = 0 to kernelSize - 1 do
                        let ih = oh * stride + kh
                        let iw = ow * stride + kw
                        if ih < inH && iw < inW then
                            sum <- sum + input.[ih, iw]
                            count <- count + 1
                output.[oh, ow] <- sum / float count
        
        output
    
    /// <summary>Create a 2D pooling layer.
    /// </summary>
    let create (poolType: PoolType) (kernelSize: int) ?(stride: int) ?(padding: Padding) : Pool2DLayer =
        let stride = defaultArg stride kernelSize
        let padding = defaultArg padding Valid
        
        {
            PoolType = poolType
            KernelSize = kernelSize
            Stride = stride
            Padding = padding
        }

/// <summary>Flatten operations for converting 4D to 2D.
/// </summary>
module Flatten =
    
    /// <summary>Create a flatten layer.
    /// </summary>
    let create (?startDim: int) (?endDim: int) : FlattenLayer =
        {
            StartDim = defaultArg startDim 1
            EndDim = defaultArg endDim -1
        }
    
    /// <summary>Flatten a 4D array [N,C,H,W] to 2D [N, C*H*W].
    /// </summary>
    let flatten4D (input: float[,,,]) : float[,] =
        let n = input.GetLength(0)
        let c = input.GetLength(1)
        let h = input.GetLength(2)
        let w = input.GetLength(3)
        let flatSize = c * h * w
        
        let output = Array2D.zeroCreate n flatSize
        
        for batch = 0 to n - 1 do
            let mutable idx = 0
            for ch = 0 to c - 1 do
                for row = 0 to h - 1 do
                    for col = 0 to w - 1 do
                        output.[batch, idx] <- input.[batch, ch, row, col]
                        idx <- idx + 1
        
        output

/// <summary>Batch Normalization for 2D data.
/// </summary>
module BatchNorm2D =
    
    /// <summary>Create BatchNorm2D layer.
    /// </summary>
    let create (numFeatures: int) ?(momentum: float) ?(epsilon: float) 
                  (seed: int option) : BatchNorm2DLayer =
        
        let momentum = defaultArg momentum 0.1
        let epsilon = defaultArg epsilon 1e-5
        let rng = match seed with Some s -> Random(s) | None -> Random()
        
        // Initialize gamma (scale) to 1, beta (shift) to 0
        let gammaData = Array.create numFeatures 1.0
        let betaData = Array.create numFeatures 0.0
        
        {
            NumFeatures = numFeatures
            Gamma = Graph.parameter "BN_gamma" [|numFeatures|] gammaData
            Beta = Graph.parameter "BN_beta" [|numFeatures|] betaData
            RunningMean = Array.zeroCreate numFeatures
            RunningVar = Array.create numFeatures 1.0
            Momentum = momentum
            Epsilon = epsilon
        }
    
    /// <summary>Forward pass for batch norm (training mode).
    /// Normalizes: y = (x - mean) / sqrt(var + eps) * gamma + beta
    /// </summary>
    let forwardTrain (layer: BatchNorm2DLayer) (input: float[,,,]) : float[,,,] =
        let n = input.GetLength(0)
        let c = input.GetLength(1)
        let h = input.GetLength(2)
        let w = input.GetLength(3)
        
        let output = Array4D.zeroCreate n c h w
        
        // Calculate mean and var per channel
        for ch = 0 to c - 1 do
            let mutable sum = 0.0
            let mutable sumSq = 0.0
            let count = float (n * h * w)
            
            for batch = 0 to n - 1 do
                for row = 0 to h - 1 do
                    for col = 0 to w - 1 do
                        let x = input.[batch, ch, row, col]
                        sum <- sum + x
                        sumSq <- sumSq + x * x
            
            let mean = sum / count
            let var = sumSq / count - mean * mean
            
            // Update running statistics
            layer.RunningMean.[ch] <- 
                (1.0 - layer.Momentum) * layer.RunningMean.[ch] + 
                layer.Momentum * mean
            layer.RunningVar.[ch] <- 
                (1.0 - layer.Momentum) * layer.RunningVar.[ch] + 
                layer.Momentum * var
            
            // Normalize
            let gamma = 1.0  // Would come from layer.Gamma
            let beta = 0.0   // Would come from layer.Beta
            
            for batch = 0 to n - 1 do
                for row = 0 to h - 1 do
                    for col = 0 to w - 1 do
                        let x = input.[batch, ch, row, col]
                        let normalized = (x - mean) / sqrt (var + layer.Epsilon)
                        output.[batch, ch, row, col] <- normalized * gamma + beta
        
        output

/// <summary>Helper operations for computer vision.
/// </summary>
module ImageOps =
    
    /// <summary>Resize image using bilinear interpolation.
    /// </summary>
    let resizeBilinear (input: float[,]) (newHeight: int) (newWidth: int) : float[,] =
        let inH = input.GetLength(0)
        let inW = input.GetLength(1)
        let output = Array2D.zeroCreate newHeight newWidth
        
        let scaleY = float (inH - 1) / float (newHeight - 1)
        let scaleX = float (inW - 1) / float (newWidth - 1)
        
        for y = 0 to newHeight - 1 do
            for x = 0 to newWidth - 1 do
                let srcY = float y * scaleY
                let srcX = float x * scaleX
                
                let y0 = int (floor srcY)
                let x0 = int (floor srcX)
                let y1 = min (y0 + 1) (inH - 1)
                let x1 = min (x0 + 1) (inW - 1)
                
                let dy = srcY - float y0
                let dx = srcX - float x0
                
                let v00 = input.[y0, x0]
                let v01 = input.[y0, x1]
                let v10 = input.[y1, x0]
                let v11 = input.[y1, x1]
                
                let v0 = v00 * (1.0 - dx) + v01 * dx
                let v1 = v10 * (1.0 - dx) + v11 * dx
                
                output.[y, x] <- v0 * (1.0 - dy) + v1 * dy
        
        output
    
    /// <summary>Normalize image to [0, 1] range.
    /// </summary>
    let normalize (input: float[,]) : float[,] =
        let minVal = 
            seq { for y in 0..input.GetLength(0)-1 do
                  for x in 0..input.GetLength(1)-1 do yield input.[y,x] }
            |> Seq.min
        let maxVal =
            seq { for y in 0..input.GetLength(0)-1 do
                  for x in 0..input.GetLength(1)-1 do yield input.[y,x] }
            |> Seq.max
        
        let range = maxVal - minVal
        if range = 0.0 then
            input
        else
            input |> Array2D.map (fun v -> (v - minVal) / range)
    
    /// <summary>Standardize image with mean and std.
    /// </summary>
    let standardize (input: float[,]) (mean: float) (std: float) : float[,] =
        input |> Array2D.map (fun v -> (v - mean) / std)