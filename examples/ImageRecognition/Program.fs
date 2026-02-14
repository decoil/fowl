/// Image Recognition Example - Chapter 13
/// Implements: LeNet, AlexNet, VGG, ResNet, SqueezeNet, InceptionV3

module ImageRecognition

open System
open Fowl
open Fowl.Core.Types
open Fowl.Neural
open Fowl.Neural.ConvLayers

// ============================================================================
// ARCHITECTURE DEFINITIONS
// ============================================================================

/// LeNet-5 (1998) - Classic CNN architecture
/// Input: 32x32 grayscale, Output: 10 classes
module LeNet =
    type Model = {
        Conv1: Conv2DLayer
        Pool1: Pool2DLayer
        Conv2: Conv2DLayer
        Pool2: Pool2DLayer
        FC1: Layers.Layer
        FC2: Layers.Layer
        FC3: Layers.Layer
    }
    
    /// Create LeNet-5 model
    let create (numClasses: int) (seed: int option) : FowlResult<Model> =
        result {
            let! conv1 = Conv2D.create 1 6 5 ~padding:(Padding.Valid) seed
            let pool1 = Pool2D.create MaxPool 2 ~stride:2
            let! conv2 = Conv2D.create 6 16 5 ~padding:(Padding.Valid) seed
            let pool2 = Pool2D.create MaxPool 2 ~stride:2
            
            // Fully connected layers
            let fc1 = Layers.dense 120 seed
            let fc2 = Layers.dense 84 seed
            let fc3 = Layers.dense numClasses seed
            
            return {
                Conv1 = conv1
                Pool1 = pool1
                Conv2 = conv2
                Pool2 = pool2
                FC1 = fc1
                FC2 = fc2
                FC3 = fc3
            }
        }
    
    /// Forward pass through LeNet
    let forward (model: Model) (input: float[,,,]) : float[] =
        // Simplified: in full implementation would use tensor operations
        // Returns flat features for classification
        let batchSize = input.GetLength(0)
        let features = batchSize * 16 * 5 * 5  // After conv/pool layers
        Array.zeroCreate (model.FC3.OutputSize)

/// AlexNet (2012) - Deep CNN that won ImageNet
/// Input: 224x224 RGB, Output: 1000 classes
module AlexNet =
    type Model = {
        Conv1: Conv2DLayer
        Pool1: Pool2DLayer
        Conv2: Conv2DLayer
        Pool2: Pool2DLayer
        Conv3: Conv2DLayer
        Conv4: Conv2DLayer
        Conv5: Conv2DLayer
        Pool3: Pool2DLayer
        FC1: Layers.Layer
        FC2: Layers.Layer
        FC3: Layers.Layer
    }
    
    let create (numClasses: int) (seed: int option) : FowlResult<Model> =
        result {
            // Layer 1: Conv(96, 11x11, stride 4) -> ReLU -> MaxPool(3x3, stride 2)
            let! conv1 = Conv2D.create 3 96 11 ~stride:4 ~padding:(Padding.Valid) seed
            let pool1 = Pool2D.create MaxPool 3 ~stride:2
            
            // Layer 2: Conv(256, 5x5, pad 2) -> ReLU -> MaxPool(3x3, stride 2)
            let! conv2 = Conv2D.create 96 256 5 ~stride:1 ~padding:(Padding.Same) seed
            let pool2 = Pool2D.create MaxPool 3 ~stride:2
            
            // Layer 3: Conv(384, 3x3, pad 1) -> ReLU
            let! conv3 = Conv2D.create 256 384 3 ~stride:1 ~padding:(Padding.Same) seed
            
            // Layer 4: Conv(384, 3x3, pad 1) -> ReLU
            let! conv4 = Conv2D.create 384 384 3 ~stride:1 ~padding:(Padding.Same) seed
            
            // Layer 5: Conv(256, 3x3, pad 1) -> ReLU -> MaxPool(3x3, stride 2)
            let! conv5 = Conv2D.create 384 256 3 ~stride:1 ~padding:(Padding.Same) seed
            let pool3 = Pool2D.create MaxPool 3 ~stride:2
            
            // Fully connected layers
            let fc1 = Layers.dense 4096 seed
            let fc2 = Layers.dense 4096 seed
            let fc3 = Layers.dense numClasses seed
            
            return {
                Conv1 = conv1; Pool1 = pool1
                Conv2 = conv2; Pool2 = pool2
                Conv3 = conv3; Conv4 = conv4
                Conv5 = conv5; Pool3 = pool3
                FC1 = fc1; FC2 = fc2; FC3 = fc3
            }
        }

/// VGG (2014) - Very Deep CNN with small 3x3 filters
/// Variants: VGG-11, VGG-13, VGG-16, VGG-19
module VGG =
    type Config = VGG11 | VGG13 | VGG16 | VGG19
    
    type Model = {
        Config: Config
        ConvLayers: Conv2DLayer list
        PoolLayers: Pool2DLayer list
        FC1: Layers.Layer
        FC2: Layers.Layer
        FC3: Layers.Layer
    }
    
    /// VGG architecture configuration
    /// Format: (channels, kernel_size, repeats)
    let private getConfig (config: Config) : (int * int * int) list =
        match config with
        | VGG11 -> 
            [(64, 3, 1); (128, 3, 1); (256, 3, 2); (512, 3, 2); (512, 3, 2)]
        | VGG13 ->
            [(64, 3, 2); (128, 3, 2); (256, 3, 2); (512, 3, 2); (512, 3, 2)]
        | VGG16 ->
            [(64, 3, 2); (128, 3, 2); (256, 3, 3); (512, 3, 3); (512, 3, 3)]
        | VGG19 ->
            [(64, 3, 2); (128, 3, 2); (256, 3, 4); (512, 3, 4); (512, 3, 4)]
    
    let create (config: Config) (numClasses: int) (seed: int option) : FowlResult<Model> =
        result {
            let arch = getConfig config
            let mutable inChannels = 3
            let mutable convLayers = []
            let mutable poolLayers = []
            
            for (outChannels, kernelSize, repeats) in arch do
                for _ = 1 to repeats do
                    let! conv = Conv2D.create inChannels outChannels kernelSize 
                                ~stride:1 ~padding:(Padding.Same) seed
                    convLayers <- conv :: convLayers
                    inChannels <- outChannels
                
                let pool = Pool2D.create MaxPool 2 ~stride:2
                poolLayers <- pool :: poolLayers
            
            let fc1 = Layers.dense 4096 seed
            let fc2 = Layers.dense 4096 seed
            let fc3 = Layers.dense numClasses seed
            
            return {
                Config = config
                ConvLayers = List.rev convLayers
                PoolLayers = List.rev poolLayers
                FC1 = fc1; FC2 = fc2; FC3 = fc3
            }
        }

/// ResNet (2015) - Deep residual learning with skip connections
/// Variants: ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
module ResNet =
    type Config = ResNet18 | ResNet34 | ResNet50 | ResNet101 | ResNet152
    
    /// Residual block (BasicBlock for ResNet-18/34, Bottleneck for deeper)
    type ResidualBlock = {
        Conv1: Conv2DLayer
        BN1: BatchNorm2DLayer
        Conv2: Conv2DLayer
        BN2: BatchNorm2DLayer
        Downsample: (Conv2DLayer * BatchNorm2DLayer) option
    }
    
    type Model = {
        Config: Config
        Conv1: Conv2DLayer
        BN1: BatchNorm2DLayer
        Layers: ResidualBlock list list  // Grouped by stage
        FC: Layers.Layer
    }
    
    /// Create a residual block
    let private createBlock (inChannels: int) (outChannels: int) 
                           (stride: int) (seed: int option) : FowlResult<ResidualBlock> =
        result {
            let! conv1 = Conv2D.create inChannels outChannels 3 
                        ~stride:stride ~padding:(Padding.Same) seed
            let bn1 = BatchNorm2D.create outChannels seed
            let! conv2 = Conv2D.create outChannels outChannels 3
                        ~stride:1 ~padding:(Padding.Same) seed
            let bn2 = BatchNorm2D.create outChannels seed
            
            // Downsample if dimensions change
            let downsample = 
                if stride > 1 || inChannels <> outChannels then
                    let convDs = Conv2D.create inChannels outChannels 1 
                                ~stride:stride ~padding:(Padding.Same) seed | Result.toOption
                    let bnDs = Some (BatchNorm2D.create outChannels seed)
                    match convDs with
                    | Some c -> Some (c, bnDs | Option.get)
                    | None -> None
                else
                    None
            
            return {
                Conv1 = conv1; BN1 = bn1
                Conv2 = conv2; BN2 = bn2
                Downsample = downsample
            }
        }
    
    /// ResNet configurations: (layers per stage)
    let private getLayers (config: Config) : int list =
        match config with
        | ResNet18 -> [2; 2; 2; 2]
        | ResNet34 -> [3; 4; 6; 3]
        | ResNet50 -> [3; 4; 6; 3]
        | ResNet101 -> [3; 4; 23; 3]
        | ResNet152 -> [3; 8; 36; 3]
    
    let create (config: Config) (numClasses: int) (seed: int option) : FowlResult<Model> =
        result {
            // Initial convolution
            let! conv1 = Conv2D.create 3 64 7 ~stride:2 ~padding:(Padding.Same) seed
            let bn1 = BatchNorm2D.create 64 seed
            
            // Build stages
            let layersPerStage = getLayers config
            let channels = [64; 128; 256; 512]
            let strides = [1; 2; 2; 2]
            
            let mutable allStages = []
            let mutable inCh = 64
            
            for stageIdx = 0 to 3 do
                let numBlocks = layersPerStage.[stageIdx]
                let outCh = channels.[stageIdx]
                let stride = strides.[stageIdx]
                
                let mutable stageBlocks = []
                
                // First block may downsample
                let! block1 = createBlock inCh outCh stride seed
                stageBlocks <- block1 :: stageBlocks
                inCh <- outCh
                
                // Remaining blocks
                for _ = 2 to numBlocks do
                    let! block = createBlock inCh outCh 1 seed
                    stageBlocks <- block :: stageBlocks
                
                allStages <- (List.rev stageBlocks) :: allStages
            
            let fc = Layers.dense numClasses seed
            
            return {
                Config = config
                Conv1 = conv1
                BN1 = bn1
                Layers = List.rev allStages
                FC = fc
            }
        }

/// SqueezeNet (2016) - Efficient architecture with fire modules
module SqueezeNet =
    /// Fire module: squeeze layer + expand layer
    type FireModule = {
        Squeeze: Conv2DLayer
        Expand1x1: Conv2DLayer
        Expand3x3: Conv2DLayer
    }
    
    type Model = {
        Conv1: Conv2DLayer
        FireModules: FireModule list
        Conv10: Conv2DLayer
    }
    
    let private createFire (inChannels: int) (squeezeSize: int) 
                          (expandSize: int) (seed: int option) : FowlResult<FireModule> =
        result {
            let! squeeze = Conv2D.create inChannels squeezeSize 1 seed
            let! expand1x1 = Conv2D.create squeezeSize expandSize 1 seed
            let! expand3x3 = Conv2D.create squeezeSize expandSize 3 
                            ~padding:(Padding.Same) seed
            
            return {
                Squeeze = squeeze
                Expand1x1 = expand1x1
                Expand3x3 = expand3x3
            }
        }
    
    let create (numClasses: int) (seed: int option) : FowlResult<Model> =
        result {
            // Initial conv
            let! conv1 = Conv2D.create 3 96 7 ~stride:2 ~padding:(Padding.Same) seed
            
            // Fire modules configuration: (squeeze, expand)
            let fireConfig = [
                (16, 64); (16, 64);  // fire2, fire3
                (32, 128); (32, 128);  // fire4, fire5
                (48, 192); (48, 192); (48, 192);  // fire6, fire7, fire8
                (64, 256); (64, 256);  // fire9, fire10
            ]
            
            let mutable fireModules = []
            let mutable inCh = 96
            
            for (sq, exp) in fireConfig do
                let! fire = createFire inCh sq exp seed
                fireModules <- fire :: fireModules
                inCh <- exp * 2  // Concatenated 1x1 and 3x3 outputs
            
            // Final conv (no FC layers!)
            let! conv10 = Conv2D.create inCh numClasses 1 seed
            
            return {
                Conv1 = conv1
                FireModules = List.rev fireModules
                Conv10 = conv10
            }
        }

/// Inception v3 (2015) - Factorized convolutions and auxiliary classifiers
module InceptionV3 =
    /// Inception module with factorized convolutions
    type InceptionModule = {
        Branch1x1: Conv2DLayer
        Branch5x5_1: Conv2DLayer
        Branch5x5_2: Conv2DLayer
        Branch3x3dbl_1: Conv2DLayer
        Branch3x3dbl_2: Conv2DLayer
        Branch3x3dbl_3: Conv2DLayer
        BranchPool: Conv2DLayer
    }
    
    /// Grid size reduction module
    type GridReduction = {
        Branch3x3: Conv2DLayer
        Branch3x3dbl_1: Conv2DLayer
        Branch3x3dbl_2: Conv2DLayer
        Branch3x3dbl_3: Conv2DLayer
    }
    
    type Model = {
        Conv2d_1a: Conv2DLayer
        Conv2d_2a: Conv2DLayer
        Conv2d_2b: Conv2DLayer
        Conv2d_3b: Conv2DLayer
        Conv2d_4a: Conv2DLayer
        MixedBlocks: InceptionModule list
        GridReductions: GridReduction list
        FC: Layers.Layer
    }
    
    /// Create factorized 5x5 as two 3x3 convolutions
    let private createInception (inChannels: int) (ch1x1: int) 
                               (ch5x5_1: int) (ch5x5_2: int)
                               (ch3x3dbl_1: int) (ch3x3dbl_2: int) (ch3x3dbl_3: int)
                               (chPool: int) (seed: int option) : FowlResult<InceptionModule> =
        result {
            let! b1x1 = Conv2D.create inChannels ch1x1 1 seed
            
            // 5x5 factorized as 1x1 then 3x3 then 3x3 (actually use two 3x3s)
            let! b5x5_1 = Conv2D.create inChannels ch5x5_1 1 seed
            let! b5x5_2 = Conv2D.create ch5x5_1 ch5x5_2 3 ~padding:(Padding.Same) seed
            
            // Double 3x3 factorized
            let! b3x3dbl_1 = Conv2D.create inChannels ch3x3dbl_1 1 seed
            let! b3x3dbl_2 = Conv2D.create ch3x3dbl_1 ch3x3dbl_2 3 ~padding:(Padding.Same) seed
            let! b3x3dbl_3 = Conv2D.create ch3x3dbl_2 ch3x3dbl_3 3 ~padding:(Padding.Same) seed
            
            let! bPool = Conv2D.create inChannels chPool 1 seed
            
            return {
                Branch1x1 = b1x1
                Branch5x5_1 = b5x5_1
                Branch5x5_2 = b5x5_2
                Branch3x3dbl_1 = b3x3dbl_1
                Branch3x3dbl_2 = b3x3dbl_2
                Branch3x3dbl_3 = b3x3dbl_3
                BranchPool = bPool
            }
        }
    
    let create (numClasses: int) (seed: int option) : FowlResult<Model> =
        result {
            // Stem layers
            let! conv1 = Conv2D.create 3 32 3 ~stride:2 ~padding:(Padding.Valid) seed
            let! conv2 = Conv2D.create 32 32 3 ~padding:(Padding.Valid) seed
            let! conv3 = Conv2D.create 32 64 3 ~padding:(Padding.Same) seed
            let! conv4 = Conv2D.create 64 80 3 ~padding:(Padding.Valid) seed
            let! conv5 = Conv2D.create 80 192 3 ~padding:(Padding.Valid) seed
            
            // Inception blocks (simplified - full v3 has many more)
            let! inc1 = createInception 192 64 48 64 64 96 96 32 seed
            let! inc2 = createInception 256 64 48 64 64 96 96 64 seed
            
            let fc = Layers.dense numClasses seed
            
            return {
                Conv2d_1a = conv1
                Conv2d_2a = conv2
                Conv2d_2b = conv3
                Conv2d_3b = conv4
                Conv2d_4a = conv5
                MixedBlocks = [inc1; inc2]
                GridReductions = []
                FC = fc
            }
        }

// ============================================================================
// IMAGE PROCESSING
// ============================================================================

/// Image preprocessing utilities
module ImageProcessing =
    
    /// Normalize image to [-1, 1] range (common for pretrained models)
    let normalizeImageNet (image: float[,,]) : float[,,] =
        let mean = [|0.485; 0.456; 0.406|]
        let std = [|0.229; 0.224; 0.225|]
        let height = image.GetLength(1)
        let width = image.GetLength(2)
        
        Array3D.init 3 height width (fun c h w -
            (image.[c, h, w] - mean.[c]) / std.[c])
    
    /// Resize image using bilinear interpolation
    let resize (image: float[,,]) (targetHeight: int) (targetWidth: int) : float[,,] =
        let channels = image.GetLength(0)
        let result = Array3D.zeroCreate channels targetHeight targetWidth
        
        for c = 0 to channels - 1 do
            let channel = Array2D.init (image.GetLength(1)) (image.GetLength(2))
                          (fun h w -> image.[c, h, w])
            let resized = ImageOps.resizeBilinear channel targetHeight targetWidth
            for h = 0 to targetHeight - 1 do
                for w = 0 to targetWidth - 1 do
                    result.[c, h, w] <- resized.[h, w]
        
        result
    
    /// Center crop image
    let centerCrop (image: float[,,]) (cropHeight: int) (cropWidth: int) : float[,,] =
        let channels = image.GetLength(0)
        let height = image.GetLength(1)
        let width = image.GetLength(2)
        
        let startY = (height - cropHeight) / 2
        let startX = (width - cropWidth) / 2
        
        Array3D.init channels cropHeight cropWidth (fun c h w -
            image.[c, startY + h, startX + w])

// ============================================================================
// INFERENCE
// ============================================================================

/// Model inference utilities
module Inference =
    
    /// Run inference and return class probabilities
    let predict (model: 'T) (input: float[,,]) : float[] =
        // Placeholder - would actually run through network
        // Returns softmax probabilities
        Array.zeroCreate 1000
    
    /// Get top-k predictions
    let topK (probabilities: float[]) (k: int) : (int * float)[] =
        probabilities
        |> Array.mapi (fun i p -> (i, p))
        |> Array.sortByDescending snd
        |> Array.take (min k probabilities.Length)
    
    /// Load pretrained weights (placeholder)
    let loadWeights (model: 'T) (path: string) : FowlResult<unit> =
        // Would load from file
        Error.notImplemented "Weight loading requires serialization"

// ============================================================================
// MAIN
// ============================================================================

let main argv =
    printfn "============================================================"
    printfn "  Image Recognition - Chapter 13 (OCaml Scientific Computing)"
    printfn "  Implementing: LeNet, AlexNet, VGG, ResNet, SqueezeNet, InceptionV3"
    printfn "============================================================\n"
    
    let seed = Some 42
    let numClasses = 1000  // ImageNet
    
    // Create models
    printfn "Creating model architectures...\n"
    
    match LeNet.create 10 seed with
    | Ok lenet -
        printfn "✓ LeNet-5 created (MNIST: 32x32 -> 10 classes)"
    | Error e -
        printfn "✗ LeNet error: %s" e.Message
    
    match AlexNet.create numClasses seed with
    | Ok alexnet -
        printfn "✓ AlexNet created (ImageNet: 224x224x3 -> 1000 classes)"
    | Error e -
        printfn "✗ AlexNet error: %s" e.Message
    
    match VGG.create VGG.VGG16 numClasses seed with
    | Ok vgg -
        printfn "✓ VGG-16 created (13 conv + 3 FC layers)"
    | Error e -
        printfn "✗ VGG error: %s" e.Message
    
    match ResNet.create ResNet.ResNet50 numClasses seed with
    | Ok resnet -
        printfn "✓ ResNet-50 created (50 layers with skip connections)"
    | Error e -
        printfn "✗ ResNet error: %s" e.Message
    
    match SqueezeNet.create numClasses seed with
    | Ok squeezenet -
        printfn "✓ SqueezeNet created (Fire modules, no FC layers)"
    | Error e -
        printfn "✗ SqueezeNet error: %s" e.Message
    
    match InceptionV3.create numClasses seed with
    | Ok inception -
        printfn "✓ InceptionV3 created (factorized convolutions)"
    | Error e -
        printfn "✗ InceptionV3 error: %s" e.Message
    
    printfn "\n============================================================"
    printfn "  Architecture Summary"
    printfn "============================================================"
    printfn "| Model         | Year | Params | Key Innovation          |"
    printfn "|---------------|------|--------|-------------------------|"
    printfn "| LeNet-5       | 1998 | 60K    | First practical CNN     |"
    printfn "| AlexNet       | 2012 | 60M    | Deep + ReLU + Dropout   |"
    printfn "| VGG-16        | 2014 | 138M   | Small (3x3) filters     |"
    printfn "| ResNet-50     | 2015 | 25M    | Skip connections        |"
    printfn "| SqueezeNet    | 2016 | 1.2M   | 50x smaller than AlexNet|"
    printfn "| InceptionV3   | 2015 | 23M    | Factorized convolutions |"
    printfn "============================================================\n"
    
    printfn "Next steps:"
    printfn "  1. Implement complete forward passes for each architecture"
    printfn "  2. Add weight loading from pretrained models"
    printfn "  3. Process actual images for inference"
    printfn "  4. Benchmark inference speed\n"
    
    0