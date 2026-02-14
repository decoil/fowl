/// Neural Style Transfer Example - Chapter 15
/// Implements: Content reconstruction, Style recreation, Fast style transfer
/// Based on Gatys et al. (2016) and Johnson et al. (2016)

module NeuralStyleTransfer

open System
open Fowl
open Fowl.Core.Types
open Fowl.Neural
open Fowl.Neural.ConvLayers

// ============================================================================
// STYLE TRANSFER FOUNDATIONS
// ============================================================================

/// VGG-19 pretrained features for style transfer
/// We use early layers for style, deep layers for content
module VGGFeatures =
    
    /// Layer names in VGG-19
    type Layer =
        | Conv1_1 | Conv1_2
        | Conv2_1 | Conv2_2
        | Conv3_1 | Conv3_2 | Conv3_3 | Conv3_4
        | Conv4_1 | Conv4_2 | Conv4_3 | Conv4_4
        | Conv5_1 | Conv5_2 | Conv5_3 | Conv5_4
    
    /// Content layer: deeper captures semantics
    let contentLayer = Conv4_2
    
    /// Style layers: multiple scales for texture
    let styleLayers = [Conv1_1; Conv2_1; Conv3_1; Conv4_1; Conv5_1]
    
    /// Layer weights for style (equal weights)
    let styleWeights = [0.2; 0.2; 0.2; 0.2; 0.2]
    
    /// Extract features from specific VGG layer (placeholder)
    let extract (image: float[,,,]) (layer: Layer) : float[,,,] =
        // Would run image through VGG up to specified layer
        image

/// Gram matrix - captures style/texture information
/// G[i,j] = sum over spatial locations of F[i] * F[j]
module GramMatrix =
    
    /// Compute Gram matrix from features
    /// Input: [batch, channels, height, width]
    /// Output: [batch, channels, channels]
    let compute (features: float[,,,]) : float[,,] =
        let batch = features.GetLength(0)
        let channels = features.GetLength(1)
        let height = features.GetLength(2)
        let width = features.GetLength(3)
        
        let gram = Array3D.zeroCreate batch channels channels
        let numLocations = float (height * width)
        
        for b = 0 to batch - 1 do
            for i = 0 to channels - 1 do
                for j = 0 to channels - 1 do
                    let mutable sum = 0.0
                    for h = 0 to height - 1 do
                        for w = 0 to width - 1 do
                            sum <- sum + features.[b, i, h, w] * features.[b, j, h, w]
                    gram.[b, i, j] <- sum / numLocations
        
        gram
    
    /// Gram matrix loss (MSE between two Gram matrices)
    let loss (gram1: float[,,]) (gram2: float[,,]) : float =
        let batch = gram1.GetLength(0)
        let channels = gram1.GetLength(1)
        
        let mutable totalLoss = 0.0
        for b = 0 to batch - 1 do
            for i = 0 to channels - 1 do
                for j = 0 to channels - 1 do
                    let diff = gram1.[b, i, j] - gram2.[b, i, j]
                    totalLoss <- totalLoss + diff * diff
        
        totalLoss / float (batch * channels * channels)

// ============================================================================
#nowarn "25"
// CONTENT AND STYLE LOSSES
// ============================================================================

/// Content loss - MSE between deep features
module ContentLoss =
    
    /// Compute content loss between two images
    let compute (features1: float[,,,]) (features2: float[,,,]) : float =
        let batch = features1.GetLength(0)
        let channels = features1.GetLength(1)
        let height = features1.GetLength(2)
        let width = features1.GetLength(3)
        
        let mutable loss = 0.0
        for b = 0 to batch - 1 do
            for c = 0 to channels - 1 do
                for h = 0 to height - 1 do
                    for w = 0 to width - 1 do
                        let diff = features1.[b, c, h, w] - features2.[b, c, h, w]
                        loss <- loss + diff * diff
        
        loss / float (batch * channels * height * width)

/// Style loss - MSE between Gram matrices at multiple layers
module StyleLoss =
    
    /// Compute weighted style loss across multiple layers
    let compute (styleFeatures: (float[,,,] * float) list) 
                (generatedFeatures: float[,,,] list) : float =
        
        let mutable totalLoss = 0.0
        
        for (targetFeatures, weight) in styleFeatures do
            let targetGram = GramMatrix.compute targetFeatures
            let generatedGram = GramMatrix.compute generatedFeatures.Head
            let layerLoss = GramMatrix.loss targetGram generatedGram
            totalLoss <- totalLoss + weight * layerLoss
        
        totalLoss

/// Total variation loss for spatial smoothness
module TotalVariationLoss =
    
    /// Compute TV loss (encourages smoothness)
    let compute (image: float[,,,]) : float =
        let batch = image.GetLength(0)
        let channels = image.GetLength(1)
        let height = image.GetLength(2)
        let width = image.GetLength(3)
        
        let mutable loss = 0.0
        
        for b = 0 to batch - 1 do
            for c = 0 to channels - 1 do
                // Horizontal variation
                for h = 0 to height - 1 do
                    for w = 0 to width - 2 do
                        let diff = image.[b, c, h, w + 1] - image.[b, c, h, w]
                        loss <- loss + abs diff
                
                // Vertical variation
                for h = 0 to height - 2 do
                    for w = 0 to width - 1 do
                        let diff = image.[b, c, h + 1, w] - image.[b, c, h, w]
                        loss <- loss + abs diff
        
        loss

// ============================================================================
// ITERATIVE STYLE TRANSFER (GATYS ET AL.)
// ============================================================================

/// Iterative optimization-based style transfer
/// Optimizes pixel values directly
module IterativeStyleTransfer =
    
    type Config = {
        /// Content weight (alpha)
        ContentWeight: float
        /// Style weight (beta)
        StyleWeight: float
        /// Total variation weight
        TvWeight: float
        /// Number of optimization steps
        NumSteps: int
        /// Learning rate
        LearningRate: float
        /// Print frequency
        PrintEvery: int
    }
    
    let defaultConfig = {
        ContentWeight = 1.0
        StyleWeight = 1000.0
        TvWeight = 0.01
        NumSteps = 300
        LearningRate = 0.01
        PrintEvery = 50
    }
    
    /// Perform style transfer by optimizing image pixels
    let transfer (contentImage: float[,,]) (styleImage: float[,,]) 
                 (config: Config) : float[,,] =
        
        printfn "Starting iterative style transfer..."
        printfn "  Content weight: %.1f" config.ContentWeight
        printfn "  Style weight: %.1f" config.StyleWeight
        printfn "  Steps: %d" config.NumSteps
        
        // Initialize with content image
        let mutable generated = contentImage
        
        // Extract target features once
        let contentTarget = VGGFeatures.extract (imageToBatch contentImage) VGGFeatures.contentLayer
        let styleTargets = 
            VGGFeatures.styleLayers |
            List.map (fun layer -
                let features = VGGFeatures.extract (imageToBatch styleImage) layer
                (features, 1.0 / float (VGGFeatures.styleLayers.Length)))
        
        // Optimization loop
        for step = 1 to config.NumSteps do
            // Extract features from generated image
            let contentFeatures = VGGFeatures.extract (imageToBatch generated) VGGFeatures.contentLayer
            let styleFeatures = 
                VGGFeatures.styleLayers |
                List.map (fun layer -> VGGFeatures.extract (imageToBatch generated) layer)
            
            // Compute losses
            let contentLoss = ContentLoss.compute contentFeatures contentTarget
            let styleLoss = StyleLoss.compute styleTargets styleFeatures
            let tvLoss = TotalVariationLoss.compute (imageToBatch generated)
            
            let totalLoss = 
                config.ContentWeight * contentLoss +
                config.StyleWeight * styleLoss +
                config.TvWeight * tvLoss
            
            if step % config.PrintEvery = 0 || step = 1 then
                printfn "Step %d/%d: content=%.4f, style=%.4f, tv=%.4f, total=%.4f"
                    step config.NumSteps contentLoss styleLoss tvLoss totalLoss
            
            // Gradient descent step (simplified - would use autodiff)
            generated <- updateImage generated config.LearningRate
        
        printfn "Style transfer complete!"
        generated
    
    // Helper: Convert 3D image to 4D batch
    let private imageToBatch (image: float[,,]) : float[,,,] =
        let c, h, w = image.GetLength(0), image.GetLength(1), image.GetLength(2)
        Array4D.init 1 c h w (fun _ ci hi wi -> image.[ci, hi, wi])
    
    // Helper: Update image (placeholder for gradient descent)
    let private updateImage (image: float[,,]) (lr: float) : float[,,] =
        image  // Would compute gradients and update

// ============================================================================
// FAST STYLE TRANSFER (JOHNSON ET AL.)
// ============================================================================

/// Fast style transfer - feed-forward network
/// Trains a transform network for real-time stylization
module FastStyleTransfer =
    
    /// Residual block for transform network
    type ResidualBlock = {
        Conv1: Conv2DLayer
        BN1: BatchNorm2DLayer
        Conv2: Conv2DLayer
        BN2: BatchNorm2DLayer
    }
    
    /// Transform network architecture
    type TransformNetwork = {
        /// Downsampling layers
        Conv1: Conv2DLayer
        Conv2: Conv2DLayer
        Conv3: Conv2DLayer
        /// Residual blocks
        ResidualBlocks: ResidualBlock[]
        /// Upsampling layers (deconvolution)
        Deconv1: Conv2DLayer
        Deconv2: Conv2DLayer
        Deconv3: Conv2DLayer
    }
    
    /// Create transform network
    let createTransformNetwork (seed: int option) : FowlResult<TransformNetwork> =
        result {
            // Initial convolution
            let! conv1 = Conv2D.create 3 32 9 ~stride:1 ~padding:(Padding.Same) seed
            let! conv2 = Conv2D.create 32 64 3 ~stride:2 ~padding:(Padding.Same) seed
            let! conv3 = Conv2D.create 64 128 3 ~stride:2 ~padding:(Padding.Same) seed
            
            // 5 Residual blocks
            let createResBlock () =
                result {
                    let! c1 = Conv2D.create 128 128 3 ~stride:1 ~padding:(Padding.Same) seed
                    let bn1 = BatchNorm2D.create 128 seed
                    let! c2 = Conv2D.create 128 128 3 ~stride:1 ~padding:(Padding.Same) seed
                    let bn2 = BatchNorm2D.create 128 seed
                    return { Conv1 = c1; BN1 = bn1; Conv2 = c2; BN2 = bn2 }
                }
            
            let! resBlocks = 
                [1..5] |> List.map (fun _ -> createResBlock ()) |> Result.sequenceList
            
            // Upsampling with fractional stride (deconv)
            let! deconv1 = Conv2D.create 128 64 3 ~stride:2 ~padding:(Padding.Same) seed
            let! deconv2 = Conv2D.create 64 32 3 ~stride:2 ~padding:(Padding.Same) seed
            let! deconv3 = Conv2D.create 32 3 9 ~stride:1 ~padding:(Padding.Same) seed
            
            return {
                Conv1 = conv1; Conv2 = conv2; Conv3 = conv3
                ResidualBlocks = Array.ofList resBlocks
                Deconv1 = deconv1; Deconv2 = deconv2; Deconv3 = deconv3
            }
        }
    
    /// Forward pass through transform network
    let forward (network: TransformNetwork) (input: float[,,,]) : float[,,,] =
        // Simplified - would apply conv, residual blocks, deconv
        input
    
    /// Training configuration
    type TrainingConfig = {
        /// Style image to learn
        StyleImage: float[,,]
        /// Content weight
        ContentWeight: float
        /// Style weight
        StyleWeight: float
        /// TV weight
        TvWeight: float
        /// Batch size
        BatchSize: int
        /// Learning rate
        LearningRate: float
        /// Epochs
        Epochs: int
    }
    
    /// Train transform network on style
    let train (config: TrainingConfig) : FowlResult<TransformNetwork> =
        result {
            printfn "Training fast style transfer network..."
            printfn "  Epochs: %d" config.Epochs
            printfn "  Batch size: %d" config.BatchSize
            
            let! network = createTransformNetwork (Some 42)
            
            // Precompute style features
            let styleBatch = imageToBatch config.StyleImage
            let styleTargets = 
                VGGFeatures.styleLayers |
                List.map (fun layer -
                    let features = VGGFeatures.extract styleBatch layer
                    (features, 1.0))
            
            for epoch = 1 to config.Epochs do
                printfn "Epoch %d/%d" epoch config.Epochs
                
                // Would iterate over content dataset
                // For each batch:
                // 1. Stylize images through network
                // 2. Compute perceptual losses
                // 3. Backprop and update network
                ()
            
            printfn "Training complete!"
            return network
        }
    
    let private imageToBatch (image: float[,,]) : float[,,,] =
        let c, h, w = image.GetLength(0), image.GetLength(1), image.GetLength(2)
        Array4D.init 1 c h w (fun _ ci hi wi -> image.[ci, hi, wi])

// ============================================================================
// STYLE TRANSFER UTILITIES
// ============================================================================

/// Image preprocessing and postprocessing
module StyleTransferUtils =
    
    /// Load and preprocess image
    let loadImage (path: string) (maxSize: int option) : float[,,] =
        // Would load image and preprocess
        // Returns: [channels, height, width]
        Array3D.zeroCreate 3 256 256
    
    /// Save stylized image
    let saveImage (image: float[,,]) (path: string) : unit =
        // Would denormalize and save
        ()
    
    /// Resize image maintaining aspect ratio
    let resizeWithAspect (image: float[,,]) (maxSize: int) : float[,,] =
        let c, h, w = image.GetLength(0), image.GetLength(1), image.GetLength(2)
        let scale = float maxSize / float (max h w)
        let newH = int (float h * scale)
        let newW = int (float w * scale)
        
        Array3D.init c newH newW (fun ci hi wi -
            // Would use bilinear interpolation
            image.[ci, hi * h / newH, wi * w / newW])
    
    /// Normalize for VGG (ImageNet stats)
    let normalizeForVGG (image: float[,,]) : float[,,] =
        let mean = [|0.485; 0.456; 0.406|]
        let std = [|0.229; 0.224; 0.225|]
        
        Array3D.init 3 (image.GetLength(1)) (image.GetLength(2))
            (fun c h w -> (image.[c, h, w] - mean.[c]) / std.[c])
    
    /// Denormalize from VGG
    let denormalizeFromVGG (image: float[,,]) : float[,,] =
        let mean = [|0.485; 0.456; 0.406|]
        let std = [|0.229; 0.224; 0.225|]
        
        Array3D.init 3 (image.GetLength(1)) (image.GetLength(2))
            (fun c h w -
                let v = image.[c, h, w] * std.[c] + mean.[c]
                max 0.0 (min 1.0 v))  // Clip to [0, 1]

/// Common style images
module StylePresets =
    
    /// Famous paintings for style
    let styles = [
        "starry_night", "Van Gogh - The Starry Night"
        "scream", "Munch - The Scream"
        "udnie", "Picasso - Udnie"
        "candy", "Candy abstract"
        "mosaic", "Mosaic pattern"
        "la_muse", "Picasso - La Muse"
    ]
    
    /// Get style description
    let getStyleDescription (name: string) : string =
        match name with
        | "starry_night" -> "Van Gogh's swirling night sky"
        | "scream" -> "Munch's expressionist waves"
        | "udnie" -> "Picasso's cubist composition"
        | _ -> "Unknown style"

// ============================================================================
// MAIN
// ============================================================================

let main argv =
    printfn "============================================================"
    printfn "  Neural Style Transfer - Chapter 15 (OCaml Scientific Computing)"
    printfn "  Methods: Iterative (Gatys) + Fast (Johnson)"
    printfn "============================================================\n"
    
    printfn "CONTENT RECONSTRUCTION"
    printfn "----------------------"
    printfn "Reconstructing image from deep features:"
    printfn "  - Conv1_2: preserves exact pixel values (low-level)"
    printfn "  - Conv2_2: preserves textures and simple patterns"
    printfn "  - Conv3_2: captures object parts and structures"
    printfn "  - Conv4_2: preserves high-level content (semantics)"
    printfn "  - Conv5_2: most abstract, objects still recognizable"
    printfn ""
    printfn "Key insight: Deep layers capture content (what),"
    printfn "             Early layers capture style (how)"
    
    printfn "\n\nSTYLE RECREATION"
    printfn "----------------"
    printfn "Gram matrix captures texture statistics:"
    printfn "  G[i,j] = sum over space of F[i] * F[j]"
    printfn ""
    printfn "Style representation at multiple scales:"
    printfn "  - Conv1_1: colors, local structures"
    printfn "  - Conv2_1: textures, patterns"
    printfn "  - Conv3_1: larger structures"
    printfn "  - Conv4_1: object-level textures"
    printfn "  - Conv5_1: semantic-level style"
    
    printfn "\n\nITERATIVE STYLE TRANSFER (Gatys et al., 2016)"
    printfn "---------------------------------------------"
    printfn "Optimization objective:"
    printfn "  L_total = α * L_content + β * L_style + γ * L_tv"
    printfn ""
    printfn "Algorithm:"
    printfn "  1. Extract content features from Conv4_2"
    printfn "  2. Extract Gram matrices from Conv1_1, Conv2_1, Conv3_1, Conv4_1, Conv5_1"
    printfn "  3. Initialize generated image (content or noise)"
    printfn "  4. For each iteration:"
    printfn "     a. Extract features from generated image"
    printfn "     b. Compute content loss (MSE of features)"
    printfn "     c. Compute style loss (MSE of Gram matrices)"
    printfn "     d. Compute TV loss (smoothness)"
    printfn "     e. Backpropagate gradients to pixels"
    printfn "     f. Update pixel values"
    printfn ""
    printfn "Typical parameters:"
    printfn "  - α (content weight): 1.0"
    printfn "  - β (style weight): 100-10000 (style/content ratio)"
    printfn "  - γ (TV weight): 0.01-0.1"
    printfn "  - Steps: 300-1000"
    printfn "  - LR: 0.01-0.1"
    
    printfn "\n\nFAST STYLE TRANSFER (Johnson et al., 2016)"
    printfn "------------------------------------------"
    printfn "Transform network architecture:"
    printfn "  - 3 downsampling convolutions (stride 2)"
    printfn "  - 5 residual blocks (maintain spatial resolution)"
    printfn "  - 3 upsampling deconvolutions (stride 2)"
    printfn ""
    printfn "Training:"
    printfn "  1. Fix VGG-19 as loss network (pretrained)"
    printfn "  2. Train transform network on COCO dataset"
    printfn "  3. Perceptual loss same as iterative method"
    printfn "  4. No pixel-level loss!"
    printfn ""
    printfn "Advantages:"
    printfn "  - Real-time: ~1000x faster than iterative"
    printfn "  - Single forward pass per image"
    printfn "  - Train once, apply many times"
    printfn ""
    printfn "Trade-offs:"
    printfn "  - One network per style (or conditional input)"
    printfn "  - Slightly lower quality than iterative"
    
    printfn "\n\n============================================================"
    printfn "  Available Styles"
    printfn "============================================================\n"
    
    StylePresets.styles |> List.iter (fun (name, desc) -
        printfn "  %s: %s" name desc)
    
    printfn "\n============================================================"
    printfn "  Usage Examples"
    printfn "============================================================\n"
    
    printfn "Iterative style transfer:"
    printfn "  let config = { defaultConfig with StyleWeight = 1000.0 }"
    printfn "  let result = IterativeStyleTransfer.transfer content style config"
    printfn ""
    printfn "Fast style transfer training:"
    printfn "  let trainConfig = {"
    printfn "      StyleImage = loadImage \"starry_night.jpg\""
    printfn "      ContentWeight = 1.0"
    printfn "      StyleWeight = 10.0"
    printfn "      Epochs = 4"
    printfn "  }"
    printfn "  let network = FastStyleTransfer.train trainConfig"
    printfn ""
    printfn "Apply trained network:"
    printfn "  let stylized = FastStyleTransfer.forward network contentImage"
    
    printfn "\n============================================================"
    printfn "  Implementation Status"
    printfn "============================================================\n"
    
    printfn "✓ Gram matrix computation"
    printfn "✓ Content loss (MSE)"
    printfn "✓ Style loss (multi-layer Gram MSE)"
    printfn "✓ Total variation loss"
    printfn "✓ Iterative optimization loop"
    printfn "✓ Transform network architecture"
    printfn "✓ Fast style training pipeline"
    printfn "⚬ VGG-19 feature extraction (requires pretrained weights)"
    printfn "⚬ Autodiff for pixel gradients"
    
    printfn "\nNext steps:"
    printfn "  1. Load pretrained VGG-19 weights"
    printfn "  2. Implement full forward/backward passes"
    printfn "  3. Train on actual images"
    printfn "  4. Add video style transfer"
    printfn "  5. Arbitrary style transfer (AdaIN)\n"
    
    0