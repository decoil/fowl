# Fowl Examples

Comprehensive use case examples demonstrating Fowl's capabilities for scientific computing.

Based on patterns from **OCaml Scientific Computing** and practical applications.

---

## Examples

### Part III Use Cases (Deep Learning)

#### 1. ImageRecognition (Chapter 13)
**CNN architectures: LeNet, AlexNet, VGG, ResNet, SqueezeNet, InceptionV3**

```bash
cd examples/ImageRecognition
dotnet run
```

Features:
- LeNet-5: Classic CNN for MNIST (32x32 → 10 classes)
- AlexNet: ImageNet winner (224x224 → 1000 classes, 60M params)
- VGG family: VGG-11/13/16/19 with small 3x3 filters
- ResNet family: ResNet-18/34/50/101/152 with skip connections
- SqueezeNet: Efficient fire modules (1.2M params, 50x smaller)
- InceptionV3: Factorized convolutions and multi-branch architecture

**Output:**
```
Creating model architectures...

✓ LeNet-5 created (MNIST: 32x32 -> 10 classes)
✓ AlexNet created (ImageNet: 224x224x3 -> 1000 classes)
✓ VGG-16 created (13 conv + 3 FC layers)
✓ ResNet-50 created (50 layers with skip connections)
✓ SqueezeNet created (Fire modules, no FC layers)
✓ InceptionV3 created (factorized convolutions)

Architecture Summary:
| Model         | Year | Params | Key Innovation          |
|---------------|------|--------|-------------------------|
| LeNet-5       | 1998 | 60K    | First practical CNN     |
| AlexNet       | 2012 | 60M    | Deep + ReLU + Dropout   |
| VGG-16        | 2014 | 138M   | Small (3x3) filters     |
| ResNet-50     | 2015 | 25M    | Skip connections        |
| SqueezeNet    | 2016 | 1.2M   | 50x smaller than AlexNet|
| InceptionV3   | 2015 | 23M    | Factorized convolutions |
```

---

#### 2. InstanceSegmentation (Chapter 14)
**R-CNN evolution: R-CNN → Fast R-CNN → Faster R-CNN → Mask R-CNN**

```bash
cd examples/InstanceSegmentation
dotnet run
```

Features:
- R-CNN (2014): Selective search → CNN → SVM
- Fast R-CNN (2015): RoI pooling, single CNN pass
- Faster R-CNN (2015): Region Proposal Network (RPN)
- Mask R-CNN (2017): Adds instance segmentation masks
- Anchor generation (multi-scale, multi-aspect)
- RoI Align with bilinear interpolation
- Non-Maximum Suppression (NMS)

**Output:**
```
Architecture Evolution:

1. R-CNN (2014)
   - Selective search: ~2000 region proposals
   - CNN feature extraction per region
   - SVM classification + BBox regression
   - SLOW: ~50s per image (CPU), ~13s (GPU)

2. Fast R-CNN (2015)
   - Single CNN pass for entire image
   - RoI pooling: project regions to feature map
   - Joint training of classifier + regressor
   - FAST: ~2.3s per image

3. Faster R-CNN (2015)
   - Region Proposal Network (RPN)
   - Anchor boxes: multi-scale, multi-aspect
   - Shared features between RPN and detection
   - REAL-TIME: ~0.2s per image

4. Mask R-CNN (2017)
   - Adds mask prediction branch
   - RoI Align: bilinear interpolation (vs quantization)
   - Instance segmentation: pixel-level masks
   - STATE-OF-THE-ART: ~0.2s + mask output

Configuration:
  - Classes: 81 (80 COCO + background)
  - RoIs per image: 512
  - Mask size: 28x28
```

---

#### 3. NeuralStyleTransfer (Chapter 15)
**Content + Style transfer: Iterative (Gatys) + Fast (Johnson)**

```bash
cd examples/NeuralStyleTransfer
dotnet run
```

Features:
- Content reconstruction from deep VGG-19 features
- Style recreation via Gram matrices (multi-scale)
- Total variation loss for spatial smoothness
- Iterative style transfer: L_total = α*L_content + β*L_style + γ*L_tv
- Fast style transfer: Transform network (3 down + 5 residual + 3 up)

**Output:**
```
CONTENT RECONSTRUCTION
Reconstructing image from deep features:
  - Conv4_2: preserves high-level content (semantics)

STYLE RECREATION
Gram matrix captures texture statistics:
  G[i,j] = sum over space of F[i] * F[j]
Style layers: Conv1_1, Conv2_1, Conv3_1, Conv4_1, Conv5_1

ITERATIVE STYLE TRANSFER (Gatys et al., 2016)
Optimization objective:
  L_total = α * L_content + β * L_style + γ * L_tv

Typical parameters:
  - α (content weight): 1.0
  - β (style weight): 100-10000
  - γ (TV weight): 0.01-0.1
  - Steps: 300-1000

FAST STYLE TRANSFER (Johnson et al., 2016)
Transform network: 3 down + 5 residual + 3 up
Advantages: Real-time (~1000x faster than iterative)

Available Styles:
  - starry_night: Van Gogh's swirling night sky
  - scream: Munch's expressionist waves
  - udnie: Picasso's cubist composition
```

---

#### 4. RecommenderSystem (Chapter 16)
**Vector search: PCA, Random Projection, VP-Trees, LSH, Collaborative Filtering**

```bash
cd examples/RecommenderSystem
dotnet run
```

Features:
- Vector storage: Dense vectors with cosine/Euclidean/dot similarity
- PCA: Eigendecomposition-based dimensionality reduction
- Random Projection: Johnson-Lindenstrauss lemma (O(n) per vector)
- VP-Tree: Metric space partitioning with triangle inequality
- LSH: Locality Sensitive Hashing for approximate NN
- Collaborative Filtering: Matrix factorization (SVD) with SGD

**Output:**
```
DIMENSIONALITY REDUCTION
PCA (Principal Component Analysis):
  - Finds directions of maximum variance
  - Computationally expensive: O(n³)

Random Projection (Johnson-Lindenstrauss):
  - Fast: O(n) per vector
  - Preserves distances: (1±ε) with high probability
  - minDims = 4·log(n/δ)/(ε²/2 - ε³/3)

TREE-BASED SEARCH (VP-TREE)
Vantage Point Tree:
  - Metric space partitioning
  - Triangle inequality for pruning
  - Build: O(n log n), Query: O(log n) average

LOCALITY SENSITIVE HASHING (LSH)
Approximate nearest neighbors:
  - h(v) = floor((a·v + b) / r)
  - Multiple hash tables for recall
  - Build: O(n·L·k), Query: O(L·k + candidates)

COLLABORATIVE FILTERING
Matrix Factorization (SVD++):
  r̂_ui = μ + b_u + b_i + Σ_f U_uf · V_if
  Training: SGD on squared error

Performance Comparison:
Method                    | Build    | Query    | Approx?
-------------------------|----------|----------|----------
Brute Force              | O(1)     | O(n)     | Exact
VP-Tree                  | O(n log n)| O(log n)| Exact
LSH                      | O(nLk)   | O(Lk)    | Approx
Random Projection + Tree | O(n log n)| O(log n)| Approx
```

---

### Statistical & ML Examples

#### 5. FinancialAnalysis
**Time series analysis for stock data**

```bash
cd examples/FinancialAnalysis
dotnet run
```

Features:
- Simple and exponential moving averages
- Volatility calculation (annualized)
- Trend detection via linear regression
- Value at Risk (VaR)
- Jarque-Bera normality test

**Output:**
```
Simple Moving Average (20-day): 112.50
Exponential Moving Average: 113.20
Annualized Volatility: 18.45%
Trend: UPWARD (slope: 0.0234)
95% VaR: -2.34%
```

---

#### 6. LinearRegression
**Predictive modeling for housing prices**

```bash
cd examples/LinearRegression
dotnet run
```

Features:
- Feature normalization
- Multiple linear regression
- R-squared and RMSE evaluation
- Statistical significance testing
- Price prediction

**Output:**
```
Model Coefficients:
  Bias: 245000.00
  SqFt: 120.50
  Bedrooms: 15000.00
  Bathrooms: 25000.00

Model Performance:
  R-squared: 0.9845
  RMSE: $5234.20
```

---

#### 7. MonteCarlo
**Portfolio risk simulation**

```bash
cd examples/MonteCarlo
dotnet run
```

Features:
- Covariance matrix calculation
- Cholesky decomposition
- Correlated random sampling (Box-Muller)
- 10,000 scenario simulation
- VaR at 95% and 99% confidence

**Output:**
```
Expected Daily Return: 0.0156%
Annualized Volatility: 12.45%
95% VaR: -1.89%
99% VaR: -2.67%
```

---

#### 8. Clustering
**Customer segmentation with K-means**

```bash
cd examples/Clustering
dotnet run
```

Features:
- K-means algorithm
- Silhouette score calculation
- Optimal k selection
- Cluster interpretation

**Output:**
```
Optimal k = 4 (silhouette: 0.6234)

Cluster 0: 8 customers (26.7%)
  → Target Customers (high income, high spend)

Cluster 1: 12 customers (40.0%)
  → Average Customers

Final Silhouette Score: 0.6234
✅ Strong cluster structure
```

---

#### 9. SignalProcessing
**FFT and spectral analysis**

```bash
cd examples/SignalProcessing
dotnet run
```

Features:
- Discrete Fourier Transform (DFT)
- Power spectral density
- Peak detection
- Moving average and exponential filters
- Sine/square wave generation

**Output:**
```
Dominant Frequencies:
  1. 50.0 Hz (power: 245.67)
  2. 120.0 Hz (power: 62.34)

Moving Average Filter:
  Std reduction: 0.42 → 0.15 (64.3% reduction)
```

---

#### 10. Optimization
**Gradient descent and simulated annealing**

```bash
cd examples/Optimization
dotnet run
```

Features:
- Gradient descent optimizer
- Simulated annealing
- Rosenbrock function
- Beale function
- Convergence analysis

**Output:**
```
Rosenbrock Function:
  Gradient Descent:
    Solution: (0.999823, 0.999645)
    Value: 3.14e-08
    Iterations: 12453

  Simulated Annealing:
    Solution: (1.000012, 1.000024)
    Value: 1.45e-10
```

---

#### 11. PhysicsSimulation
**Projectile motion and pendulum dynamics**

```bash
cd examples/PhysicsSimulation
dotnet run
```

Features:
- Projectile motion with air resistance
- Simple pendulum simulation
- Energy conservation check
- Period vs amplitude analysis
- Numerical integration (Euler)

**Output:**
```
Projectile Motion:
  Range: 64.23 meters
  Max height: 15.67 meters
  Flight time: 3.42 seconds

Simple Pendulum:
  Measured period: 2.0073 seconds
  Theoretical: 2.0064 seconds
  Error: 0.04%

✅ Energy conserved (within numerical error)
```

---

## Running All Examples

```bash
# From repository root
for dir in examples/*/; do
  echo "Running $dir..."
  (cd "$dir" && dotnet run)
  echo ""
done
```

---

## Common Patterns

### Error Handling
All examples use Fowl's `Result` type:

```fsharp
match runAnalysis() with
| Ok () -> printfn "Success!"
| Error e -> printfn "Error: %A" e
```

### Module Usage
Examples demonstrate integration across Fowl modules:

- **Fowl.Core**: Ndarray operations, matrix manipulations
- **Fowl.Stats**: Distributions, hypothesis tests, descriptive stats
- **Fowl.Linalg**: Factorizations, linear solvers

### Performance
Examples include performance notes where relevant:
- FFT implementation is O(n²) for clarity (production: use FFTW)
- Monte Carlo uses 10k iterations (production: use 100k+)

---

## Extending Examples

Each example is self-contained and can be extended:

1. **Add visualization**: Integrate with Plotly.NET
2. **More data**: Load real datasets from CSV
3. **Parallelization**: Use Fowl.Parallel for speedup
4. **GPU acceleration**: Integrate with CUDA/OpenCL

---

## References

Based on:
- **OCaml Scientific Computing** - Use case patterns
- **Numerical Recipes** - Algorithms
- **Owl Tutorial** - API design

---

*Examples are continuously updated. Check back for more!*