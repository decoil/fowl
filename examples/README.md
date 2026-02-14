# Fowl Examples

Comprehensive use case examples demonstrating Fowl's capabilities for scientific computing.

Based on patterns from **OCaml Scientific Computing** and practical applications.

---

## Examples

### 1. FinancialAnalysis
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

### 2. LinearRegression
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

### 3. MonteCarlo
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

### 4. Clustering
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

### 5. SignalProcessing
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

### 6. Optimization
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

### 7. PhysicsSimulation
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