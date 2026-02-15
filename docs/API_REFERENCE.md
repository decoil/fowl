# Fowl API Reference

Complete API documentation for all Fowl modules.

---

## Table of Contents

1. [Fowl.Core](#fowlcore)
2. [Fowl.Linalg](#fowllinalg)
3. [Fowl.Stats](#fowlstats)
4. [Fowl.Neural](#fowlneural)
5. [Fowl.FFT](#fowlfft)
6. [Fowl.AD](#fowlad)
7. [Fowl.Optimization](#fowloptimization)

---

## Fowl.Core

Core types and ndarray operations.

### Types

#### `Ndarray<'T>`

N-dimensional array type.

```fsharp
type Ndarray<'T> = class end
```

**Properties:**
- `Shape : int[]` - Dimensions of the array
- `Size : int` - Total number of elements
- `Rank : int` - Number of dimensions

#### `FowlError`

Error type for all Fowl operations.

```fsharp
type FowlError =
    | InvalidArgument of string
    | DimensionMismatch of string
    | InvalidShape of string
    | SingularMatrix of string
    | NotImplemented of string
    | InvalidState of string
    | NativeLibraryError of string
    | UnknownError of string
```

#### `FowlResult<'T>`

Result type alias.

```fsharp
type FowlResult<'T> = Result<'T, FowlError>
```

### Ndarray Module

#### `zeros`

Creates array filled with zeros.

```fsharp
val zeros : shape:int[] -> FowlResult<Ndarray<float>>
```

**Parameters:**
- `shape` - Dimensions of the array

**Returns:** Ndarray filled with 0.0

**Example:**
```fsharp
let arr = Ndarray.zeros [|3; 4|] |> Result.get
// Creates 3x4 matrix of zeros
```

---

#### `ones`

Creates array filled with ones.

```fsharp
val ones : shape:int[] -> FowlResult<Ndarray<float>>
```

**Parameters:**
- `shape` - Dimensions of the array

**Returns:** Ndarray filled with 1.0

---

#### `eye`

Creates identity matrix.

```fsharp
val eye : n:int -> FowlResult<Ndarray<float>>
```

**Parameters:**
- `n` - Size of identity matrix (n x n)

**Returns:** Identity matrix

---

#### `ofArray`

Creates Ndarray from 1D array.

```fsharp
val ofArray : data:'T[] -> shape:int[] -> FowlResult<Ndarray<'T>>
```

**Parameters:**
- `data` - Source array
- `shape` - Target shape (product must equal data length)

**Returns:** Reshaped Ndarray

---

#### `ofArray2D`

Creates Ndarray from 2D array.

```fsharp
val ofArray2D : data:'T[,] -> FowlResult<Ndarray<'T>>
```

**Parameters:**
- `data` - 2D source array

**Returns:** Ndarray with shape [|rows; cols|]

---

#### `arange`

Creates array with evenly spaced values.

```fsharp
val arange : start:float -> stop:float -> step:float -> FowlResult<Ndarray<float>>
```

**Parameters:**
- `start` - Starting value
- `stop` - Ending value (exclusive)
- `step` - Step size

**Returns:** 1D Ndarray

---

#### `linspace`

Creates array with linearly spaced values.

```fsharp
val linspace : start:float -> stop:float -> num:int -> FowlResult<Ndarray<float>>
```

**Parameters:**
- `start` - Starting value
- `stop` - Ending value (inclusive)
- `num` - Number of samples

**Returns:** 1D Ndarray

---

#### `add`

Element-wise addition.

```fsharp
val add : Ndarray<float> -> Ndarray<float> -> FowlResult<Ndarray<float>>
```

**Parameters:**
- `a` - First array
- `b` - Second array (broadcastable to a's shape)

**Returns:** Element-wise sum

---

#### `sub`

Element-wise subtraction.

```fsharp
val sub : Ndarray<float> -> Ndarray<float> -> FowlResult<Ndarray<float>>
```

---

#### `mul`

Element-wise multiplication.

```fsharp
val mul : Ndarray<float> -> Ndarray<float> -> FowlResult<Ndarray<float>>
```

---

#### `div`

Element-wise division.

```fsharp
val div : Ndarray<float> -> Ndarray<float> -> FowlResult<Ndarray<float>>
```

---

#### `pow`

Element-wise power.

```fsharp
val pow : Ndarray<float> -> float -> FowlResult<Ndarray<float>>
```

---

#### `map`

Apply function element-wise.

```fsharp
val map : ('T -> 'U) -> Ndarray<'T> -> FowlResult<Ndarray<'U>>
```

---

#### `sum`

Sum all elements.

```fsharp
val sum : Ndarray<float> -> float
```

---

#### `mean`

Mean of all elements.

```fsharp
val mean : Ndarray<float> -> float
```

---

#### `max`

Maximum value.

```fsharp
val max : Ndarray<float> -> float
```

---

#### `min`

Minimum value.

```fsharp
val min : Ndarray<float> -> float
```

---

#### `reshape`

Reshape array to new dimensions.

```fsharp
val reshape : Ndarray<'T> -> int[] -> FowlResult<Ndarray<'T>>
```

---

#### `flatten`

Flatten to 1D.

```fsharp
val flatten : Ndarray<'T> -> FowlResult<Ndarray<'T>>
```

---

#### `get`

Get element at indices.

```fsharp
val get : Ndarray<'T> -> int[] -> FowlResult<'T>
```

---

#### `set`

Set element at indices.

```fsharp
val set : Ndarray<'T> -> int[] -> 'T -> unit
```

---

#### `shape`

Get shape of array.

```fsharp
val shape : Ndarray<'T> -> int[] option
```

---

#### `toArray`

Convert to 1D array.

```fsharp
val toArray : Ndarray<'T> -> 'T[]
```

---

#### `toArray2D`

Convert to 2D array.

```fsharp
val toArray2D : Ndarray<'T> -> FowlResult<'T[,]>
```

---

### Error Module

#### `invalidArgument`

Create InvalidArgument error.

```fsharp
val invalidArgument : string -> FowlResult<'T>
```

---

#### `dimensionMismatch`

Create DimensionMismatch error.

```fsharp
val dimensionMismatch : string -> FowlResult<'T>
```

---

#### `notImplemented`

Create NotImplemented error.

```fsharp
val notImplemented : string -> FowlResult<'T>
```

---

## Fowl.Linalg

Linear algebra operations.

### Matrix Module

#### `matmul`

Matrix multiplication.

```fsharp
val matmul : Ndarray<float> -> Ndarray<float> -> FowlResult<Ndarray<float>>
```

**Parameters:**
- `a` - Left matrix (m x n)
- `b` - Right matrix (n x p)

**Returns:** Product matrix (m x p)

---

#### `dot`

Dot product of two vectors.

```fsharp
val dot : Ndarray<float> -> Ndarray<float> -> FowlResult<float>
```

---

#### `outer`

Outer product of two vectors.

```fsharp
val outer : Ndarray<float> -> Ndarray<float> -> FowlResult<Ndarray<float>>
```

---

#### `transpose`

Matrix transpose.

```fsharp
val transpose : Ndarray<float> -> FowlResult<Ndarray<float>>
```

---

### Factorizations Module

#### `lu`

LU decomposition with partial pivoting.

```fsharp
val lu : Ndarray<float> -> FowlResult<Ndarray<float> * Ndarray<float> * Ndarray<float>>
```

**Returns:** (L, U, P) where P*A = L*U

---

#### `qr`

QR decomposition.

```fsharp
val qr : Ndarray<float> -> FowlResult<Ndarray<float> * Ndarray<float>>
```

**Returns:** (Q, R) where A = Q*R

---

#### `svd`

Singular Value Decomposition.

```fsharp
val svd : Ndarray<float> -> FowlResult<Ndarray<float> * float[] * Ndarray<float>>
```

**Returns:** (U, S, Vt) where A = U*diag(S)*Vt

---

#### `cholesky`

Cholesky decomposition.

```fsharp
val cholesky : Ndarray<float> -> FowlResult<Ndarray<float>>
```

**Returns:** L where A = L*L^T (A must be positive definite)

---

#### `eigSymmetric`

Eigenvalue decomposition for symmetric matrices.

```fsharp
val eigSymmetric : Ndarray<float> -> FowlResult<Ndarray<float> * Ndarray<float>>
```

**Returns:** (eigenvalues, eigenvectors)

---

#### `inv`

Matrix inverse.

```fsharp
val inv : Ndarray<float> -> FowlResult<Ndarray<float>>
```

---

#### `det`

Matrix determinant.

```fsharp
val det : Ndarray<float> -> FowlResult<float>
```

---

#### `solve`

Solve linear system Ax = b.

```fsharp
val solve : Ndarray<float> -> Ndarray<float> -> FowlResult<Ndarray<float>>
```

---

### AdvancedOps Module

#### `lstsq`

Least squares solution.

```fsharp
val lstsq : Ndarray<float> -> Ndarray<float> -> FowlResult<Ndarray<float> * float * int * float[]>
```

**Returns:** (x, residual, rank, singular_values)

---

#### `pinv`

Moore-Penrose pseudoinverse.

```fsharp
val pinv : Ndarray<float> -> FowlResult<Ndarray<float>>
```

---

#### `rank`

Matrix rank.

```fsharp
val rank : Ndarray<float> -> FowlResult<int>
```

---

#### `cond`

Condition number.

```fsharp
val cond : Ndarray<float> -> FowlResult<float>
```

---

## Fowl.Stats

Statistical functions and distributions.

### Descriptive Module

#### `mean`

Arithmetic mean.

```fsharp
val mean : float[] -> FowlResult<float>
```

---

#### `median`

Median value.

```fsharp
val median : float[] -> FowlResult<float>
```

---

#### `var`

Variance.

```fsharp
val var : float[] -> FowlResult<float>
```

---

#### `std`

Standard deviation.

```fsharp
val std : float[] -> FowlResult<float>
```

---

#### `percentile`

Percentile value.

```fsharp
val percentile : float[] -> float -> FowlResult<float>
```

---

#### `skewness`

Skewness.

```fsharp
val skewness : float[] -> FowlResult<float>
```

---

#### `kurtosis`

Kurtosis.

```fsharp
val kurtosis : float[] -> FowlResult<float>
```

---

### GaussianDistribution Module

#### `pdf`

Probability density function.

```fsharp
val pdf : mu:float -> sigma:float -> x:float -> FowlResult<float>
```

---

#### `cdf`

Cumulative distribution function.

```fsharp
val cdf : mu:float -> sigma:float -> x:float -> FowlResult<float>
```

---

#### `ppf`

Percent point function (inverse CDF).

```fsharp
val ppf : mu:float -> sigma:float -> p:float -> FowlResult<float>
```

---

#### `rvs`

Random variate sampling.

```fsharp
val rvs : mu:float -> sigma:float -> seed:int option -> FowlResult<float>
```

---

### HypothesisTests Module

#### `ttest_one_sample`

One-sample t-test.

```fsharp
val ttest_one_sample : float[] -> mu0:float -> FowlResult<TestResult>
```

**Returns:** `{ Statistic: float; PValue: float; DegreesOfFreedom: int }`

---

#### `ttest_independent`

Independent two-sample t-test.

```fsharp
val ttest_independent : float[] -> float[] -> FowlResult<TestResult>
```

---

#### `chi2_goodness`

Chi-square goodness of fit test.

```fsharp
val chi2_goodness : float[] -> float[] -> FowlResult<Chi2Result>
```

---

### Anova Module

#### `oneWay`

One-way ANOVA.

```fsharp
val oneWay : float[][] -> FowlResult<AnovaResult>
```

**Returns:** `{ FStatistic: float; PValue: float; DFBetween: int; DFWithin: int; ... }`

---

### Correlation Module

#### `pearsonCorrelation`

Pearson correlation coefficient.

```fsharp
val pearsonCorrelation : float[] -> float[] -> FowlResult<float>
```

---

#### `covariance`

Covariance.

```fsharp
val covariance : float[] -> float[] -> FowlResult<float>
```

---

#### `correlationMatrix`

Correlation matrix.

```fsharp
val correlationMatrix : float[][] -> FowlResult<float[,]>
```

---

## Fowl.Neural

Neural network components.

### Graph Module

#### `input`

Create input node.

```fsharp
val input : string -> int[] -> Node
```

---

#### `parameter`

Create parameter node.

```fsharp
val parameter : string -> int[] -> float[] -> Node
```

---

#### `constantArray`

Create constant node.

```fsharp
val constantArray : float[] -> int[] -> Node
```

---

#### `add`

Addition operation.

```fsharp
val add : Node -> Node -> Node
```

---

#### `mul`

Multiplication operation.

```fsharp
val mul : Node -> Node -> Node
```

---

#### `matmul`

Matrix multiplication.

```fsharp
val matmul : Node -> Node -> FowlResult<Node>
```

---

#### `activate`

Apply activation function.

```fsharp
val activate : ActivationType -> Node -> Node
```

---

### Layers Module

#### `dense`

Create dense (fully connected) layer.

```fsharp
val dense : int -> int -> ActivationType option -> int option -> FowlResult<DenseLayer>
```

**Parameters:**
- `inputSize` - Number of input features
- `outputSize` - Number of output features
- `activation` - Optional activation function
- `seed` - Random seed for initialization

---

#### `forwardDense`

Forward pass through dense layer.

```fsharp
val forwardDense : DenseLayer -> Node -> FowlResult<Node>
```

---

### RecurrentLayers Module

#### `createLSTM`

Create LSTM layer.

```fsharp
val createLSTM : int -> int -> ?numLayers:int -> ?dropout:float -> ?seed:int -> FowlResult<LSTMLayer>
```

---

#### `lstmForward`

LSTM forward pass.

```fsharp
val lstmForward : LSTMLayer -> float[][][] -> float[][][]
```

---

#### `createGRU`

Create GRU layer.

```fsharp
val createGRU : int -> int -> ?numLayers:int -> ?dropout:float -> ?seed:int -> FowlResult<GRULayer>
```

---

### Optimizer Module

#### `sgd`

Stochastic gradient descent.

```fsharp
val sgd : float -> float -> Optimizer
```

---

#### `adam`

Adam optimizer.

```fsharp
val adam : float -> float -> float -> float -> Optimizer
```

---

## Fowl.FFT

Fast Fourier Transform operations.

### FFT Module

#### `fft`

Fast Fourier Transform.

```fsharp
val fft : Complex[] -> FowlResult<Complex[]>
```

---

#### `ifft`

Inverse FFT.

```fsharp
val ifft : Complex[] -> FowlResult<Complex[]>
```

---

#### `rfft`

Real FFT (optimized for real input).

```fsharp
val rfft : float[] -> FowlResult<Complex[]>
```

---

#### `psd`

Power spectral density.

```fsharp
val psd : Complex[] -> float[]
```

---

#### `fftfreq`

FFT frequency bins.

```fsharp
val fftfreq : int -> float -> float[]
```

---

## Fowl.AD

Algorithmic differentiation.

### Dual Module

#### `diff`

Compute derivative.

```fsharp
val diff : (Dual -> Dual) -> float -> float
```

---

#### `grad`

Compute gradient.

```fsharp
val grad : (Dual[] -> Dual) -> float[] -> float[]
```

---

#### `hessian`

Compute Hessian matrix.

```fsharp
val hessian : (Dual[] -> Dual) -> float[] -> float[,]
```

---

## Fowl.Optimization

Optimization algorithms.

### GradientDescent Module

#### `minimize`

Minimize using gradient descent.

```fsharp
val minimize : (float[] -> float) -> (float[] -> float[]) -> float[] -> OptimizationOptions -> OptimizationResult
```

**Returns:** `{ X: float[]; Fun: float; Nit: int; Success: bool; Message: string }`

---

### Adam Module

#### `minimize`

Minimize using Adam.

```fsharp
val minimize : (float[] -> float) -> (float[] -> float[]) -> float[] -> ?beta1:float -> ?beta2:float -> ?epsilon:float -> OptimizationOptions -> OptimizationResult
```

---

### SimulatedAnnealing Module

#### `minimize`

Minimize using simulated annealing.

```fsharp
val minimize : (float[] -> float) -> float[] -> (float * float)[] -> ?initialTemp:float -> ?coolingRate:float -> OptimizationOptions -> OptimizationResult
```

---

## Type Abbreviations

```fsharp
type Complex = System.Numerics.Complex
type SliceSpec = int option * int option
type ActivationType = ReLU | Sigmoid | Tanh | Softmax | LeakyReLU | ELU
```

---

_Last Updated: 2026-02-15_