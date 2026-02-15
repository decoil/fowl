# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-15

### Added

#### Core Features
- **Ndarray**: N-dimensional arrays with broadcasting, slicing, and indexing
- **Linear Algebra**: LU, QR, SVD, Cholesky, eigenvalue decompositions
- **Statistics**: 14+ distributions, hypothesis tests, regression
- **Signal Processing**: FFT, DCT, filters, convolution
- **Optimization**: Gradient descent, Adam, RMSprop, L-BFGS
- **Algorithmic Differentiation**: Forward and reverse mode AD
- **Neural Networks**: Computation graphs, layers, training

#### Documentation
- Comprehensive tutorial book (Chapters 1-6, 8, 11)
- API reference documentation
- Case studies and examples
- Architecture documentation

#### Performance
- SIMD acceleration (AVX2, SSE2)
- Parallel operations
- Cache-optimized algorithms
- Memory pooling

#### Testing
- Unit tests for Core, Linalg, Stats, AD, Neural, Optimization
- Property-based tests with FsCheck
- Integration tests

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- XML documentation formatting throughout codebase
- Error handling with Result types

### Security
- N/A (initial release)

## Future Releases

### [0.2.0] - Planned
- Complete Data module (CSV type provider)
- GPU acceleration (CUDA)
- More neural network layers
- Additional optimization algorithms

### [0.3.0] - Planned
- Distributed computing support
- AutoML features
- Model serving capabilities

### [1.0.0] - Planned
- Stable API
- Complete documentation
- Production-ready performance
