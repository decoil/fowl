# Fowl: Hands-On Scientific Computing in F#

## A Comprehensive Tutorial Book

**Version:** 0.1.0  
**Date:** February 2026  
**Author:** Fowl Development Team

---

## Table of Contents

### Part I: Foundations
1. [Introduction to Fowl](chapter01.md)
2. [Ndarray Basics](chapter02.md)
3. [Array Manipulation](chapter03.md)
4. [Linear Algebra Fundamentals](chapter04.md)

### Part II: Statistics & Probability
5. [Descriptive Statistics](chapter05.md)
6. [Probability Distributions](chapter06.md)
7. [Hypothesis Testing](chapter07.md)

### Part III: Advanced Numerical Computing
8. [Signal Processing](chapter08.md)
9. [Optimization](chapter09.md)
10. [Algorithmic Differentiation](chapter10.md)

### Part IV: Machine Learning
11. [Neural Networks](chapter11.md)
12. [Deep Learning](chapter12.md)
13. [Regression & Classification](chapter13.md)

### Part V: Real-World Applications
14. [Time Series Analysis](chapter14.md)
15. [Image Processing](chapter15.md)
16. [Case Studies](chapter16.md)

### Appendices
- [A. API Reference](appendix-a.md)
- [B. Performance Tuning](appendix-b.md)
- [C. Interop with .NET](appendix-c.md)
- [D. Migration from NumPy/Owl](appendix-d.md)

---

## How to Use This Book

This book is designed for:
- **F# developers** getting started with numerical computing
- **Data scientists** transitioning from Python/R to F#
- **Researchers** needing high-performance functional computing
- **Students** learning scientific computing concepts

Each chapter includes:
- Conceptual explanations
- Working code examples
- Exercises with solutions
- Performance tips
- Common pitfalls

---

## Prerequisites

- Basic F# knowledge (types, functions, pattern matching)
- Familiarity with linear algebra concepts
- .NET 8.0 or later

---

## Code Examples

All code examples in this book are tested and runnable. You can find the complete source code in the `examples/` directory of the Fowl repository.

```fsharp
// Standard preamble for all examples
open Fowl
open Fowl.Core.Types

// Helper to unwrap Results
let unwrap = function
    | Ok x -> x
    | Error e -> failwithf "Error: %s" e.Message
```

---

## Getting Help

- GitHub Issues: https://github.com/decoil/fowl/issues
- Documentation: https://fowl.dev/docs
- Community Discord: https://discord.gg/fowl

---

*This book is a living document. Contributions welcome!*
