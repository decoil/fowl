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
5. [Statistics and Probability](chapter05.md)
6. [Optimization](chapter06.md)
7. [Algorithmic Differentiation](chapter07.md)

### Part III: Signal Processing
8. [Signal Processing](chapter08.md)
9. [Advanced Signal Processing](chapter09.md)

### Part IV: Machine Learning
10. [Neural Networks](chapter10.md)
11. [Advanced Neural Networks](chapter11.md)

### Part V: Real-World Applications
12. [Time Series Analysis](chapter12.md) *[Planned]*
13. [Image Processing](chapter13.md) *[Planned]*
14. [Case Studies](chapter14.md) *[Planned]*

### Appendices
- [A. API Reference](appendix-a.md) *[Planned]*
- [B. Performance Tuning](appendix-b.md) *[Planned]*
- [C. Interop with .NET](appendix-c.md) *[Planned]*
- [D. Migration from NumPy/Owl](appendix-d.md) *[Planned]*

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
