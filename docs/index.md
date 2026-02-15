# Fowl Documentation

Welcome to the Fowl documentation. This site contains comprehensive guides, API references, and examples for using Fowl - the high-performance numerical computing library for F#.

## Quick Links

- [Getting Started](getting-started.md)
- [API Reference](api-reference.md)
- [Tutorial Book](../BOOK/README.md)
- [Examples](../examples/)
- [GitHub Repository](https://github.com/decoil/fowl)

## Installation

```bash
dotnet add package Fowl
```

## Quick Example

```fsharp
open Fowl
open Fowl.Core.Types

// Create arrays
let a = Ndarray.zeros<Float64> [|3; 3|] |> unwrap
let b = Ndarray.ones<Float64> [|3; 3|] |> unwrap

// Operations
let c = Ndarray.add a b |> unwrap
```

## Documentation Structure

1. **User Guide** - Learn Fowl from scratch
2. **API Reference** - Complete API documentation
3. **Tutorials** - Step-by-step guides
4. **Case Studies** - Real-world applications
5. **Contributing** - How to contribute

## Support

- [GitHub Issues](https://github.com/decoil/fowl/issues)
- [Discussions](https://github.com/decoil/fowl/discussions)

---

*Built with F# and .NET*
