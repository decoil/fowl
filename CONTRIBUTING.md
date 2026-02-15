# Contributing to Fowl

Thank you for your interest in contributing to Fowl! This document provides guidelines for contributing to the project.

## 🤝 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## 🚀 Getting Started

### Prerequisites

- .NET 8.0 SDK or later
- Git
- An IDE (VS Code, Rider, or Visual Studio)

### Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/fowl.git
cd fowl
dotnet build
```

### Running Tests

```bash
# Run all tests
dotnet test

# Run with verbose output
dotnet test --verbosity normal

# Run specific test
dotnet test --filter "FullyQualifiedName~NdarrayTests"
```

## 📋 Contribution Guidelines

### Code Style

We follow standard F# conventions:

- **Indentation**: 4 spaces
- **Naming**: PascalCase for types, camelCase for values
- **Documentation**: XML docs for all public APIs

```fsharp
/// <summary>
/// Brief description of what this does.
/// </summary>
/// <param name="x">Description of parameter.</param>
/// <returns>Description of return value.</returns>
let myFunction (x: int) : int =
    x + 1
```

### Important: XML Documentation Format

**Never** put closing XML tags on the same line as code:

```fsharp
// WRONG
/// </summary>let myFunction x = ...

// CORRECT
/// </summary>
let myFunction x = ...
```

### Commit Messages

Follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `style`: Formatting
- `chore`: Maintenance

Examples:
```
feat(linalg): add Cholesky decomposition
fix(core): handle edge case in slicing
docs: update README with examples
test(stats): add tests for distributions
```

### Testing Requirements

- All new features must include tests
- Maintain or improve code coverage
- Tests should be deterministic (use seeded RNGs)

### Pull Request Process

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Add tests
4. Ensure all tests pass
5. Update documentation if needed
6. Submit PR with clear description

## 🏗️ Project Structure

```
src/
├── Fowl.Core/       # Core ndarray operations
├── Fowl.Linq/       # Linear algebra
├── Fowl.Stats/      # Statistics
├── Fowl.FFT/        # Signal processing
├── Fowl.AD/         # Automatic differentiation
├── Fowl.Neural/     # Neural networks
└── ...

tests/
├── Fowl.Core.Tests/
├── Fowl.Linq.Tests/
└── ...
```

## 📝 Documentation

- Update relevant docs in `docs/` or `BOOK/`
- Add XML documentation to public APIs
- Include code examples

## 🐛 Reporting Bugs

When reporting bugs, please include:

- Fowl version
- .NET version
- Operating system
- Minimal code to reproduce
- Expected vs actual behavior

## 💡 Feature Requests

We welcome feature requests! Please:

- Check if it's already requested
- Describe the use case
- Provide examples if possible

## 🧪 Performance Contributions

When optimizing:

- Include benchmarks
- Show before/after comparison
- Ensure correctness is maintained

## 📚 Resources

- [F# Style Guide](https://docs.microsoft.com/en-us/dotnet/fsharp/style-guide/)
- [Owl Documentation](https://ocaml.xyz/)
- [Architecture Book](https://link.springer.com/book/10.1007/978-3-030-97636-9)

## ❓ Questions?

- Open a [GitHub Discussion](https://github.com/decoil/fowl/discussions)
- Join our [Discord](https://discord.gg/fowl)

Thank you for contributing! 🎉
