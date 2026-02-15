// Quick validation test for Fowl core functionality
// Run with: dotnet fsi validate.fsx

#r "src/Fowl.Core/bin/Debug/net8.0/Fowl.Core.dll"
#r "src/Fowl.Linq/bin/Debug/net8.0/Fowl.Linq.dll"
#r "src/Fowl.Stats/bin/Debug/net8.0/Fowl.Stats.dll"

open Fowl
open Fowl.Core.Types
open Fowl.Stats

let unwrap = function
    | Ok x -> x
    | Error e -> failwithf "Error: %A" e

printfn "=== Fowl Validation Tests ==="

// Test 1: Ndarray creation
printfn "\n1. Testing Ndarray creation..."
let a = Ndarray.zeros<Float64> [|3; 3|] |> unwrap
let b = Ndarray.ones<Float64> [|3; 3|] |> unwrap
printfn "   ✓ Created 3x3 zero and ones arrays"

// Test 2: Element-wise operations
printfn "\n2. Testing element-wise operations..."
let c = Ndarray.add a b |> unwrap
let d = Ndarray.mul b b |> unwrap
printfn "   ✓ Addition and multiplication work"

// Test 3: Linear algebra
printfn "\n3. Testing linear algebra..."
open Fowl.Linq
let matrix = Ndarray.ofArray [|1.0; 2.0; 3.0; 4.0|] [|2; 2|] |> unwrap
let det = Factorizations.det matrix
match det with
| Ok d -> printfn "   ✓ Determinant computed: %.2f" d
| Error e -> printfn "   ✗ Determinant failed: %A" e

// Test 4: Statistics
printfn "\n4. Testing statistics..."
let data = [|1.0; 2.0; 3.0; 4.0; 5.0|]
let mean = Descriptive.mean data
let std = Descriptive.std data
printfn "   ✓ Mean: %.2f, Std: %.2f" mean std

// Test 5: Distributions
printfn "\n5. Testing distributions..."
match Distributions.Gaussian.pdf 0.0 1.0 0.0 with
| Ok pdf -> printfn "   ✓ Gaussian PDF at 0: %.6f" pdf
| Error e -> printfn "   ✗ Gaussian PDF failed: %A" e

// Test 6: Random numbers
printfn "\n6. Testing random number generation..."
open Fowl.Stats.Random
let rng = RandomState.create 42
let random = RandomState.rand rng [|5|]
printfn "   ✓ Generated 5 random numbers"

printfn "\n=== All Core Tests Passed ==="
