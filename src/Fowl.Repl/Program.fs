open System
open Fowl

printfn "Fowl - F# Numerical Computing Library"
printfn "======================================"
printfn ""

// Demo: Create arrays
let a = Ndarray.ones<Float64> [|3; 3|]
let b = Ndarray.create<Float64> [|3; 3|] 2.0

printfn "Array a (ones):"
for i = 0 to 2 do
    for j = 0 to 2 do
        printf "%f " (Ndarray.get a [|i; j|])
    printfn ""

printfn ""
printfn "Array b (twos):"
for i = 0 to 2 do
    for j = 0 to 2 do
        printf "%f " (Ndarray.get b [|i; j|])
    printfn ""

// Demo: Element-wise operations
let c = Ndarray.add a b
printfn ""
printfn "a + b:"
for i = 0 to 2 do
    for j = 0 to 2 do
        printf "%f " (Ndarray.get c [|i; j|])
    printfn ""

// Demo: linspace
let x = Ndarray.linspace 0.0 10.0 5
printfn ""
printfn "linspace(0, 10, 5): %A" (Ndarray.toArray x)

// Demo: slicing
let big = Ndarray.ones<Float64> [|5; 5|]
let sliced = Fowl.Core.Slice.slice big [|SliceSpec.Range(Some 1, Some 4, None); SliceSpec.All|]
printfn ""
printfn "Original shape: %A" (Ndarray.shape big)
printfn "Sliced shape: %A" (Ndarray.shape sliced)

printfn ""
printfn "Press any key to exit..."
Console.ReadKey() |> ignore
