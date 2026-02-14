# Owl Tutorial Chapter 3: Slicing and Broadcasting

## Slicing

### Two Types
1. **Basic Slicing** - `[start; stop; step]` format per dimension
2. **Fancy Slicing** - uses `I`, `L`, `R` constructors for more control

### Index Constructors (Fancy Slicing)
```ocaml
type index =
  | I of int      (* single index *)
  | L of int list (* list of indices *)
  | R of int list (* index range [start; stop; step] *)
```

**F# equivalent:**
```fsharp
type Index =
    | I of int
    | L of int list
    | R of int list
```

### Core Functions
```ocaml
(* Basic slicing *)
val get_slice: int list list -> ('a, 'b) t -> ('a, 'b) t
val set_slice: int list list -> ('a, 'b) t -> ('a, 'b) t -> unit

(* Fancy slicing *)
val get_fancy: index list -> ('a, 'b) t -> ('a, 'b) t
val set_fancy: index list -> ('a, 'b) t -> ('a, 'b) t -> unit
```

### Extended Operators (OCaml 4.06+)
```ocaml
x.%{i; j; k}          (* indexing - get *)
x.%{i; j; k} <- v     (* indexing - set *)

x.${[0;4]; [6;-1]}    (* basic slicing - get *)
x.${[0;4]; [6;-1]} <- y  (* basic slicing - set *)

x.!{L [2;2;1]; R [6;-1]}  (* fancy slicing - get *)
x.!{L [2;2;1]; R [6;-1]} <- y  (* fancy slicing - set *)
```

**F# approach:** Use indexed properties or slice syntax:
```fsharp
// F# has built-in slice syntax
let x = array.[0..4, 6..^1]  // similar to basic slicing
// For fancy slicing, would need custom operators or methods
```

### Slicing Rules
1. `R [start; stop; step]` - range definition
2. Step can be positive or negative (not 0)
3. Negative indices: `a` → `n + a` where `n` is dimension size
4. Empty `R []` → all indices `[0; n-1; 1]`
5. Single `[start]` → specific index `[start; start; 1]`
6. Two `[start; stop]` → range with step inferred:
   - `start < stop` → step = 1
   - `start > stop` → step = -1
7. Missing dimensions → assume all indices (expanded)

### Examples
```ocaml
let x = Mat.sequential 5 5

(* Flip vertically *)
let flip x = Mat.get_slice [ [-1; 0]; [] ] x

(* Reverse all elements *)
let reverse x = Mat.get_slice [ [-1; 0]; [-1; 0] ] x

(* Rotate 90° clockwise *)
let rotate90 x = Mat.(transpose x |> get_slice [ []; [-1; 0] ])

(* Circular shift *)
let cshift x n =
  let c = Mat.col_num x in
  let h = Utils.Array.(range (c - n) (c - 1)) |> Array.to_list in
  let t = Utils.Array.(range 0 (c - n - 1)) |> Array.to_list in
  Mat.get_fancy [ R []; L (h @ t) ] x
```

## Broadcasting

### What Is It?
Implicitly matching shapes in binary operations without memory allocation for tiling.

### Shape Constraints
For each dimension, one must be true:
1. Both dimensions are equal, OR
2. Either dimension is 1

**Valid examples:**
```
x: [|2; 1; 3|]  y: [|1; 1; 1|] ✓
x: [|2; 1; 3|]  y: [|2; 3; 1|] ✓
x: [|2; 1; 3|]  y: [|2; 3; 3|] ✓
```

**Invalid examples:**
```
x: [|2; 1; 3|]  y: [|1; 1; 2|] ✗ (3 ≠ 2, neither is 1)
```

### Dimension Expansion
If y has fewer dimensions, prepend 1s:
```
x: [|2; 3; 4; 5|]
y: [|4; 5|] → [|1; 1; 4; 5|]
```

### Supported Operations
- Basic: `add`, `sub`, `mul`, `div`, `pow`
- Comparison: `elt_equal`, `elt_not_equal`, `elt_less`, etc.
- Other: `min2`, `max2`, `atan2`, `hypot`, `fmod`

## Copy vs View

**Owl:** Slicing returns a **copy** (modifying slice doesn't affect original)
**NumPy:** Slicing returns a **view** (modifying slice affects original)

This is a key design decision for Fowl - copies are safer but use more memory.

## Internal Implementation

Owl uses a `slice_pair` struct in C:
```c
struct slice_pair {
  int64_t dim;     // dimensions
  int64_t dep;     // recursion depth
  intnat *n;       // y's shape
  void *x;         // source/destination
  int64_t posx;    // current offset
  int64_t *ofsx;   // offset per dimension
  int64_t *incx;   // stride per dimension
  void *y;         // destination/source
  int64_t posy;
  int64_t *ofsy;
  int64_t *incy;
};
```

Algorithm: nested loops with cursor movement based on offsets and strides.

## F# Design Notes

### Slicing Syntax
F# has built-in slice syntax for arrays:
```fsharp
let arr = Array2D.zeroCreate 10 10
let slice = arr.[0..4, 6..]  // F# native syntax
```

For fancy slicing, need custom solution:
```fsharp
type SliceIndex =
    | Single of int
    | Range of int * int * int  // start, stop, step
    | List of int array

type NDArray with
    member x.GetSlice([<ParamArray>] indices: SliceIndex[]) = ...
```

### Broadcasting
Implement with SRTP or explicit type:
```fsharp
module Broadcasting =
    let broadcast (x: 'T[,]) (y: 'T[,]) =
        // Check shape constraints
        // Expand dimensions as needed
        // Return broadcasted pair
        failwith "TODO"
```

---
_Learned: 2026-02-13_
