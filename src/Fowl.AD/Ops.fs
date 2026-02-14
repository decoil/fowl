module Fowl.AD.Ops

open Fowl
open Fowl.Core.Types
open Fowl.AD.Types
open Fowl.AD.Core

/// Single Input Single Output operator builder
let build_siso (module S : Operator) =
    let rec f a =
        let ff = function
            | F x -> S.ff_f x
            | Arr x -> S.ff_arr x
            | _ -> failwithf "Invalid input to %s" S.label
        
        let fd a = f a
        
        let r a =
            let adjoint cp ca t = (S.dr (primal a) cp ca, a) :: t
            let register t = a :: t
            let label = S.label, [a]
            adjoint, register, label
        
        match a with
        | DF (ap, at, ai) ->
            let cp = fd ap
            DF (cp, S.df cp ap at, ai)
        | DR (ap, _, _, _, ai, _) ->
            let cp = fd ap
            DR (cp, ref (zero cp), r a, ref 0, ai, ref 0)
        | ap -> ff ap
    
    f

/// Pair Input Single Output operator builder
let build_piso (module S : Operator with type t = t) =
    let rec f a b =
        match a, b with
        | DF (ap, at, ai), DF (bp, bt, bi) when ai = bi ->
            let cp = f ap bp
            DF (cp, S.df cp ap at, ai)
        | DR (ap, _, _, _, ai, _), DR (bp, _, _, _, bi, _) when ai = bi ->
            let cp = f ap bp
            let r a b =
                let adjoint cp ca t = 
                    let abar, bbar = S.dr (primal a) (primal b) cp ca
                    (abar, a) :: (bbar, b) :: t
                let register t = a :: b :: t
                let label = S.label ^ "_d_d", [a; b]
                adjoint, register, label
            DR (cp, ref (zero cp), r a b, ref 0, ai, ref 0)
        | ap, bp ->
            match ap, bp with
            | F a, F b -> S.ff_f a b
            | Arr a, Arr b -
003e S.ff_arr a b
            | F a, Arr b -> S.ff_f_arr a b
            | Arr a, F b -> S.ff_arr_f a b
            | _ -> failwithf "Invalid inputs to %s" S.label
    
    f

/// Math operators
module Maths =
    /// Sine
    let sin =
        build_siso (module struct
            let label = "sin"
            let ff_f a = F (sin a)
            let ff_arr a = match Ndarray.map sin a with Arr x -> Arr x | x -> x
            let df _cp ap at = match mul at (cos ap) with Ok z -> z | Error e -> failwithf "df failed: %A" e
            let dr a _cp ca = match mul !ca (cos a) with Ok z -> z | Error e -> failwithf "dr failed: %A" e
        end)
    
    /// Cosine
    and cos =
        build_siso (module struct
            let label = "cos"
            let ff_f a = F (cos a)
            let ff_arr a = Arr (Ndarray.map cos a)
            let df _cp ap at = neg (mul at (sin ap))
            let dr a _cp ca = neg (mul !ca (sin a))
        end)
    
    /// Exponential
    and exp =
        build_siso (module struct
            let label = "exp"
            let ff_f a = F (exp a)
            let ff_arr a = Arr (Ndarray.map exp a)
            let df cp _ap at = mul at cp
            let dr _a cp ca = mul !ca cp
        end)
    
    /// Natural logarithm
    and log =
        build_siso (module struct
            let label = "log"
            let ff_f a = F (log a)
            let ff_arr a = Arr (Ndarray.map log a)
            let df _cp ap at = div at ap
            let dr a _cp ca = div !ca a
        end)
    
    /// Negation
    and neg x =
        match x with
        | F a -> F (-a)
        | Arr a -> Arr (Ndarray.map (fun x -> -x) a)
        | DF (ap, at, ai) -> DF (neg ap, neg at, ai)
        | DR (ap, aa, op, af, ai, tracker) -> DR (neg ap, aa, op, af, ai, tracker)
    
    /// Addition (pair input)
    and add a b =
        match a, b with
        | F x, F y -> F (x + y)
        | Arr x, Arr y -
003e match Ndarray.add x y with Ok z -> Arr z | Error e -> failwithf "Add failed: %A" e
        | DF (ap, at, ai), DF (bp, bt, bi) when ai = bi ->
            DF (add ap bp, add at bt, ai)
        | DR (ap, aa, opa, afa, ai, trackera), DR (bp, ab, opb, afb, bi, trackerb) when ai = bi ->
            let cp = add ap bp
            let r =
                let adjoint cp ca t = (ca, a) :: (ca, b) :: t
                let register t = a :: b :: t
                let label = "add_d_d", [a; b]
                adjoint, register, label
            DR (cp, ref (zero cp), r, ref 0, ai, ref 0)
        | _ -> failwith "Type mismatch in add"
    
    /// Multiplication
    and mul a b =
        match a, b with
        | F x, F y -> F (x * y)
        | Arr x, Arr y -
003e match Ndarray.mul x y with Ok z -> Arr z | Error e -> failwithf "Mul failed: %A" e
        | DF (ap, at, ai), DF (bp, bt, bi) when ai = bi ->
            // Product rule: d(a*b) = da*b + a*db
            let term1 = mul at bp
            let term2 = mul ap bt
            DF (mul ap bp, add term1 term2, ai)
        | DR (ap, _, _, _, ai, _), DR (bp, _, _, _, bi, _) when ai = bi ->
            let cp = mul ap bp
            let r =
                let adjoint cp ca t =
                    let abar = mul !ca bp
                    let bbar = mul !ca ap
                    (abar, a) :: (bbar, b) :: t
                let register t = a :: b :: t
                let label = "mul_d_d", [a; b]
                adjoint, register, label
            DR (cp, ref (zero cp), r, ref 0, ai, ref 0)
        | _ -> failwith "Type mismatch in mul"
    
    /// Division
    and div a b =
        match a, b with
        | F x, F y -> F (x / y)
        | Arr x, Arr y -
003e match Ndarray.div x y with Ok z -> Arr z | Error e -> failwithf "Div failed: %A" e
        | DF (ap, at, ai), DF (bp, bt, bi) when ai = bi ->
            // Quotient rule: d(a/b) = (da*b - a*db) / b^2
            let num = sub (mul at bp) (mul ap bt)
            let den = mul bp bp
            DF (div ap bp, div num den, ai)
        | DR (ap, _, _, _, ai, _), DR (bp, _, _, _, bi, _) when ai = bi ->
            let cp = div ap bp
            let r =
                let adjoint cp ca t =
                    let abar = div !ca bp
                    let bbar = neg (mul !ca (div ap (mul bp bp)))
                    (abar, a) :: (bbar, b) :: t
                let register t = a :: b :: t
                let label = "div_d_d", [a; b]
                adjoint, register, label
            DR (cp, ref (zero cp), r, ref 0, ai, ref 0)
        | _ -> failwith "Type mismatch in div"
    
    /// Power
    and pow a b =
        match a, b with
        | F x, F y -> F (x ** y)
        | DF (ap, at, ai), DF (bp, bt, bi) when ai = bi ->
            // d(a^b) = a^b * (b/a * da + log(a) * db)
            let cp = pow ap bp
            let term1 = mul bp (div at ap)
            let term2 = mul bt (log ap)
            DF (cp, mul cp (add term1 term2), ai)
        | _ -> failwith "pow not fully implemented"
