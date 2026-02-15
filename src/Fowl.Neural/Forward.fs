namespace Fowl.Neural

open System
open Fowl
open Fowl.Core.Types

/// <summary>Module for executing the forward pass through the computation graph.
/// Computes values for all nodes in topological order.
/// </summary>
module Forward =
    
    /// <summary>Element-wise addition of two arrays.
    /// </summary>
    let private addArrays (a: float[]) (b: float[]) : float[] =
        Array.map2 (+) a b
    
    /// <summary>Element-wise subtraction of two arrays.
    /// </summary>
    let private subArrays (a: float[]) (b: float[]) : float[] =
        Array.map2 (-) a b
    
    /// <summary>Element-wise multiplication of two arrays.
    /// </summary>
    let private mulArrays (a: float[]) (b: float[]) : float[] =
        Array.map2 (*) a b
    
    /// <summary>Element-wise division of two arrays.
    /// </summary>
    let private divArrays (a: float[]) (b: float[]) : float[] =
        Array.map2 (/) a b
    
    /// <summary>Matrix multiplication for 2D arrays.
    /// Assumes row-major storage.
    /// </summary>
    let private matmul2D (a: float[]) (b: float[]) (m: int) (n: int) (p: int) : float[] =
        let result = Array.zeroCreate (m * p)
        for i = 0 to m - 1 do
            for j = 0 to p - 1 do
                let mutable sum = 0.0
                for k = 0 to n - 1 do
                    sum <- sum + a.[i * n + k] * b.[k * p + j]
                result.[i * p + j] <- sum
        result
    
    /// <summary>Apply activation function element-wise.
    /// </summary>
    let private applyActivation (fn: ActivationFn) (x: float[]) : float[] =
        match fn with
        | ReLU -> Array.map (max 0.0) x
        | Sigmoid -> Array.map (fun v -> 1.0 / (1.0 + exp (-v))) x
        | Tanh -> Array.map tanh x
        | Softmax ->
            // Stable softmax: subtract max before exp
            let maxVal = Array.max x
            let expShifted = Array.map (fun v -> exp (v - maxVal)) x
            let sumExp = Array.sum expShifted
            Array.map (fun v -> v / sumExp) expShifted
        | LeakyReLU alpha -> Array.map (fun v -> if v > 0.0 then v else alpha * v) x
        | ELU alpha -> Array.map (fun v -> if v > 0.0 then v else alpha * (exp v - 1.0)) x
        | Identity -> x
    
    /// <summary>Execute a single node's operation.
    /// </summary>
    let private executeOp (op: Operation) (parentValues: float[] list) : FowlResult<float[]> =
        match op, parentValues with
        | Const c, _ -> Ok [|c|]
        | ConstArray (data, _), _ -> Ok data
        | Parameter _, [v] -> Ok v  // Parameters already have values
        | Input _, [v] -> Ok v      // Inputs are provided
        | Add, [a; b] -> Ok (addArrays a b)
        | Sub, [a; b] -> Ok (subArrays a b)
        | Mul, [a; b] -> Ok (mulArrays a b)
        | Div, [a; b] -> Ok (divArrays a b)
        | MatMul, [a; b] ->
            // Assume 2D matrices for now
            Ok (matmul2D a b 1 1 1)  // Simplified - need proper shape handling
        | Sum _, [a] -> Ok [|Array.sum a|]
        | Mean _, [a] -> Ok [|Array.average a|]
        | Activation fn, [a] -> Ok (applyActivation fn a)
        | Dropout rate, [a] ->
            // During training: randomly zero out elements with probability 'rate'
            // Scale remaining elements by 1/(1-rate)
            let rng = Random()
            let scale = 1.0 / (1.0 - rate)
            let result = Array.map (fun x -> if rng.NextDouble() < rate then 0.0 else x * scale) a
            Ok result
        | _ -> Error.notImplemented (sprintf "Operation not yet implemented: %A" op)
    
    /// <summary>Execute forward pass starting from given nodes.
    /// Computes values for all nodes in topological order.
    /// </summary>
    let run (outputNodes: Node list) : FowlResult<unit> =
        let sorted = Graph.topologicalSort outputNodes
        
        let rec processNodes (nodes: Node list) =
            match nodes with
            | [] -> Ok ()
            | node :: rest ->
                // Skip if value already computed
                if node.Value.IsSome then
                    processNodes rest
                else
                    // Get parent values
                    let parentValues = 
                        node.Parents 
                        |> List.map (fun p -> p.Value) 
                        |> List.choose id
                    
                    if parentValues.Length <> node.Parents.Length then
                        Error.invalidState (sprintf "Node %d has uncomputed parents" node.Id)
                    else
                        match executeOp node.Op parentValues with
                        | Ok value ->
                            node.Value <- Some value
                            processNodes rest
                        | Error e -> Error e
        
        processNodes sorted
    
    /// <summary>Execute forward pass with input values.
    /// Sets input node values then computes the graph.
    /// </summary>
    let runWithInputs (outputNode: Node) (inputs: Map<string, float[]>) : FowlResult<float[]> =
        // Get all nodes in graph
        let sorted = Graph.topologicalSort [outputNode]
        
        // Set input values
        for node in sorted do
            match node.Op with
            | Input name when inputs.ContainsKey name ->
                node.Value <- Some inputs.[name]
            | _ -> ()
        
        // Run forward pass
        match run [outputNode] with
        | Ok () ->
            match outputNode.Value with
            | Some v -> Ok v
            | None -> Error.invalidState "Output node has no value after forward pass"
        | Error e -> Error e