namespace Fowl.Neural

open System
open Fowl
open Fowl.Core.Types

/// <summary>Module for computing gradients via reverse-mode automatic differentiation.
/// Implements backpropagation through the computation graph.
/// </summary>module Backward =
    
    /// <summary>Initialize gradients for all nodes to zero.
    /// Called before starting backward pass.
    /// </summary>let initGrads (nodes: Node list) : unit =
        for node in nodes do
            if node.Grad.IsNone then
                let size = Shape.numel node.Shape
                node.Grad <- Some (Array.zeroCreate size)
    
    /// <summary>Reset all gradients to zero.
    /// Call before each training iteration.
    /// </summary>let resetGrads (nodes: Node list) : unit =
        for node in nodes do
            match node.Grad with
            | Some grad -> Array.fill grad 0 grad.Length 0.0
            | None -> ()
    
    /// <summary>Element-wise addition of gradients.
    /// Gradients accumulate from all children.
    /// </summary>let private accumGrad (grad: float[]) (delta: float[]) : unit =
        for i = 0 to grad.Length - 1 do
            grad.[i] <- grad.[i] + delta.[i]
    
    /// <summary>Compute gradient for element-wise addition.
    /// If z = x + y, then dz/dx = dz/dy = dz/dz
    /// </summary>let private gradAdd (outputGrad: float[]) (inputShape: Shape) : float[] =
        outputGrad  // Gradient flows through unchanged
    
    /// <summary>Compute gradient for element-wise subtraction.
    /// If z = x - y, then dz/dx = dz/dz, dz/dy = -dz/dz
    /// </summary>let private gradSub (isFirst: bool) (outputGrad: float[]) : float[] =
        if isFirst then
            outputGrad
        else
            Array.map (~-) outputGrad
    
    /// <summary>Compute gradient for element-wise multiplication.
    /// If z = x * y, then dz/dx = y * dz/dz, dz/dy = x * dz/dz
    /// </summary>let private gradMul (siblingValue: float[]) (outputGrad: float[]) : float[] =
        Array.map2 (*) siblingValue outputGrad
    
    /// <summary>Compute gradient for element-wise division.
    /// If z = x / y, then dz/dx = dz/dz / y, dz/dy = -x * dz/dz / y²
    /// </summary>let private gradDiv (isFirst: bool) (siblingValue: float[]) 
                          (thisValue: float[]) (outputGrad: float[]) : float[] =
        if isFirst then
            Array.map2 (/) outputGrad siblingValue
        else
            // -x / y² * grad
            Array.map3 (fun x y grad -> -x * grad / (y * y)) thisValue siblingValue outputGrad
    
    /// <summary>Compute gradient for matrix multiplication.
    /// If Z = X @ Y, then dZ/dX = grad @ Y^T, dZ/dY = X^T @ grad
    /// </summary>let private gradMatMul (isFirst: bool) (a: float[]) (b: float[])
                              (aShape: Shape) (bShape: Shape) (outputGrad: float[]) : float[] =
        // Simplified for 2D case
        // TODO: Handle arbitrary dimensions
        if isFirst then
            // grad @ B^T
            // grad: [M, P], B: [N, P], result: [M, N]
            let m = aShape.[0]
            let n = bShape.[0]
            let p = if bShape.Length > 1 then bShape.[1] else 1
            
            let result = Array.zeroCreate (m * n)
            for i = 0 to m - 1 do
                for j = 0 to n - 1 do
                    let mutable sum = 0.0
                    for k = 0 to p - 1 do
                        sum <- sum + outputGrad.[i * p + k] * b.[j * p + k]
                    result.[i * n + j] <- sum
            result
        else
            // A^T @ grad
            // A: [M, N], grad: [M, P], result: [N, P]
            let m = aShape.[0]
            let n = if aShape.Length > 1 then aShape.[1] else 1
            let p = if bShape.Length > 1 then bShape.[1] else 1
            
            let result = Array.zeroCreate (n * p)
            for i = 0 to n - 1 do
                for j = 0 to p - 1 do
                    let mutable sum = 0.0
                    for k = 0 to m - 1 do
                        sum <- sum + a.[k * n + i] * outputGrad.[k * p + j]
                    result.[i * p + j] <- sum
            result
    
    /// <summary>Compute gradient for activation functions.
    /// </summary>let private gradActivation (fn: ActivationFn) (input: float[]) (outputGrad: float[]) : float[] =
        match fn with
        | ReLU ->
            Array.map2 (fun x grad -> if x > 0.0 then grad else 0.0) input outputGrad
        | Sigmoid ->
            // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            // But we don't have output here, so recompute
            let sig x = 1.0 / (1.0 + exp (-x))
            Array.map2 (fun x grad -> 
                let s = sig x
                grad * s * (1.0 - s)) input outputGrad
        | Tanh ->
            // tanh'(x) = 1 - tanh²(x)
            Array.map2 (fun x grad -> let t = tanh x in grad * (1.0 - t * t)) input outputGrad
        | Softmax ->
            // Softmax gradient is complex - depends on loss function
            // For now, assume softmax is followed by cross-entropy loss
            // which simplifies the gradient to (softmax - target)
            outputGrad  // Pass through, let loss handle it
        | LeakyReLU alpha ->
            Array.map2 (fun x grad -> if x > 0.0 then grad else alpha * grad) input outputGrad
        | ELU alpha ->
            Array.map2 (fun x grad -> 
                if x > 0.0 then grad else grad * alpha * exp x) input outputGrad
        | Identity ->
            outputGrad
    
    /// <summary>Compute gradient for sum operation.
    /// If z = sum(x), then dz/dx[i] = dz/dz for all i
    /// </summary>let private gradSum (inputShape: Shape) (outputGrad: float[]) : float[] =
        // Broadcast the scalar gradient to input shape
        let size = Shape.numel inputShape
        Array.init size (fun _ -> outputGrad.[0])
    
    /// <summary>Compute gradient for mean operation.
    /// If z = mean(x), then dz/dx[i] = dz/dz / n for all i
    /// </summary>let private gradMean (inputShape: Shape) (outputGrad: float[]) : float[] =
        let size = Shape.numel inputShape
        let factor = outputGrad.[0] / float size
        Array.init size (fun _ -> factor)
    
    /// <summary>Compute local gradients for an operation.
    /// Returns list of gradients for each parent.
    /// </summary>let private computeLocalGrads (node: Node) : FowlResult<float[] list> =
        match node.Op, node.Parents with
        | Const _, _ | ConstArray _, _ | Input _, _ ->
            Ok []  // No parents, no gradients to propagate
        | Parameter _, _ ->
            Ok []
        | Add, [x; y] ->
            match node.Grad with
            | Some grad -> Ok [gradAdd grad x.Shape; gradAdd grad y.Shape]
            | None -> Error.invalidState "No gradient for Add node"
        | Sub, [x; y] ->
            match node.Grad with
            | Some grad -> Ok [gradSub true grad; gradSub false grad]
            | None -> Error.invalidState "No gradient for Sub node"
        | Mul, [x; y] ->
            match node.Grad, x.Value, y.Value with
            | Some grad, Some xv, Some yv -> Ok [gradMul yv grad; gradMul xv grad]
            | _ -> Error.invalidState "Missing value or gradient for Mul node"
        | Div, [x; y] ->
            match node.Grad, x.Value, y.Value with
            | Some grad, Some xv, Some yv -> Ok [gradDiv true yv xv grad; gradDiv false yv xv grad]
            | _ -> Error.invalidState "Missing value or gradient for Div node"
        | MatMul, [x; y] ->
            match node.Grad, x.Value, y.Value with
            | Some grad, Some xv, Some yv -> Ok [gradMatMul true xv yv x.Shape y.Shape grad;
                                              gradMatMul false xv yv x.Shape y.Shape grad]
            | _ -> Error.invalidState "Missing value or gradient for MatMul node"
        | Activation fn, [x] ->
            match node.Grad, x.Value with
            | Some grad, Some xv -> Ok [gradActivation fn xv grad]
            | _ -> Error.invalidState "Missing value or gradient for Activation node"
        | Sum _, [x] ->
            match node.Grad with
            | Some grad -> Ok [gradSum x.Shape grad]
            | None -> Error.invalidState "No gradient for Sum node"
        | Mean _, [x] ->
            match node.Grad with
            | Some grad -> Ok [gradMean x.Shape grad]
            | None -> Error.invalidState "No gradient for Mean node"
        | _ ->
            Error.notImplemented (sprintf "Gradient not implemented for %A" node.Op)
    
    /// <summary>Execute backward pass starting from output nodes.
    /// Computes gradients for all nodes in reverse topological order.
    /// </summary>let run (outputNodes: Node list) : FowlResult<unit> =
        // Get all nodes in topological order
        let sorted = Graph.topologicalSort outputNodes
        
        // Initialize gradients
        initGrads sorted
        
        // Set output gradients to 1 (seed for backprop)
        for node in outputNodes do
            match node.Grad with
            | Some grad -> Array.fill grad 0 grad.Length 1.0
            | None -> ()
        
        // Process in reverse topological order
        let rec processReverse (nodes: Node list) =
            match nodes with
            | [] -> Ok ()
            | node :: rest ->
                // Compute local gradients
                match computeLocalGrads node with
                | Ok localGrads -
                    // Accumulate gradients to parents
                    List.iter2 (fun parent localGrad -
                        match parent.Grad with
                        | Some parentGrad -> accumGrad parentGrad localGrad
                        | None -> ()
                    ) node.Parents localGrads
                    
                    processReverse rest
                | Error e -> Error e
        
        processReverse (List.rev sorted)
    
    /// <summary>Get gradient for a specific node after backward pass.
    /// </summary>let getGrad (node: Node) : float[] option =
        node.Grad
    
    /// <summary>Get gradients for all parameters.
    /// Useful for optimizer updates.
    /// </summary>let getParameterGrads (nodes: Node list) : (Node * float[]) list =
        nodes
        |> List.filter (fun node -> 
            match node.Op with
            | Parameter _ -> true
            | _ -> false)
        |> List.choose (fun node -
            match node.Grad with
            | Some grad -> Some (node, grad)
            | None -> None)