namespace Fowl.Neural

open Fowl
open Fowl.Core.Types

/// <summary>Represents the shape of a tensor in the computation graph.</summary>type Shape = int[]

/// <summary>Activation functions supported by the neural network module.</summary>type ActivationFn =
    | ReLU
    | Sigmoid
    | Tanh
    | Softmax
    | LeakyReLU of float
    | ELU of float
    | Identity

/// <summary>Loss functions for training.
/// Each loss function stores both the forward computation and its gradient function.</summary>type LossFn =
    | MSE              // Mean squared error: (pred - target)²
    | CrossEntropy     // Cross-entropy for classification
    | BinaryCrossEntropy // Binary cross-entropy
    | NLL              // Negative log-likelihood
    | Custom of (float[] -> float[] -> float) * (float[] -> float[] -> float[])

/// <summary>Operations that can be performed in the computation graph.
/// Each operation knows how to compute its output and gradients.
/// </summary>type Operation =
    | Input of string                    // Input node with name
    | Const of float                     // Constant scalar value
    | ConstArray of float[] * Shape      // Constant array with shape
    | Add                                // Element-wise addition
    | Sub                                // Element-wise subtraction
    | Mul                                // Element-wise multiplication
    | Div                                // Element-wise division
    | MatMul                             // Matrix multiplication
    | Transpose of int list              // Transpose with axis permutation
    | Reshape of Shape                   // Reshape to new shape
    | Sum of int option                  // Sum all elements or along axis
    | Mean of int option                 // Mean all elements or along axis
    | Activation of ActivationFn         // Activation function
    | Loss of LossFn * Node option       // Loss function with optional target
    | Parameter of string                // Trainable parameter
    | Custom of string * (float[] -> float[] -> float[])  // Name * gradient function

/// <summary>A node in the computation graph.
/// Nodes are mutable to support lazy evaluation and gradient accumulation.
/// </summary>and Node = {
    /// Unique identifier for this node
    Id: int
    /// Shape of the tensor this node produces
    Shape: Shape
    /// Operation that computes this node's value
    Op: Operation
    /// Parent nodes (inputs to this operation)
    Parents: Node list
    /// Cached output value (lazy evaluation)
    mutable Value: float[] option
    /// Accumulated gradient (for backprop)
    mutable Grad: float[] option
    /// Children nodes (nodes that use this as input)
    mutable Children: Node list
}

/// <summary>A computation graph containing all nodes.
/// </summary>type Graph = {
    Nodes: Map<int, Node>
    Inputs: Map<string, Node>
    Parameters: Node list
    mutable NextId: int
}

/// <summary>Module for creating and manipulating computation graphs.
/// </summary>module Graph =
    /// Global counter for generating unique node IDs
    let private idCounter = ref 0
    
    /// Get the next unique node ID
    let private nextId() = 
        let id = !idCounter
        idCounter := id + 1
        id
    
    /// Create an empty graph
    let empty = {
        Nodes = Map.empty
        Inputs = Map.empty
        Parameters = []
        NextId = 0
    }
    
    /// <summary>Create an input node.
    /// </summary>let input (name: string) (shape: Shape) : Node =
        { Id = nextId(); Shape = shape; Op = Input name
          Parents = []; Value = None; Grad = None; Children = [] }
    
    /// <summary>Create a constant node.
    /// </summary>let constant (value: float) : Node =
        { Id = nextId(); Shape = [||]; Op = Const value
          Parents = []; Value = Some [|value|]; Grad = None; Children = [] }
    
    /// <summary>Create a constant array node.
    /// </summary>let constantArray (data: float[]) (shape: Shape) : Node =
        { Id = nextId(); Shape = shape; Op = ConstArray(data, shape)
          Parents = []; Value = Some data; Grad = None; Children = [] }
    
    /// <summary>Create a trainable parameter node.
    /// </summary>let parameter (name: string) (shape: Shape) (init: float[]) : Node =
        { Id = nextId(); Shape = shape; Op = Parameter name
          Parents = []; Value = Some init; Grad = Some (Array.zeroCreate init.Length); Children = [] }
    
    /// <summary>Create an addition node.
    /// </summary>let add (a: Node) (b: Node) : Node =
        // Infer shape (broadcasting)
        let shape = 
            if a.Shape = b.Shape then a.Shape
            elif a.Shape = [||] then b.Shape
            elif b.Shape = [||] then a.Shape
            else a.Shape  // TODO: proper broadcasting rules
        
        let node = { Id = nextId(); Shape = shape; Op = Add
                     Parents = [a; b]; Value = None; Grad = None; Children = [] }
        a.Children <- node :: a.Children
        b.Children <- node :: b.Children
        node
    
    /// <summary>Create a subtraction node.
    /// </summary>let sub (a: Node) (b: Node) : Node =
        let node = { Id = nextId(); Shape = a.Shape; Op = Sub
                     Parents = [a; b]; Value = None; Grad = None; Children = [] }
        a.Children <- node :: a.Children
        b.Children <- node :: b.Children
        node
    
    /// <summary>Create a multiplication node.
    /// </summary>let mul (a: Node) (b: Node) : Node =
        let node = { Id = nextId(); Shape = a.Shape; Op = Mul
                     Parents = [a; b]; Value = None; Grad = None; Children = [] }
        a.Children <- node :: a.Children
        b.Children <- node :: b.Children
        node
    
    /// <summary>Create a matrix multiplication node.
    /// </summary>let matmul (a: Node) (b: Node) : FowlResult<Node> =
        if a.Shape.Length < 2 || b.Shape.Length < 2 then
            Error.invalidArgument "matmul requires 2D tensors"
        elif Array.last a.Shape <>> Array.head b.Shape then
            Error.invalidArgument (sprintf "Shape mismatch: %A @ %A" a.Shape b.Shape)
        else
            let outShape = 
                Array.append (Array.sub a.Shape 0 (a.Shape.Length - 1))
                             (Array.sub b.Shape 1 (b.Shape.Length - 1))
            
            let node = { Id = nextId(); Shape = outShape; Op = MatMul
                         Parents = [a; b]; Value = None; Grad = None; Children = [] }
            a.Children <- node :: a.Children
            b.Children <- node :: b.Children
            Ok node
    
    /// <summary>Apply an activation function.
    /// </summary>let activate (fn: ActivationFn) (x: Node) : Node =
        let node = { Id = nextId(); Shape = x.Shape; Op = Activation fn
                     Parents = [x]; Value = None; Grad = None; Children = [] }
        x.Children <- node :: x.Children
        node
    
    /// <summary>Compute the sum of all elements.
    /// </summary>let sum (x: Node) : Node =
        let node = { Id = nextId(); Shape = [||]; Op = Sum None
                     Parents = [x]; Value = None; Grad = None; Children = [] }
        x.Children <- node :: x.Children
        node
    
    /// <summary>Compute the mean of all elements.
    /// </summary>let mean (x: Node) : Node =
        let node = { Id = nextId(); Shape = [||]; Op = Mean None
                     Parents = [x]; Value = None; Grad = None; Children = [] }
        x.Children <- node :: x.Children
        node
    
    /// <summary>Topological sort of nodes (Kahn's algorithm).
    /// Returns nodes in order they can be computed.
    /// </summary>let topologicalSort (nodes: Node list) : Node list =
        let mutable visited = Set.empty
        let mutable result = []
        
        let rec visit (node: Node) =
            if not (visited.Contains node.Id) then
                visited <- visited.Add node.Id
                for parent in node.Parents do
                    visit parent
                result <- node :: result
        
        for node in nodes do
            visit node
        
        List.rev result