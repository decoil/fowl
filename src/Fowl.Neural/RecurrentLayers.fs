namespace Fowl.Neural

open System
open Fowl
open Fowl.Core.Types

/// <summary>
/// Recurrent Neural Network layers.
/// LSTM and GRU implementations.
/// </summary>
module RecurrentLayers =
    
    /// <summary>
    /// LSTM cell state at time t.
    /// </summary>
type LSTMState = {
        /// Hidden state h_t
        Hidden: float[]
        /// Cell state c_t
        Cell: float[]
    }
    
    /// <summary>
    /// LSTM layer configuration.
    /// </summary>
type LSTMLayer = {
        /// Input size
        InputSize: int
        /// Hidden size
        HiddenSize: int
        /// Number of layers (stacked LSTM)
        NumLayers: int
        /// Dropout probability between layers
        Dropout: float
        /// Bias terms
        UseBias: bool
        /// Weights for input-to-hidden (each layer)
        WeightsInput: float[,][]
        /// Weights for hidden-to-hidden (each layer)
        WeightsHidden: float[,][]
        /// Biases (each layer, 4 gates)
        Biases: float[][]
    }
    
    /// <summary>
    /// GRU layer configuration.
    /// </summary>
type GRULayer = {
        /// Input size
        InputSize: int
        /// Hidden size  
        HiddenSize: int
        /// Number of layers
        NumLayers: int
        /// Dropout probability
        Dropout: float
        /// Weights for update gate
        WeightsUpdate: float[,][]
        /// Weights for reset gate
        WeightsReset: float[,][]
        /// Weights for new gate
        WeightsNew: float[,][]
        /// Biases
        Biases: float[][]
    }
    
    /// <summary>
    /// Create LSTM layer with Xavier initialization.
    /// </summary>
    let createLSTM (inputSize: int) (hiddenSize: int) 
                   ?(numLayers: int) ?(dropout: float) ?(seed: int) : FowlResult<LSTMLayer> =
        result {
            let numLayers = defaultArg numLayers 1
            let dropout = defaultArg dropout 0.0
            let seed = defaultArg seed 42
            let rng = Random(seed)
            
            if inputSize <= 0 then
                return! Error.invalidArgument "LSTM inputSize must be positive"
            if hiddenSize <= 0 then
                return! Error.invalidArgument "LSTM hiddenSize must be positive"
            if numLayers <= 0 then
                return! Error.invalidArgument "LSTM numLayers must be positive"
            
            // Xavier initialization std
            let initStd = sqrt (2.0 / float (inputSize + hiddenSize))
            
            // Initialize weights for each layer
            let weightsInput = Array.init numLayers (fun layer -
                let inSize = if layer = 0 then inputSize else hiddenSize
                Array2D.init hiddenSize (inSize * 4) (fun _ _ -
                    (rng.NextDouble() * 2.0 - 1.0) * initStd))
            
            let weightsHidden = Array.init numLayers (fun _ -
                Array2D.init hiddenSize (hiddenSize * 4) (fun _ _ -
                    (rng.NextDouble() * 2.0 - 1.0) * initStd))
            
            // Initialize biases - set forget gate bias to 1.0 (common practice)
            let biases = Array.init numLayers (fun _ -
                let b = Array.zeroCreate (hiddenSize * 4)
                for i = hiddenSize * 2 to hiddenSize * 3 - 1 do
                    b.[i] <- 1.0  // Forget gate bias
                b)
            
            return {
                InputSize = inputSize
                HiddenSize = hiddenSize
                NumLayers = numLayers
                Dropout = dropout
                UseBias = true
                WeightsInput = weightsInput
                WeightsHidden = weightsHidden
                Biases = biases
            }
        }
    
    /// <summary>
    /// Sigmoid activation function.
    /// </summary>
    let private sigmoid (x: float) : float =
        1.0 / (1.0 + exp (-x))
    
    /// <summary>
    /// Hyperbolic tangent.
    /// </summary>
    let private tanhF (x: float) : float =
        tanh x
    
    /// <summary>
    /// Single LSTM step forward.
    /// Returns new hidden and cell states.
    /// </summary>
    let lstmStep (layer: LSTMLayer) (layerIdx: int)
                 (input: float[]) (prevState: LSTMState) : LSTMState =
        
        let hPrev = prevState.Hidden
        let cPrev = prevState.Cell
        
        // Compute gates
        // x = [input; hPrev] concatenated
        let x = Array.append input hPrev
        
        // Linear transformation
        let wInput = layer.WeightsInput.[layerIdx]
        let wHidden = layer.WeightsHidden.[layerIdx]
        let bias = layer.Biases.[layerIdx]
        
        // Compute pre-activations
        let mutable gates = Array.copy bias
        
        // Input contribution
        for i = 0 to layer.HiddenSize * 4 - 1 do
            for j = 0 to input.Length - 1 do
                gates.[i] <- gates.[i] + wInput.[i, j] * input.[j]
        
        // Hidden contribution  
        for i = 0 to layer.HiddenSize * 4 - 1 do
            for j = 0 to layer.HiddenSize - 1 do
                gates.[i] <- gates.[i] + wHidden.[i, j] * hPrev.[j]
        
        // Apply activations to gates
        // gates layout: [input_gate, forget_gate, cell_gate, output_gate]
        let hiddenSize = layer.HiddenSize
        let iGate = Array.init hiddenSize (fun j -> sigmoid gates.[j])
        let fGate = Array.init hiddenSize (fun j -> sigmoid gates.[hiddenSize + j])
        let gGate = Array.init hiddenSize (fun j -> tanhF gates.[2*hiddenSize + j])
        let oGate = Array.init hiddenSize (fun j -> sigmoid gates.[3*hiddenSize + j])
        
        // Cell state: c_t = f * c_{t-1} + i * g
        let cNew = Array.init hiddenSize (fun j -
            fGate.[j] * cPrev.[j] + iGate.[j] * gGate.[j])
        
        // Hidden state: h_t = o * tanh(c_t)
        let hNew = Array.init hiddenSize (fun j -
            oGate.[j] * tanhF cNew.[j])
        
        { Hidden = hNew; Cell = cNew }
    
    /// <summary>
    /// LSTM forward pass for a sequence.
    /// Input: [seq_len, batch, input_size]
    /// Output: [seq_len, batch, hidden_size]
    /// </summary>
    let lstmForward (layer: LSTMLayer) (input: float[][][]) : float[][][] =
        let seqLen = input.Length
        let batchSize = input.[0].Length
        
        // Initialize states
        let mutable states = 
            Array.init batchSize (fun _ -
                { 
                    Hidden = Array.zeroCreate layer.HiddenSize
                    Cell = Array.zeroCreate layer.HiddenSize
                })
        
        let outputs = Array.init seqLen (fun _ -
            Array.zeroCreate batchSize)
        
        // Process sequence
        for t = 0 to seqLen - 1 do
            for b = 0 to batchSize - 1 do
                let inp = input.[t].[b]
                let newState = lstmStep layer 0 inp states.[b]
                states.[b] <- newState
                outputs.[t].[b] <- newState.Hidden
        
        outputs
    
    /// <summary>
    /// Create GRU layer with Xavier initialization.
    /// </summary>
    let createGRU (inputSize: int) (hiddenSize: int)
                  ?(numLayers: int) ?(dropout: float) ?(seed: int) : FowlResult<GRULayer> =
        result {
            let numLayers = defaultArg numLayers 1
            let dropout = defaultArg dropout 0.0
            let seed = defaultArg seed 42
            let rng = Random(seed)
            
            if inputSize <= 0 then
                return! Error.invalidArgument "GRU inputSize must be positive"
            if hiddenSize <= 0 then
                return! Error.invalidArgument "GRU hiddenSize must be positive"
            
            let initStd = sqrt (2.0 / float (inputSize + hiddenSize))
            
            // GRU has 3 gates: update (z), reset (r), new (n)
            let weightsUpdate = Array.init numLayers (fun layer -
                let inSize = if layer = 0 then inputSize else hiddenSize
                Array2D.init hiddenSize (inSize + hiddenSize) (fun _ _ -
                    (rng.NextDouble() * 2.0 - 1.0) * initStd))
            
            let weightsReset = Array.init numLayers (fun layer -
                let inSize = if layer = 0 then inputSize else hiddenSize
                Array2D.init hiddenSize (inSize + hiddenSize) (fun _ _ -
                    (rng.NextDouble() * 2.0 - 1.0) * initStd))
            
            let weightsNew = Array.init numLayers (fun layer -
                let inSize = if layer = 0 then inputSize else hiddenSize
                Array2D.init hiddenSize (inSize + hiddenSize) (fun _ _ -
                    (rng.NextDouble() * 2.0 - 1.0) * initStd))
            
            let biases = Array.init numLayers (fun _ -
                Array.zeroCreate (hiddenSize * 3))
            
            return {
                InputSize = inputSize
                HiddenSize = hiddenSize
                NumLayers = numLayers
                Dropout = dropout
                WeightsUpdate = weightsUpdate
                WeightsReset = weightsReset
                WeightsNew = weightsNew
                Biases = biases
            }
        }
    
    /// <summary>
    /// Single GRU step forward.
    /// </summary>
    let gruStep (layer: GRULayer) (layerIdx: int)
                (input: float[]) (hPrev: float[]) : float[] =
        
        let hiddenSize = layer.HiddenSize
        let wUpdate = layer.WeightsUpdate.[layerIdx]
        let wReset = layer.WeightsReset.[layerIdx]
        let wNew = layer.WeightsNew.[layerIdx]
        let bias = layer.Biases.[layerIdx]
        
        // Concatenate input and previous hidden
        let x = Array.append input hPrev
        
        // Update gate: z = sigmoid(W_z * x + b_z)
        let zGate = Array.init hiddenSize (fun i -
            let mutable sum = bias.[i]
            for j = 0 to x.Length - 1 do
                sum <- sum + wUpdate.[i, j] * x.[j]
            sigmoid sum)
        
        // Reset gate: r = sigmoid(W_r * x + b_r)
        let rGate = Array.init hiddenSize (fun i -
            let mutable sum = bias.[hiddenSize + i]
            for j = 0 to x.Length - 1 do
                sum <- sum + wReset.[i, j] * x.[j]
            sigmoid sum)
        
        // New gate: n = tanh(W_n * [input; r * hPrev] + b_n)
        let xNew = Array.append input (Array.map2 (*) rGate hPrev)
        let nGate = Array.init hiddenSize (fun i -
            let mutable sum = bias.[2*hiddenSize + i]
            for j = 0 to xNew.Length - 1 do
                sum <- sum + wNew.[i, j] * xNew.[j]
            tanhF sum)
        
        // Hidden state: h = (1-z) * n + z * hPrev
        Array.init hiddenSize (fun i -
            (1.0 - zGate.[i]) * nGate.[i] + zGate.[i] * hPrev.[i])
    
    /// <summary>
    /// GRU forward pass for a sequence.
    /// </summary>
    let gruForward (layer: GRULayer) (input: float[][][]) : float[][][] =
        let seqLen = input.Length
        let batchSize = input.[0].Length
        
        let mutable hidden = 
            Array.init batchSize (fun _ -
                Array.zeroCreate layer.HiddenSize)
        
        let outputs = Array.init seqLen (fun _ -
            Array.zeroCreate batchSize)
        
        for t = 0 to seqLen - 1 do
            for b = 0 to batchSize - 1 do
                let newHidden = gruStep layer 0 input.[t].[b] hidden.[b]
                hidden.[b] <- newHidden
                outputs.[t].[b] <- newHidden
        
        outputs