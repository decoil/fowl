// SentimentAnalysis.fsx
// NLP sentiment classifier using word embeddings

#r "nuget: Fowl"

open System
open Fowl
open Fowl.Core.Types
open Fowl.Neural
open Fowl.Neural.Graph
open Fowl.Neural.Layers
open Fowl.Neural.Forward
open Fowl.Neural.Backward
open Fowl.Neural.Loss
open Fowl.Neural.Optimizer
open Fowl.Stats

printfn "=== Sentiment Analysis with RNN ==="

// Simplified word embedding (in practice, use pre-trained like GloVe/Word2Vec)
let vocabularySize = 1000
let embeddingDim = 50
let hiddenSize = 64

// Generate random embedding matrix
let rng = Random(42)
let embeddings =
    Array2D.init vocabularySize embeddingDim (fun _ _ -
        (rng.NextDouble() - 0.5) * 0.1)

/// Look up word embedding
let embedWord (wordId: int) : float[] =
    Array.init embeddingDim (fun i -> embeddings.[wordId, i])

/// Sentiment analysis model
let buildSentimentModel () =
    result {
        // LSTM for sequence processing
        let! lstm = RecurrentLayers.LSTM.create embeddingDim hiddenSize (Some 42)
        
        // Output layer for binary classification
        let! output = Layers.dense hiddenSize 1 (Some Sigmoid) (Some 43)
        
        let forward (sequence: int[]) =
            result {
                // Embed each word
                let embedded =
                    sequence
                    |> Array.map (fun wordId -
                        embedWord wordId)
                
                // Process with LSTM
                let input = Graph.input "sequence" [|sequence.Length; embeddingDim|]
                
                // Simplified: average embeddings then pass to dense
                // (Proper implementation would use full LSTM sequence processing)
                let! lstmOut = RecurrentLayers.LSTM.forward lstm input
                
                // Take last output for classification
                let! sentiment = Layers.forwardDense output lstmOut
                
                return sentiment
            }
        
        return (lstm, output, forward)
    }

/// Generate synthetic review data
type Review = {
    Text: int[]  // Word IDs
    Sentiment: float  // 0 = negative, 1 = positive
}

let generateSyntheticReviews (numSamples: int) : Review[] =
    [|
        for i in 1..numSamples do
            // Random sequence length
            let length = rng.Next(10, 50)
            
            // Generate word IDs
            let text = Array.init length (fun _ -> rng.Next(vocabularySize))
            
            // Random sentiment (biased toward positive for longer reviews with certain words)
            let sentiment = if rng.NextDouble() > 0.5 then 1.0 else 0.0
            
            { Text = text; Sentiment = sentiment }
    |]

/// Training function
let trainSentimentModel (model) (reviews: Review[]) (epochs: int) =
    result {
        let (lstm, output, forward) = model
        
        // Create graph
        let input = input "text" [||]
        let target = input "sentiment" [|1|]
        
        let! prediction = forward [||]  // Placeholder
        let! loss = Loss.binaryCrossEntropy prediction target
        
        let params = 
            Layers.getParameters lstm @
            Layers.getParameters output
        
        let optimizer = Optimizer.adam 0.001 0.9 0.999 1e-8
        
        // Training loop
        for epoch = 1 to epochs do
            let mutable totalLoss = 0.0
            let mutable correct = 0
            
            for review in reviews do
                // Skip very long reviews for simplicity
                if review.Text.Length < 30 then
                    params |> List.iter (fun (p, _) -> p.Grad.Value <- None)
                    
                    let! pred = forward review.Text.[..29]  // First 30 words
                    
                    let inputs = Map ["sentiment", [|review.Sentiment|]]
                    let! lossNode = Loss.binaryCrossEntropy pred (input "y" [|1|])
                    
                    do! Forward.run [lossNode]
                    do! Backward.run [lossNode]
                    
                    Optimizer.updateAdam optimizer params
                    
                    totalLoss <- totalLoss + (lossNode.Value.Value |> Option.defaultValue 0.0)
                    
                    // Accuracy
                    let predVal = pred.Value.Value |> Option.map (fun v -> if v.[0] > 0.5 then 1.0 else 0.0) |> Option.defaultValue 0.0
                    if abs(predVal - review.Sentiment) < 0.5 then
                        correct <- correct + 1
            
            let avgLoss = totalLoss / float reviews.Length
            let accuracy = float correct / float reviews.Length * 100.0
            
            if epoch % 5 = 0 then
                printfn "Epoch %d/%d: Loss = %.4f, Accuracy = %.1f%%" epoch epochs avgLoss accuracy
        
        return model
    }

/// Main execution
let runExample () =
    result {
        printfn "\nGenerating synthetic review data..."
        let trainReviews = generateSyntheticReviews 500
        let testReviews = generateSyntheticReviews 100
        
        printfn "Training samples: %d" trainReviews.Length
        printfn "Test samples: %d" testReviews.Length
        
        printfn "\nBuilding sentiment model..."
        let! model = buildSentimentModel ()
        
        printfn "\nTraining..."
        let! trained = trainSentimentModel model trainReviews 20
        
        printfn "\nExample prediction:"
        let exampleReview = trainReviews.[0]
        printfn "Review length: %d words" exampleReview.Text.Length
        printfn "True sentiment: %.0f" exampleReview.Sentiment
        
        printfn "\n=== Sentiment Analysis Complete ==="
        return trained
    }

// Run
match runExample () with
| Ok _ -> ()
| Error e -> printfn "Error: %A" e
