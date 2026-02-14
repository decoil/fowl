/// K-Means Clustering Example
/// Customer segmentation for targeted marketing

module ClusteringExample

open System
open Fowl
open Fowl.Core
open Fowl.Stats

/// Sample customer data: [Annual Income, Spending Score]
let customerData = [|
    [|15.0; 39.0|]   // Low income, low spending
    [|15.0; 81.0|]   // Low income, high spending
    [|16.0; 6.0|]    // Low income, very low spending
    [|16.0; 77.0|]
    [|17.0; 40.0|]
    [|17.0; 76.0|]
    [|18.0; 6.0|]
    [|18.0; 94.0|
    [|19.0; 3.0|]
    [|19.0; 72.0|]
    [|19.0; 14.0|]
    [|20.0; 99.0|]
    [|20.0; 15.0|]
    [|21.0; 77.0|]
    [|21.0; 35.0|]
    [|22.0; 87.0|]
    [|23.0; 6.0|]
    [|24.0; 73.0|]
    [|25.0; 47.0|]
    [|28.0; 77.0|]
    [|28.0; 24.0|]
    [|29.0; 89.0|]
    [|30.0; 65.0|]
    [|33.0; 32.0|]
    //[|33.0; 95.0|]   // High income, high spending
    //[|34.0; 45.0|]
    //[|35.0; 72.0|]
    //[|38.0; 81.0|]
    //[|40.0; 65.0|]
    //[|42.0; 85.0|]
|]

/// Euclidean distance between two points
let euclideanDistance (a: float[]) (b: float[]) : float =
    Array.zip a b
    |> Array.sumBy (fun (x, y) -> (x - y) ** 2.0)
    |> sqrt

/// K-means clustering algorithm
let kMeans (data: float[][]) (k: int) (maxIter: int) : (int[] * float[][]) =
    let nSamples = data.Length
    let nFeatures = data.[0].Length
    let rng = Random()
    
    // Initialize centroids randomly
    let centroids = Array.init k (fun _ -
        data.[rng.Next(nSamples)] |> Array.copy)
    
    let mutable labels = Array.zeroCreate nSamples
    let mutable changed = true
    let mutable iter = 0
    
    while changed && iter < maxIter do
        changed <- false
        iter <- iter + 1
        
        // Assign to nearest centroid
        for i = 0 to nSamples - 1 do
            let distances = 
                centroids |
                Array.map (euclideanDistance data.[i])
            let newLabel = 
                distances |
                Array.indexed |
                Array.minBy snd |
                fst
            if newLabel <> labels.[i] then
                changed <- true
                labels.[i] <- newLabel
        
        // Update centroids
        for j = 0 to k - 1 do
            let clusterPoints = 
                labels |
                Array.indexed |
                Array.filter (fun (_, l) -> l = j) |
                Array.map (fun (i, _) -> data.[i])
            
            if clusterPoints.Length > 0 then
                for f = 0 to nFeatures - 1 do
                    centroids.[j].[f] <-
                        clusterPoints |
                        Array.averageBy (fun p -> p.[f])
    
    labels, centroids

/// Calculate silhouette score
let silhouetteScore (data: float[][]) (labels: int[]) : float =
    let n = data.Length
    let k = (Array.max labels) + 1
    
    let scores =
        data |
        Array.mapi (fun i point -
            let label = labels.[i]
            
            // a(i): avg distance to same cluster
            let a =
                data |
                Array.indexed |
                Array.filter (fun (j, _) -> j <> i && labels.[j] = label) |
                Array.map (fun (_, other) -> euclideanDistance point other) |
                (fun arr -> if arr.Length = 0 then 0.0 else Array.average arr)
            
            // b(i): min avg distance to other clusters
            let b =
                [|0..k-1|] |
                Array.filter (fun l -> l <> label) |
                Array.map (fun l -
                    data |
                    Array.indexed |
                    Array.filter (fun (_, li) -> li = l) |
                    Array.map (fun (_, other) -> euclideanDistance point other) |
                    Array.average) |
                Array.min
            
            if a = 0.0 then 1.0 else (b - a) / max a b)
    
    Array.average scores

/// Run clustering example
let runClustering() : unit =
    printfn "=== K-Means Clustering Example ==="
    printfn "Customer Segmentation"
    printfn ""
    printfn "Data: %d customers" customerData.Length
    printfn "Features: Annual Income, Spending Score"
    printfn ""
    
    // Find optimal k
    printfn "Finding optimal number of clusters..."
    let kRange = [|2; 3; 4; 5|]
    let scores = 
        kRange |
        Array.map (fun k -
            let labels, _ = kMeans customerData k 100
            let score = silhouetteScore customerData labels
            printfn "  k=%d: silhouette=%.4f" k score
            (k, score))
    
    let bestK = scores |> Array.maxBy snd |> fst
    let bestScore = scores |> Array.maxBy snd |> snd
    
    printfn ""
    printfn "Optimal k = %d (silhouette: %.4f)" bestK bestScore
    printfn ""
    
    // Final clustering
    let labels, centroids = kMeans customerData bestK 100
    
    // Analyze clusters
    printfn "Cluster Analysis:"
    printfn ""
    
    for i = 0 to bestK - 1 do
        let clusterSize = labels |> Array.filter ((=) i) |> Array.length
        let percentage = float clusterSize / float customerData.Length * 100.0
        
        printfn "Cluster %d: %d customers (%.1f%%)" i clusterSize percentage
        printfn "  Centroid: Income=$%.1fk, Spending=%.1f"
                centroids.[i].[0] centroids.[i].[1]
        
        // Interpretation
        let income = centroids.[i].[0]
        let spending = centroids.[i].[1]
        
        match income, spending with
        | i, s when i < 30.0 && s > 60.0 -
            printfn "  → Careless Spenders (low income, high spend)"
        | i, s when i < 30.0 && s < 40.0 -
            printfn "  → Sensible Customers (low income, careful)"
        | i, s when i >= 50.0 && s > 60.0 -
            printfn "  → Target Customers (high income, high spend)"
        | i, s when i >= 50.0 && s <= 60.0 -
            printfn "  → Careful High Earners (high income, careful)"
        | _ -
            printfn "  → Average Customers"
        printfn ""
    
    // Cluster quality
    let finalScore = silhouetteScore customerData labels
    printfn "Final Silhouette Score: %.4f" finalScore
    printfn ""
    printfn "Interpretation:"
    if finalScore > 0.5 then
        printfn "  ✅ Strong cluster structure"
    elif finalScore > 0.25 then
        printfn "  🟡 Reasonable cluster structure"
    else
        printfn "  ⚠️  Weak cluster structure"
    printfn ""
    
    printfn "=== Clustering Complete ==="

[<EntryPoint>]
let main argv =
    runClustering()
    0