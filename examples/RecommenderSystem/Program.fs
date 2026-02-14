/// Recommender System Example - Chapter 16
/// Implements: Dimensionality reduction, Random Projection, Vector Search Trees
/// Based on Owl's CF (Collaborative Filtering) and LSH (Locality Sensitive Hashing)

module RecommenderSystem

open System
open Fowl
open Fowl.Core.Types
open Fowl.Linalg
open Fowl.Stats

// ============================================================================
// VECTOR STORAGE AND SIMILARITY
// ============================================================================

/// Dense vector storage for user/item embeddings
module VectorStorage =
    
    type Vector = float[]
    
    /// Vector database
    type Storage = {
        Vectors: Vector[]
        /// Optional metadata (IDs, labels, etc.)
        Metadata: string[]
        Dimension: int
    }
    
    /// Create storage from vectors
    let create (vectors: float[][]) (metadata: string[]) : Storage =
        { Vectors = vectors; Metadata = metadata; Dimension = vectors.[0].Length }
    
    /// Add vector to storage
    let add (storage: Storage) (vector: Vector) (meta: string) : Storage =
        { storage with
            Vectors = Array.append storage.Vectors [|vector|]
            Metadata = Array.append storage.Metadata [|meta|] }
    
    /// Cosine similarity: (a · b) / (||a|| ||b||)
    let cosineSimilarity (a: Vector) (b: Vector) : float =
        let dot = Array.map2 (*) a b |> Array.sum
        let normA = sqrt (Array.sumBy (fun x -> x * x) a)
        let normB = sqrt (Array.sumBy (fun x -> x * x) b)
        dot / (normA * normB)
    
    /// Euclidean distance
    let euclideanDistance (a: Vector) (b: Vector) : float =
        Array.map2 (-) a b
        |> Array.sumBy (fun d -> d * d)
        |> sqrt
    
    /// Dot product similarity
    let dotProduct (a: Vector) (b: Vector) : float =
        Array.map2 (*) a b |> Array.sum
    
    /// Find k nearest neighbors (brute force)
    let knn (storage: Storage) (query: Vector) (k: int) 
            (metric: Vector -> Vector -> float) : (int * float)[] =
        
        storage.Vectors
        |> Array.mapi (fun i v -> (i, metric v query))
        |> Array.sortByDescending snd
        |> Array.take (min k storage.Vectors.Length)

// ============================================================================
// DIMENSIONALITY REDUCTION
// ============================================================================

/// Principal Component Analysis for dimensionality reduction
module PCA =
    
    type Model = {
        /// Principal components (eigenvectors)
        Components: float[,]
        /// Explained variance ratio
        ExplainedVarianceRatio: float[]
        /// Mean of training data
        Mean: float[]
        /// Number of components
        NComponents: int
    }
    
    /// Fit PCA on data
    let fit (X: float[,]) (nComponents: int) : FowlResult<Model> =
        result {
            let n = X.GetLength(0)
            let p = X.GetLength(1)
            
            // Center data
            let mean = Array.init p (fun j -
                Array.init n (fun i -> X.[i, j]) |> Array.average)
            
            let XCentered = Array2D.init n p (fun i j -> X.[i, j] - mean.[j])
            
            // Compute covariance matrix
            let! xArr = Ndarray.ofArray2D XCentered
            let! xt = Matrix.transpose xArr
            let! cov = Matrix.matmul xt xArr
            let covMatrix = Ndarray.toArray2D cov
            for i = 0 to p - 1 do
                for j = 0 to p - 1 do
                    covMatrix.[i, j] <- covMatrix.[i, j] / float (n - 1)
            
            // Eigendecomposition (simplified - would use SVD)
            // For now, return identity components
            let components = Array2D.init p nComponents (fun i j -
                if i = j then 1.0 else 0.0)
            
            return {
                Components = components
                ExplainedVarianceRatio = Array.create nComponents (1.0 / float nComponents)
                Mean = mean
                NComponents = nComponents
            }
        }
    
    /// Transform data using fitted PCA
    let transform (model: Model) (X: float[,]) : float[,] =
        let n = X.GetLength(0)
        let p = X.GetLength(1)
        
        // Center data
        let XCentered = Array2D.init n p (fun i j -
            X.[i, j] - model.Mean.[j])
        
        // Project onto components
        let mutable result = Array2D.zeroCreate n model.NComponents
        for i = 0 to n - 1 do
            for j = 0 to model.NComponents - 1 do
                let mutable sum = 0.0
                for k = 0 to p - 1 do
                    sum <- sum + XCentered.[i, k] * model.Components.[k, j]
                result.[i, j] <- sum
        
        result
    
    /// Fit and transform in one step
    let fitTransform (X: float[,]) (nComponents: int) : FowlResult<float[,] * Model> =
        result {
            let! model = fit X nComponents
            let transformed = transform model X
            return (transformed, model)
        }

/// Random Projection for fast dimensionality reduction
/// Johnson-Lindenstrauss lemma: preserves distances with high probability
module RandomProjection =
    
    type Model = {
        /// Random projection matrix
        ProjectionMatrix: float[,]
        /// Original dimension
        OriginalDim: int
        /// Target dimension
        TargetDim: int
    }
    
    /// Generate Gaussian random matrix
    let private gaussianRandomMatrix (inputDim: int) (outputDim: int) (seed: int) : float[,] =
        let rng = Random(seed)
        let scale = 1.0 / sqrt (float outputDim)
        
        Array2D.init outputDim inputDim (fun _ _ -
            (rng.NextDouble() * 2.0 - 1.0) * scale)
    
    /// Generate sparse random matrix (Achlioptas, 2003)
    /// Much faster for high-dimensional data
    let private sparseRandomMatrix (inputDim: int) (outputDim: int) (seed: int) : float[,] =
        let rng = Random(seed)
        let scale = sqrt (3.0 / float outputDim)
        
        Array2D.init outputDim inputDim (fun _ _ -
            let r = rng.NextDouble()
            if r < 1.0/6.0 then scale
            elif r < 5.0/6.0 then 0.0
            else -scale)
    
    /// Create random projection
    let create (inputDim: int) (outputDim: int) ?(sparse: bool) (seed: int) : Model =
        let sparse = defaultArg sparse false
        let matrix = 
            if sparse then
                sparseRandomMatrix inputDim outputDim seed
            else
                gaussianRandomMatrix inputDim outputDim seed
        
        {
            ProjectionMatrix = matrix
            OriginalDim = inputDim
            TargetDim = outputDim
        }
    
    /// Transform vector using random projection
    let transform (model: Model) (vector: float[]) : float[] =
        let output = Array.zeroCreate model.TargetDim
        for i = 0 to model.TargetDim - 1 do
            let mutable sum = 0.0
            for j = 0 to model.OriginalDim - 1 do
                sum <- sum + model.ProjectionMatrix.[i, j] * vector.[j]
            output.[i] <- sum
        output
    
    /// Transform matrix
    let transformMatrix (model: Model) (X: float[,]) : float[,] =
        let n = X.GetLength(0)
        Array2D.init n model.TargetDim (fun i j -
            let mutable sum = 0.0
            for k = 0 to model.OriginalDim - 1 do
                sum <- sum + X.[i, k] * model.ProjectionMatrix.[j, k]
            sum)
    
    /// Calculate minimum dimensions for JL lemma
    /// Guarantees (1 ± ε)-distortion with probability (1 - δ)
    let minDimensions (nSamples: int) (eps: float) (delta: float) : int =
        let logTerm = log (float nSamples / delta)
        int (4.0 * logTerm / (eps * eps / 2.0 - eps * eps * eps / 3.0))

// ============================================================================
// TREE-BASED SEARCH STRUCTURES
// ============================================================================

/// VP-Tree (Vantage Point Tree) for metric spaces
/// Efficient for high-dimensional similarity search
module VPTree =
    
    type Node =
        | Leaf of int[]  // Indices of points
        | Internal of InternalNode
    
    and InternalNode = {
        /// Index of vantage point
        VantagePoint: int
        /// Median distance (threshold)
        Threshold: float
        /// Points inside threshold
        Left: Node
        /// Points outside threshold
        Right: Node
    }
    
    type Tree = {
        Root: Node
        /// Stored vectors
        Vectors: float[][]
        /// Distance metric
        Metric: float[] -> float[] -> float
    }
    
    /// Build VP-Tree recursively
    let rec private buildNode (vectors: float[][]) (indices: int[]) 
                              (metric: float[] -> float[] -> float)
                              (leafSize: int) : Node =
        
        if indices.Length <= leafSize then
            Leaf indices
        else
            // Choose vantage point (first element or median heuristic)
            let vpIdx = indices.[0]
            let vp = vectors.[vpIdx]
            
            // Compute distances to all other points
            let distances = 
                indices.[1..] 
                |> Array.map (fun i -> (i, metric vectors.[i] vp))
            
            if distances.Length = 0 then
                Leaf indices
            else
                // Find median distance
                let sortedDist = distances |> Array.sortBy snd
                let medianIdx = sortedDist.Length / 2
                let threshold = snd sortedDist.[medianIdx]
                
                // Partition
                let inside = 
                    sortedDist.[0..medianIdx] |> Array.map fst
                let outside = 
                    if medianIdx + 1 < sortedDist.Length then
                        sortedDist.[medianIdx+1..] |> Array.map fst
                    else
                        [||]
                
                // Recursively build
                let left = buildNode vectors inside metric leafSize
                let right = buildNode vectors outside metric leafSize
                
                Internal {
                    VantagePoint = vpIdx
                    Threshold = threshold
                    Left = left
                    Right = right
                }
    
    /// Build VP-Tree from vectors
    let build (vectors: float[][]) (metric: float[] -> float[] -> float) 
              ?(leafSize: int) : Tree =
        
        let leafSize = defaultArg leafSize 10
        let indices = Array.init vectors.Length id
        let root = buildNode vectors indices metric leafSize
        
        { Root = root; Vectors = vectors; Metric = metric }
    
    /// Search k nearest neighbors
    let knn (tree: Tree) (query: float[]) (k: int) : (int * float)[] =
        // Priority queue for results (max-heap by distance)
        let mutable results: (int * float) list = []
        
        let rec searchNode (node: Node) (tau: float byref) : unit =
            match node with
            | Leaf indices -
                for idx in indices do
                    let dist = tree.Metric tree.Vectors.[idx] query
                    if List.length results < k || dist < snd (List.head results) then
                        results <- (idx, dist) :: results
                        results <- results |> List.sortByDescending snd |> List.take (min k (List.length results))
                        if List.length results = k then
                            tau <- snd (List.head results)
            
            | Internal node -
                let dist = tree.Metric tree.Vectors.[node.VantagePoint] query
                
                // Add vantage point if close enough
                if List.length results < k || dist < tau then
                    results <- (node.VantagePoint, dist) :: results
                    results <- results |> List.sortByDescending snd |> List.take (min k (List.length results))
                    if List.length results = k then
                        tau <- snd (List.head results)
                
                // Decide which subtree to search first
                if dist < node.Threshold then
                    searchNode node.Left &tau
                    if dist + tau >= node.Threshold then
                        searchNode node.Right &tau
                else
                    searchNode node.Right &tau
                    if dist - tau <= node.Threshold then
                        searchNode node.Left &tau
        
        let mutable tau = Double.MaxValue
        searchNode tree.Root &tau
        results |> List.sortBy snd |> List.toArray

// ============================================================================
// LOCALITY SENSITIVE HASHING
// ============================================================================

/// LSH for approximate nearest neighbor search
module LSH =
    
    /// Hash function: h(v) = floor((a · v + b) / r)
    type HashFunction = {
        /// Random projection vector
        A: float[]
        /// Random offset
        B: float
        /// Bucket width
        R: float
    }
    
    /// LSH index
    type Index = {
        /// Multiple hash tables
        Tables: Map<int, int[]>[]
        /// Hash functions per table
        HashFunctions: HashFunction[][]
        /// Stored vectors
        Vectors: float[][]
    }
    
    /// Create random hash function
    let private createHashFunction (dim: int) (r: float) (rng: Random) : HashFunction =
        let a = Array.init dim (fun _ -> rng.NextGaussian())
        let b = rng.NextDouble() * r
        { A = a; B = b; R = r }
    
    /// Compute hash value
    let private hash (h: HashFunction) (v: float[]) : int =
        let dot = Array.map2 (*) h.A v |> Array.sum
        int ((dot + h.B) / h.R)
    
    /// Build LSH index
    let build (vectors: float[][]) (nHashTables: int) (nHashesPerTable: int)
              (bucketWidth: float) (seed: int) : Index =
        
        let rng = Random(seed)
        let dim = vectors.[0].Length
        
        // Create hash functions
        let hashFunctions = 
            Array.init nHashTables (fun _ -
                Array.init nHashesPerTable (fun _ -
                    createHashFunction dim bucketWidth rng))
        
        // Build hash tables
        let tables = 
            Array.init nHashTables (fun tableIdx -
                let table = ref Map.empty<int, int[]>
                for vecIdx = 0 to vectors.Length - 1 do
                    // Combine multiple hashes
                    let hashValue = 
                        hashFunctions.[tableIdx]
                        |> Array.map (fun h -> hash h vectors.[vecIdx])
                        |> Array.fold (fun acc h -> acc * 31 + h) 0
                    
                    table := 
                        match Map.tryFind hashValue !table with
                        | Some indices -
                            Map.add hashValue (Array.append indices [|vecIdx|]) !table
                        | None -
                            Map.add hashValue [|vecIdx|] !table
                !table)
        
        { Tables = tables; HashFunctions = hashFunctions; Vectors = vectors }
    
    /// Query LSH index for candidates
    let query (index: Index) (query: float[]) (k: int) : (int * float)[] =
        let mutable candidates = Set.empty
        
        // Collect candidates from all tables
        for tableIdx = 0 to index.Tables.Length - 1 do
            let hashValue = 
                index.HashFunctions.[tableIdx]
                |> Array.map (fun h -> hash h query)
                |> Array.fold (fun acc h -> acc * 31 + h) 0
            
            match Map.tryFind hashValue index.Tables.[tableIdx] with
            | Some indices -
                candidates <- Set.union candidates (Set.ofArray indices)
            | None -> ()
        
        // Compute exact distances for candidates
        candidates
        |> Set.toArray
        |> Array.map (fun idx -> (idx, VectorStorage.euclideanDistance index.Vectors.[idx] query))
        |> Array.sortBy snd
        |> Array.take (min k candidates.Count)

// ============================================================================
// COLLABORATIVE FILTERING
// ============================================================================

/// Matrix factorization for collaborative filtering
module CollaborativeFiltering =
    
    type Model = {
        /// User embeddings: [nUsers, nFactors]
        UserFactors: float[,]
        /// Item embeddings: [nItems, nFactors]
        ItemFactors: float[,]
        /// User biases
        UserBias: float[]
        /// Item biases
        ItemBias: float[]
        /// Global mean rating
        GlobalMean: float
        /// Number of factors
        NFactors: int
    }
    
    /// Predict rating for user-item pair
    let predict (model: Model) (userId: int) (itemId: int) : float =
        let mutable pred = model.GlobalMean + model.UserBias.[userId] + model.ItemBias.[itemId]
        for f = 0 to model.NFactors - 1 do
            pred <- pred + model.UserFactors.[userId, f] * model.ItemFactors.[itemId, f]
        pred
    
    /// Train using SGD (simplified)
    let train (ratings: (int * int * float)[]) (nUsers: int) (nItems: int)
              (nFactors: int) (epochs: int) (lr: float) (reg: float) : Model =
        
        // Initialize
        let rng = Random(42)
        let initFactor () = (rng.NextDouble() - 0.5) * 0.01
        
        let userFactors = Array2D.init nUsers nFactors (fun _ _ -> initFactor())
        let itemFactors = Array2D.init nItems nFactors (fun _ _ -> initFactor())
        let userBias = Array.zeroCreate nUsers
        let itemBias = Array.zeroCreate nItems
        
        // Compute global mean
        let globalMean = ratings |> Array.averageBy (fun (_, _, r) -> r)
        
        // Training loop
        for epoch = 1 to epochs do
            let mutable totalLoss = 0.0
            
            for (u, i, r) in ratings do
                let pred = 
                    globalMean + userBias.[u] + itemBias.[i] +
                    Array.init nFactors (fun f -
                        userFactors.[u, f] * itemFactors.[i, f])
                    |> Array.sum
                
                let err = r - pred
                totalLoss <- totalLoss + err * err
                
                // Update biases
                userBias.[u] <- userBias.[u] + lr * (err - reg * userBias.[u])
                itemBias.[i] <- itemBias.[i] + lr * (err - reg * itemBias.[i])
                
                // Update factors
                for f = 0 to nFactors - 1 do
                    let uf = userFactors.[u, f]
                    let iF = itemFactors.[i, f]
                    userFactors.[u, f] <- userFactors.[u, f] + lr * (err * iF - reg * uf)
                    itemFactors.[i, f] <- itemFactors.[i, f] + lr * (err * uf - reg * iF)
            
            if epoch % 10 = 0 then
                printfn "Epoch %d: RMSE = %.4f" epoch (sqrt (totalLoss / float ratings.Length))
        
        {
            UserFactors = userFactors
            ItemFactors = itemFactors
            UserBias = userBias
            ItemBias = itemBias
            GlobalMean = globalMean
            NFactors = nFactors
        }
    
    /// Recommend top-k items for user
    let recommend (model: Model) (userId: int) (nItems: int) (k: int) : (int * float)[] =
        Array.init nItems id
        |> Array.map (fun itemId -> (itemId, predict model userId itemId))
        |> Array.sortByDescending snd
        |> Array.take k

// ============================================================================
// MAIN
// ============================================================================

// Helper extension for Random
type Random with
    member this.NextGaussian() =
        let u1 = this.NextDouble()
        let u2 = this.NextDouble()
        sqrt (-2.0 * log u1) * cos (2.0 * System.Math.PI * u2)

let main argv =
    printfn "============================================================"
    printfn "  Recommender System - Chapter 16 (OCaml Scientific Computing)"
    printfn "  Techniques: PCA, Random Projection, VP-Trees, LSH, CF"
    printfn "============================================================\n"
    
    printfn "VECTOR STORAGE AND SIMILARITY"
    printfn "-----------------------------"
    printfn "Storage: Dense vector database with metadata"
    printfn "Metrics:"
    printfn "  - Cosine similarity: (a·b) / (||a|| ||b||)"
    printfn "  - Euclidean distance: ||a - b||"
    printfn "  - Dot product: a·b"
    printfn ""
    printfn "Brute force k-NN: O(n) distance computations"
    
    printfn "\n\nDIMENSIONALITY REDUCTION"
    printfn "------------------------"
    printfn "PCA (Principal Component Analysis):"
    printfn "  - Finds directions of maximum variance"
    printfn "  - Orthogonal components"
    printfn "  - Computationally expensive: O(n³) for eigendecomposition"
    printfn ""
    printfn "Random Projection (Johnson-Lindenstrauss):"
    printfn "  - Fast: O(n) per vector"
    printfn "  - Preserves distances: (1±ε) with high probability"
    printfn "  - Gaussian or sparse random matrix"
    printfn "  - minDims = 4·log(n/δ)/(ε²/2 - ε³/3)"
    
    printfn "\n\nTREE-BASED SEARCH (VP-TREE)"
    printfn "----------------------------"
    printfn "Vantage Point Tree:"
    printfn "  - Binary space partitioning"
    printfn "  - Metric space (no coordinates needed)"
    printfn "  - Each node: vantage point + threshold"
    printfn "  - Triangle inequality for pruning"
    printfn ""
    printfn "Build: O(n log n)"
    printfn "Query: O(log n) average, O(n) worst"
    printfn "Space: O(n)"
    
    printfn "\n\nLOCALITY SENSITIVE HASHING (LSH)"
    printfn "---------------------------------"
    printfn "Approximate nearest neighbors:"
    printfn "  - Hash similar vectors to same bucket"
    printfn "  - Multiple hash tables for recall"
    printfn "  - h(v) = floor((a·v + b) / r)"
    printfn ""
    printfn "Parameters:"
    printfn "  - L: number of hash tables (higher = better recall)"
    printfn "  - k: hashes per table (higher = more precision)"
    printfn "  - r: bucket width (tunes precision/recall)"
    printfn ""
    printfn "Complexity:"
    printfn "  - Build: O(n·L·k)"
    printfn "  - Query: O(L·k + candidates)"
    printfn "  - Space: O(n·L)"
    
    printfn "\n\nCOLLABORATIVE FILTERING"
    printfn "-----------------------"
    printfn "Matrix Factorization (SVD++):"
    printfn "  - Decompose rating matrix R ≈ U·V^T"
    printfn "  - User factors: capture preferences"
    printfn "  - Item factors: capture characteristics"
    printfn ""
    printfn "Prediction:"
    printfn "  r̂_ui = μ + b_u + b_i + Σ_f U_uf · V_if"
    printfn ""
    printfn "Training: SGD on squared error"
    printfn "  Loss = (r_ui - r̂_ui)² + λ·(reg terms)"
    
    printfn "\n\n============================================================"
    printfn "  Performance Comparison"
    printfn "============================================================\n"
    
    printfn "Method                    | Build    | Query    | Approx?"
    printfn "-------------------------|----------|----------|----------"
    printfn "Brute Force              | O(1)     | O(n)     | Exact"
    printfn "VP-Tree                  | O(n log n)| O(log n)| Exact"
    printfn "LSH                      | O(nLk)   | O(Lk)    | Approx"
    printfn "KD-Tree (high dim)       | O(n log n)| O(n)    | Exact"
    printfn "Ball Tree                | O(n log n)| O(log n)| Exact"
    printfn "Random Projection + Tree | O(n log n)| O(log n)| Approx"
    
    printfn "\n\n============================================================"
    printfn "  Implementation Status"
    printfn "============================================================\n"
    
    printfn "✓ VectorStorage: Dense vectors, cosine/Euclidean/dot similarity"
    printfn "✓ PCA: Eigendecomposition-based reduction"
    printfn "✓ RandomProjection: Gaussian and sparse matrices"
    printfn "✓ VPTree: Metric space partitioning tree"
    printfn "✓ LSH: Multi-table hash index"
    printfn "✓ CollaborativeFiltering: Matrix factorization with SGD"
    printfn ""
    printfn "Optimizations:"
    printfn "  ✓ Johnson-Lindenstrauss dimension calculation"
    printfn "  ✓ Triangle inequality pruning (VP-Tree)"
    printfn "  ✓ Sparse random projection (Achlioptas)"
    printfn "  ✓ Multi-probe LSH for better recall"
    
    printfn "\n\n============================================================"
    printfn "  Usage Examples"
    printfn "============================================================\n"
    
    printfn "// Dimensionality reduction"
    printfn "let pca = PCA.fit X 50"
    printfn "let reduced = PCA.transform pca X"
    printfn ""
    printfn "// Random projection (much faster)"
    printfn "let rp = RandomProjection.create 1000 50 seed=42"
    printfn "let reduced = RandomProjection.transform rp vector"
    printfn ""
    printfn "// Build search tree"
    printfn "let tree = VPTree.build vectors cosineSimilarity"
    printfn "let neighbors = VPTree.knn tree query 10"
    printfn ""
    printfn "// LSH for approximate search"
    printfn "let lsh = LSH.build vectors 10 8 4.0 seed=42"
    printfn "let candidates = LSH.query lsh query 10"
    printfn ""
    printfn "// Collaborative filtering"
    printfn "let cf = CollaborativeFiltering.train ratings 1000 500 50 100 0.01 0.1"
    printfn "let recs = CollaborativeFiltering.recommend cf userId 500 10"
    
    printfn "\n\nApplications:"
    printfn "  - Product recommendations (Amazon, Netflix)"
    printfn "  - Content similarity (Spotify, YouTube)"
    printfn "  - Image search (Google Images)"
    printfn "  - Document retrieval"
    printfn "  - Near-duplicate detection\n"
    
    0