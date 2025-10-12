# Vector Search and Information Retrieval Glossary

A comprehensive reference guide to key concepts, metrics, and terminology used in vector search, information retrieval, and related machine learning fields.

- [B](#b) | [C](#c) | [D](#d) | [E](#e) | [F](#f) | [G](#g) | [H](#h) | [I](#i) | [J](#j) | [L](#l) | [M](#m) | [N](#n) | [P](#p) | [Q](#q) | [R](#r) | [S](#s) | [T](#t) | [V](#v)

---

## B

### BM25 (Best Matching 25)

Definition: Ranking function used by search engines to estimate relevance of documents.


Improvement over TF-IDF: Term frequency saturation, document length normalization


Parameters: k1 (term frequency saturation), b (length normalization)


### BERT (Bidirectional Encoder Representations from Transformers)
Definition: Pre-trained transformer model creating contextual word embeddings.

Key Feature: Bidirectional context understanding

Output: 768 or 1024-dimensional vectors per token


### Backpressure
Definition: Mechanism to prevent system overload by slowing input rate.

Implementation: Queue limits, rate limiting, circuit breakers


### Bit Rate
Definition: Number of bits used to represent each vector or component.

Common Values: 1-bit (binary), 8-bit (byte), 32-bit (float)

Impact: Lower bit rates reduce memory but may decrease accuracy


### Bloom Filter
Definition: Space-efficient probabilistic data structure for set membership testing.

Properties: False positives possible, no false negatives

Use Case: Pre-filtering, reducing expensive operations


### Build Time
Definition: Time required to construct search index from data.

Factors: Dataset size, algorithm complexity, hardware resources

Consideration: One-time cost for batch systems, ongoing for streaming


---

## C

### Cache Hit Rate
Definition: Percentage of data requests served from cache vs disk.

Formula: Cache hits / (Cache hits + Cache misses)

Impact: Higher hit rates significantly improve performance


### Circuit Breaker
Definition: Design pattern to prevent cascading failures in distributed systems.

States: Closed (normal), open (failing), half-open (testing recovery)


### Cold Start Problem
Definition: Performance degradation when system caches are empty.

Solution: Cache warming, prefetching, gradual traffic ramp-up


### Compression Ratio
Definition: Ratio of original size to compressed size.

Formula: Original_size / Compressed_size

Example: 32:1 ratio means compressed version is 32× smaller


### Contrastive Learning
Definition: Training approach that learns representations by contrasting positive and negative examples.

Key Idea: Pull similar examples together, push dissimilar examples apart

Examples: SimCLR, InfoNCE, triplet loss


### Cosine Similarity
Definition: Measures angle between vectors, ignoring magnitude.

Formula: cos(θ) = (A · B) / (||A|| × ||B||)

Range: -1 to 1 (higher is more similar)

Best For: Text embeddings, normalized vectors, when magnitude is irrelevant


### Curse of Dimensionality
Definition: Phenomena where high-dimensional spaces behave counterintuitively.

Effects: Distance concentration, sparsity, increased computation

Mitigation: Dimensionality reduction, approximate algorithms


---

## D

### Dense Passage Retrieval (DPR)
Definition: Approach using dense vector representations for passage retrieval.

Components: Question encoder, passage encoder

Advantage: Better than sparse methods for semantic matching


### Disk I/O
Definition: Read/write operations to persistent storage.

Impact: Can dominate query latency for large datasets

Optimization: Memory mapping, SSD storage, prefetching


### Dot Product
Definition: Sum of products of corresponding vector components.

Formula: A · B = Σ(Ai × Bi)

Range: -∞ to ∞ (higher typically more similar)

Best For: When both direction and magnitude matter


---

## E

### Embedding
Definition: Dense vector representation of data (text, images, etc.) in continuous space.

Properties: Fixed dimensionality, semantic similarity preserved as geometric proximity

Example: Word2Vec, BERT, ResNet features


### Euclidean Distance (L2)
Definition: Straight-line distance between vectors in Euclidean space.

Formula: d = √(Σ(Ai - Bi)²)

Range: 0 to ∞ (lower is more similar)

Best For: Image embeddings, when magnitude matters, normalized embeddings


---

## F

### F1-Score
Definition: Harmonic mean of precision and recall, providing a balanced measure.

Formula: F1 = 2 × (Precision × Recall) / (Precision + Recall)

Range: 0.0 to 1.0 (higher is better)

Use Case: When you need a single metric balancing precision and recall


### Faceted Search
Definition: Search interface allowing filtering by multiple attributes simultaneously.

Example: E-commerce filters for price, brand, color, size

Implementation: Combines text/vector search with structured filters


---

## G

### Graph Data Structure
Definition: Network of nodes connected by edges, used in HNSW and NSW algorithms.

Properties: Nodes = vectors, edges = similarity relationships

Operations: Navigation, insertion, deletion


---

## H

### Hamming Distance
Definition: Number of positions where binary vectors differ.

Formula: Count of positions where Ai ≠ Bi

Best For: Binary vectors, hash codes, error detection


### HNSW (Hierarchical Navigable Small World)
Definition: Graph-based approximate nearest neighbor algorithm using multiple layers for efficient navigation.

Key Features: Logarithmic search complexity, high recall, tunable parameters

Parameters: M (connections per node), ef_construction (build quality), ef_search (query quality)

Best For: High-accuracy applications, moderate memory budgets


---

## I

### Index Size
Definition: Storage space required for search index.

Measurement: Bytes, compression ratio vs original data

Factors: Algorithm choice, parameters, compression techniques


### Inverted Index
Definition: Data structure mapping each unique term to list of documents containing it.

Structure: Term → [Doc1, Doc2, ...] with frequency/position information

Use Case: Foundation of text search engines


### IVF (Inverted File Index)
Definition: Clustering-based algorithm that partitions vector space and searches only relevant clusters.

Key Features: Predictable performance, good for large datasets, memory efficient

Parameters: nlist (number of clusters), nprobes (clusters searched per query)

Best For: Large-scale deployments, memory-constrained environments


---

## J

### Jaccard Similarity
Definition: Size of intersection divided by size of union of two sets.

Formula: J(A,B) = |A ∩ B| / |A ∪ B|

Range: 0 to 1 (higher is more similar)

Best For: Set-based data, sparse binary vectors, document similarity


---

## L

### LSH (Locality Sensitive Hashing)
Definition: Hash-based algorithm where similar vectors have high probability of same hash values.

Key Features: Sub-linear query time, probabilistic guarantees

Best For: Very high-dimensional sparse vectors, streaming applications


### LSH Hash Table
Definition: Hash table where similar items have high probability of same hash bucket.

Property: Locality-sensitive hash functions

Use Case: Approximate nearest neighbor search


### Latency
Definition: Time required to process a single query and return results.

Measurement: Milliseconds (ms) or microseconds (μs)

Components: Index access, computation, network transmission

Targets: <1ms (real-time), <10ms (interactive), <100ms (batch)


### Load Balancing
Definition: Distribution of computational work across multiple processors or machines.

Goals: Maximize throughput, minimize latency, ensure fault tolerance

Methods: Round-robin, consistent hashing, load-aware routing


---

## M

### Manhattan Distance (L1)
Definition: Sum of absolute differences along each dimension.

Formula: d = Σ|Ai - Bi|

Range: 0 to ∞ (lower is more similar)

Best For: Sparse vectors, categorical data, high-dimensional spaces


### Mean Average Precision (MAP)
Definition: Average of precision values calculated at each relevant document position.

Formula: MAP = (1/|Q|) × Σ(AP(q)) for all queries q in Q

Use Case: Comprehensive evaluation metric for ranking quality across multiple queries


### Mean Reciprocal Rank (MRR)
Definition: Average of reciprocal ranks of the first relevant document for each query.

Formula: MRR = (1/|Q|) × Σ(1/rank_of_first_relevant_result)

Use Case: Evaluating systems where users typically want just one good result


### Memory Mapping
Definition: Technique mapping file contents directly into memory address space.

Advantage: OS handles caching, reduces memory copies

Use Case: Large datasets that don't fit in RAM


### Memory Usage
Definition: RAM required for index storage and query processing.

Components: Vector storage, graph structures, codebooks, caches

Optimization: Compression, memory mapping, tiered storage


---

## N

### NSW (Navigable Small World)
Definition: Predecessor to HNSW, uses single-layer graph for vector navigation.

Key Features: Simple implementation, good baseline performance

Limitation: Single layer limits scalability compared to HNSW


### Normalized Discounted Cumulative Gain (NDCG)
Definition: Ranking quality metric that considers both relevance and position.

Formula: NDCG@K = DCG@K / IDCG@K

Range: 0.0 to 1.0 (higher is better)

Use Case: When document relevance has multiple levels (not just binary relevant/irrelevant)


---

## P

### Precision
Definition: The fraction of retrieved documents that are relevant to the query.

Formula: Precision = (True Positives) / (True Positives + False Positives)

Range: 0.0 to 1.0 (higher is better)

Example: If a search returns 10 documents and 7 are relevant, precision = 0.7

Use Case: Critical when false positives are costly (medical diagnosis, legal research)


### Precision@K
Definition: Precision calculated only for the top K retrieved results.

Formula: P@K = (Relevant documents in top K) / K

Example: P@5 = 0.8 means 4 out of top 5 results were relevant

Use Case: Evaluating search quality for user-facing applications where only top results matter


### Product Quantization (PQ)
Definition: Compression technique that decomposes vectors into subvectors and quantizes each independently.

Key Features: Extreme memory reduction (10-100x), approximate distance computation

Parameters: m (number of subquantizers), k (centroids per codebook)

Best For: Memory-critical applications, mobile/edge deployments


---

## Q

### Query Expansion
Definition: Process of adding related terms to original query to improve retrieval.

Methods: Thesaurus-based, relevance feedback, word embeddings

Goal: Bridge vocabulary gap between queries and documents


### Quantization
Definition: Process of reducing precision of numerical values to save memory.

Types: Scalar quantization, vector quantization, product quantization

Trade-off: Memory savings vs accuracy loss


---

## R

### Recall
Definition: The fraction of relevant documents that are successfully retrieved.

Formula: Recall = (True Positives) / (True Positives + False Negatives)

Range: 0.0 to 1.0 (higher is better)

Example: If there are 20 relevant documents total and 15 are found, recall = 0.75

Use Case: Important when missing relevant results is costly (academic research, comprehensive analysis)


### Recall@K
Definition: Fraction of all relevant documents found within the top K results.

Formula: R@K = (Relevant documents in top K) / (Total relevant documents)

Use Case: Understanding how many relevant items users can find without scrolling


### Relevance Feedback
Definition: Technique to improve search results based on user feedback.

Types: Explicit (user marks relevant), implicit (click behavior)

Implementation: Query modification, result re-ranking


### Replication
Definition: Maintaining copies of data across multiple nodes.

Benefits: Fault tolerance, load distribution, geographic distribution

Consistency: Strong vs eventual consistency trade-offs


---

## S

### Sentence-BERT (SBERT)
Definition: Modified BERT architecture optimized for sentence-level embeddings.

Advantage: Enables efficient sentence similarity computation

Use Case: Semantic search, clustering, classification


### SIMD (Single Instruction, Multiple Data)
Definition: Computer architecture allowing parallel processing of multiple data elements.

Application: Accelerating distance calculations, vector operations

Examples: AVX2, AVX-512 instruction sets


### Sharding
Definition: Horizontal partitioning of data across multiple nodes.

Purpose: Distribute load, enable parallel processing

Challenge: Load balancing, cross-shard queries


### Stemming
Definition: Process of reducing words to their root form.

Example: "running," "runs," "ran" → "run"

Purpose: Improve recall by matching morphological variants


### Stop Words
Definition: Common words filtered out during text processing.

Examples: "the," "and," "or," "but"

Rationale: Low discriminative value, high frequency


---

## T

### TF-IDF (Term Frequency-Inverse Document Frequency)
Definition: Numerical statistic reflecting term importance in document relative to collection.

Formula: TF-IDF = (term_freq / total_terms) × log(total_docs / docs_with_term)

Purpose: Downweight common terms, upweight rare discriminative terms


### Throughput (QPS)
Definition: Number of queries processed per second.

Measurement: Queries Per Second (QPS)

Factors: System resources, query complexity, concurrency

Scaling: Often limited by CPU, memory bandwidth, or storage I/O


### Transformer
Definition: Neural network architecture using attention mechanisms for sequence processing.

Key Innovation: Self-attention allows modeling long-range dependencies

Examples: BERT, GPT, T5, RoBERTa


---

## V

### Vector Space Model
Definition: Mathematical model representing documents as vectors in multi-dimensional space.

Key Idea: Similar documents have similar vector representations

Applications: Information retrieval, recommendation systems


---

## K

### K-D Tree
Definition: Binary tree data structure for organizing points in k-dimensional space.

Properties: Recursive spatial subdivision

Limitation: Performance degrades in high dimensions (curse of dimensionality)


---

