# Precision and Recall in Vector Search

Understanding how precision and recall work in vector search systems is fundamental to building effective retrieval pipelines. These two metrics form the foundation for diagnosing performance issues and making informed optimization decisions in RAG (Retrieval-Augmented Generation) systems.

## Understanding the Basics

Let's start with a concrete example to build intuition. Imagine you've retrieved 120 documents from your vector database, and upon inspection, 36 of them are truly relevant to your query. If your entire corpus contains 60 relevant documents total, your metrics would look like this:

- Precision = 36/120 = 0.30 (30% of retrieved documents are relevant)
- Recall = 36/60 = 0.60 (you found 60% of all relevant documents)

This fundamental relationship reveals the classic tradeoff: precision measures the purity of your results, while recall measures their completeness.

## Why Not Accuracy, Specificity, or Sensitivity?

You might wonder why we focus on precision and recall rather than other classification metrics like accuracy, specificity, or sensitivity. The answer lies in the unique characteristics of information retrieval tasks.

**Accuracy** measures the proportion of all predictions (both positive and negative) that are correct. In typical classification problems with balanced classes, this works well. However, in vector search, the number of documents you *don't* retrieve (true negatives) completely dominates the calculation. Imagine a corpus with 1 million documents where only 60 are relevant to a query. If you retrieve 120 documents with 36 relevant ones, your accuracy would be:

- True Positives (TP) = 36 (relevant docs retrieved)
- False Positives (FP) = 84 (irrelevant docs retrieved)
- True Negatives (TN) = 999,916 (irrelevant docs correctly not retrieved)
- False Negatives (FN) = 24 (relevant docs missed)
- Accuracy = (36 + 999,916) / 1,000,000 = 0.999952

This 99.99% accuracy sounds impressive but tells us almost nothing useful. Even a terrible search system that returns random results would have extremely high accuracy simply because most documents are irrelevant. The metric is essentially measuring our ability to *not* retrieve irrelevant documents from a vast pool, which isn't particularly informative.

**Specificity** (also called true negative rate) measures what proportion of truly irrelevant documents we correctly avoided retrieving. Like accuracy, this metric suffers from the class imbalance problem. In our example above, specificity = 999,916 / (999,916 + 84) = 0.999916. Again, this looks great but provides little actionable insight because we're dealing with such a massive pool of irrelevant documents.

**Sensitivity** is actually identical to recall—it's the same metric with a different name commonly used in medical diagnostics and other fields. Sensitivity measures the proportion of relevant items successfully retrieved, which is exactly what recall does. In the medical context, you might talk about a test's sensitivity to detecting a disease, while in information retrieval, we talk about a system's recall of relevant documents. The mathematics are identical: TP / (TP + FN).

The practical reality of vector search is that we care deeply about two things:

1. Of the documents we retrieve, how many are actually relevant? (Precision)
2. Of all the relevant documents that exist, how many did we find? (Recall)

These metrics directly address the core user experience questions: "Is what I'm seeing useful?" and "Am I missing important information?" The extreme class imbalance in information retrieval makes accuracy and specificity mathematically uninteresting, while sensitivity is simply recall by another name.

## The Two-Stage Retrieval Pipeline

Modern vector search systems typically operate in two stages: candidate generation and reranking. Understanding where precision and recall matter in each stage is crucial for effective optimization.

### When Recall Problems Occur at the Candidate Stage

If your recall@200 is low but precision@10 looks fine, your first adjustment should be to increase candidate breadth. This means raising parameters like k (number of results), HNSW's efSearch, or IVF's nprobe settings. The strategy here is to fetch wider before reranking—you can't rank what you haven't retrieved.

Consider this scenario: the perfect answer exists in your database, your reranker is strong, but that perfect answer never appears in the candidate set. This indicates a candidate stage recall miss. Your ANN (Approximate Nearest Neighbor) search is too tight, or your quantization is too coarse. The solution is to widen your search probes or reduce compression.

Here's an interesting pattern that often emerges: after raising k from 50 to 300, recall@300 improves (as expected), but precision@10 actually gets worse. Why? The wider fetch brought in more near-duplicates and off-topic items. Your reranker or filters now need to work harder to clean up the top results.

### The "Wide Then Wise" Strategy

When working with RAG systems, you might wonder whether to increase k or invest more in reranking budget for better answers. The recommended approach follows a "wide then wise" philosophy: start by increasing k to improve recall, then apply more sophisticated reranking (for example, cross-encode the top 200 candidates and return the top 5-10).

## Understanding Approximate Nearest Neighbor Search

Before diving into specific optimization techniques, it's important to understand the fundamental tradeoff at the heart of modern vector search: the choice between exact and approximate nearest neighbor search.

In theory, finding the most similar vectors to a query would involve comparing your query against every single vector in your database—a brute-force approach that guarantees you'll find the true nearest neighbors. For a database with 1 million vectors of 1,536 dimensions each, this means performing 1 million distance calculations per query. While this exhaustive search is feasible for small datasets, it quickly becomes impractical as your corpus grows into the millions or billions of vectors.

This is where Approximate Nearest Neighbor (ANN) algorithms come into play. ANN methods make a calculated tradeoff: they sacrifice some accuracy in exchange for dramatically faster search speeds. Instead of checking every vector, they use clever data structures and algorithms to quickly narrow down the search space to a smaller set of promising candidates.

The "approximate" nature of ANN means you might not always find the absolute nearest neighbors—but in practice, you usually find neighbors that are close enough. The key insight is that for most applications, getting 95% of the true nearest neighbors in milliseconds is far more valuable than getting 100% in several seconds.

This approximation directly impacts recall. When your ANN algorithm is configured too aggressively (favoring speed over accuracy), it might miss some of the truly relevant documents—they exist in your database, but the algorithm's shortcuts cause it to overlook them. This is what we mean by "candidate stage recall miss" mentioned earlier. The perfect answer exists, but your ANN search is too tight, skipping over parts of the vector space where that answer resides.

Different ANN algorithms use different strategies to achieve this speedup, and each comes with its own set of parameters that control the precision-speed tradeoff. For a comprehensive exploration of ANN algorithms including their mathematical foundations and implementation details, see [Vector Search Algorithms Deep Dive](index_deep_dive.md).

## Index-Specific Tuning

Different index types require different approaches to optimize recall, and understanding their internal mechanics helps you make informed tuning decisions.

### HNSW Indexes

Raising efSearch in HNSW indexes increases recall by searching more neighbors, though this comes at the cost of higher latency and CPU usage. This parameter directly controls the breadth of your search graph traversal.

For a detailed understanding of HNSW's hierarchical navigation, graph construction, and advanced parameter optimization strategies, see the [HNSW section in Vector Search Algorithms](index_deep_dive.md#hnsw-hierarchical-navigable-small-world).

### IVF-PQ Indexes

With IVF-PQ indexes, if your nlist is too small and nprobe is low, you'll observe a characteristic symptom: true neighbors get stuck in other partitions, leading to recall loss. The fix is to raise nlist (creating more partitions) and increase nprobe (probing more of those partitions during search).

For in-depth coverage of IVF's clustering approach, mathematical foundations, and Product Quantization compression techniques, see the [IVF section in Vector Search Algorithms](index_deep_dive.md#ivf-inverted-file-index).

## Hybrid Search Strategies

Blending BM25 with vector search becomes valuable for specific query patterns. This hybrid approach particularly helps with queries containing rare tokens, names, IDs, or exact-phrase requirements. The blending boosts both precision and recall for lexically-heavy search intents.

## Common Diagnostic Patterns

### The Threshold Dilemma

When you lower the vector score threshold, you might see more relevant items appear in your results—but also more junk at the top. This is a classic precision-recall tradeoff in action: recall increases, precision decreases. The solution is to keep the threshold lower for recall gains, but add reranking or filters to restore purity at the top of your results.

### Filter Impact

Adding hard filters (like language=en) typically increases precision by removing wrong-language results, but can decrease recall if some relevant results get filtered out. Understanding this tradeoff helps you make informed decisions about when and where to apply filters.

A critical consideration: when ACL (Access Control List) or tenant filters are applied after candidate fetch, you risk fetching relevant documents only to discard them post-filter, leaving you with a low-recall final set. The best practice is to apply filters as early as possible in your pipeline, or expand k before filtering to compensate.

### Redundancy Problems

If your top-10 results feel redundant with near-duplicates, your perceived precision@10 drops even if the results are technically relevant. The fix involves deduplication combined with diversity sampling—cluster similar results and pick representatives before final display.

## Latency Budget Decisions

Suppose you have 80ms of extra latency budget to spend. Should you invest it in higher efSearch or cross-encoder reranking? The decision depends on your diagnostic findings:

- If target documents don't appear in candidates at all, spend the budget on efSearch to improve recall
- If candidates contain the right documents but rank them poorly, invest in reranking to improve precision

## User Behavior as a Signal

Real-world usage patterns often reveal hidden issues that metrics might miss.

When evaluation shows good precision@10 but user satisfaction remains low, the likely cause is recall holes—users can't find what they're looking for—or intent mismatch. This suggests checking recall@k metrics and running query diagnostics.

If logging reveals frequent query reformulations, where users add more words and then get good results, this signals a recall miss on the first attempt. The solution is to widen search effort or implement BM25+vector hybridization.

## Embedding and Model Considerations

Switching to smaller embeddings to cut costs often causes recall@200 to drop. This happens because less expressive vectors create weaker neighborhood geometry, resulting in lower recall. You can compensate by using higher k values, increasing probe counts, or implementing smarter hybrid approaches.

When rerankers are trained only on common (head) queries, performance on rare (tail) queries typically suffers. The fix involves retraining with tail data, adding lexical features, and using calibration techniques like score blending to stabilize precision across the query distribution.

## Targeted Optimization Scenarios

### High Recall, Low Precision

If you measure Recall@200 = 0.9 but Precision@10 = 0.2, your candidate stage is performing well—the issue is in ranking. The targeted fix involves improving reranking and feature engineering: use cross-encoders, blend with BM25, adjust field weights, implement deduplication, and apply business rules.

### Exact Match Tasks

For tasks like "find the exact policy clause," you need different strategies at each stage:

- Candidate stage: Prioritize high recall—you absolutely cannot miss the target clause
- Reranking stage: Emphasize precision using a cross-encoder on the top 200 candidates

## Production Best Practices

For a factual RAG system, a sensible starting configuration looks like this:

- Set k=200–500 for candidate generation
- Use high efSearch/nprobe values to ensure good recall
- Implement BM25+vector hybrid search
- Cross-encode the top ~200 candidates
- Return the top 5–10 final results
- Continuously log misses and tune from there

This "wide then wise" approach captures most relevant information in the candidate stage, then applies sophisticated ranking to surface the best results to users. The key is to instrument your system well, monitor real-world performance, and iterate based on actual usage patterns rather than intuition alone.