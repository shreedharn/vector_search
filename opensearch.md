# OpenSearch: From Lucene to Production Vector Search

A comprehensive guide to implementing vector search in OpenSearch, from understanding Lucene fundamentals to building production-ready multi-modal search systems.

## Overview

OpenSearch extends Apache Lucene's robust document storage and search capabilities with specialized vector search functionality, creating a unified platform for both traditional text search and modern vector-based semantic search. This guide covers the architectural foundations, implementation patterns, and advanced applications for building production vector search systems.

## Understanding Apache Lucene

Before diving into OpenSearch implementation, it's essential to understand Apache Lucene—the powerful search library that forms OpenSearch's foundation. Lucene provides the core indexing and search capabilities that OpenSearch builds upon.

### What is Lucene?

Apache Lucene is a high-performance, full-featured text search engine library written in Java. Originally created by Doug Cutting in 1999, Lucene has evolved into the de facto standard for building search applications and powers numerous search platforms including OpenSearch, Elasticsearch, and Apache Solr.

Core Capabilities:

- Inverted Index: Efficient data structure for full-text search
- Query Parsing: Rich query syntax for complex search expressions
- Scoring Models: Pluggable relevance scoring (BM25, TF-IDF, custom)
- Document Storage: Compressed field storage and retrieval
- Scalability: Designed for indexing and searching large document collections

### Lucene's Segment Architecture

Understanding Lucene's segment-based architecture is crucial for optimizing OpenSearch vector search performance.

Segments: Immutable Building Blocks

Lucene organizes indexed data into segments—immutable, self-contained indexes that can be searched independently:

```
Index Structure:
my_index/
├── segment_0/
│   ├── _0.cfs      (Compound file with all segment data)
│   ├── _0.cfe      (Compound file entries)
│   └── _0.si       (Segment info)
├── segment_1/
│   ├── _1.cfs
│   ├── _1.cfe
│   └── _1.si
└── segments_N      (Current segments metadata)
```

Key Characteristics:

1. Immutability: Once written, segments never change—this enables efficient caching and concurrent access
2. Incremental Indexing: New documents create new segments rather than modifying existing ones
3. Parallel Search: Multiple segments can be searched concurrently across threads
4. Merge Policy: Background process merges smaller segments into larger ones for optimization

Why Segments Matter for Vector Search:

- Memory Mapping: Immutable segments allow efficient memory-mapped file access for large vector datasets
- Cache Efficiency: Vectors in segments can be cached effectively without invalidation concerns
- Parallel Processing: Vector search across segments can leverage multi-core processors
- Index Growth: New vectors added as new segments without disrupting existing searches

### Lucene's Inverted Index

The inverted index is Lucene's core data structure for text search, and understanding it helps contextualize how vector indexes integrate.

Inverted Index Structure:

```
Term Dictionary:
"machine"  → [doc1:pos[5,23], doc5:pos[12], doc8:pos[3,17,44]]
"learning" → [doc1:pos[6,24], doc3:pos[8], doc5:pos[13]]
"vector"   → [doc2:pos[1], doc5:pos[2], doc9:pos[15]]

Where:
- Term: The indexed word/token
- Document ID: Which documents contain this term
- Positions: Where in each document the term appears
```

Query Processing:

1. Term Lookup: Find documents containing query terms in the inverted index (O(log N) with binary search)
2. Intersection/Union: Combine document lists based on boolean operators (AND, OR, NOT)
3. Scoring: Calculate relevance scores using BM25 or other algorithms
4. Ranking: Return top-k results sorted by score

Integration with Vector Search:

OpenSearch stores vector indexes alongside inverted indexes within the same segments:

```
Segment Contents:
├── inverted_index/     (Traditional term dictionary and postings)
├── stored_fields/      (Original document content)
├── doc_values/         (Column-oriented field data)
├── vector_data/        (Raw vector embeddings)
└── vector_index/       (HNSW or IVF graph structures)
```

This unified storage enables powerful hybrid queries that combine text filters with vector similarity searches.

### Lucene's Query Model

Lucene provides a flexible query model that OpenSearch extends for vector search:

Traditional Query Types:

- TermQuery: Exact term matching
- BooleanQuery: Combine queries with AND, OR, NOT
- PhraseQuery: Match exact phrases
- RangeQuery: Numeric or date range filtering
- FuzzyQuery: Approximate string matching

Vector Query Integration:

OpenSearch adds vector query types that integrate seamlessly with Lucene's query model:

- KnnVectorQuery: Find k-nearest neighbors in vector space
- Hybrid Queries: Combine vector similarity with text/filter constraints

Example Query Flow:

```
User Query: "machine learning" + vector similarity + category="AI"

Lucene Processing:
1. Parse text query → TermQuery("machine") AND TermQuery("learning")
2. Parse filter → TermQuery("category:AI")
3. Parse vector query → KnnVectorQuery(vector=[...], k=100)
4. Execute combined query across all segments
5. Merge and rank results
```

### Why OpenSearch Chose Lucene

OpenSearch's decision to build on Lucene provides several strategic advantages:

Proven Foundation:

- 20+ years of development and optimization
- Battle-tested at massive scale (Wikipedia, Twitter, LinkedIn)
- Active community and continuous improvement

Unified Data Model:

- Store vectors and text in the same index
- Single query API for hybrid searches
- Consistent operational model (sharding, replication, merging)

Performance Optimizations:

- Highly optimized file I/O and memory management
- Advanced compression algorithms
- Efficient query execution engine

Extensibility:

- Plugin architecture for custom functionality
- Codec system for custom index formats
- Flexible scoring and ranking models

Understanding this Lucene foundation helps you optimize OpenSearch vector search by:
- Configuring merge policies for vector-heavy workloads
- Managing segment sizes for optimal memory usage
- Leveraging segment-level parallelism in queries
- Tuning refresh intervals for index performance

## OpenSearch Vector Architecture

OpenSearch extends Apache Lucene's robust document storage and search capabilities with specialized vector search functionality, creating a unified platform for both traditional text search and modern vector-based semantic search.

### Core Architecture Components

Integrated Storage Model:

OpenSearch stores vectors alongside traditional document fields, enabling rich queries that combine text filters, metadata constraints, and vector similarity in a single operation.

```
Document Structure:
{
  "_id": "doc_123",
  "_source": {
    "title": "Machine Learning Fundamentals",
    "content": "Introduction to ML algorithms...",
    "category": "education",
    "timestamp": "2024-01-15T10:00:00Z",
    "content_vector": [0.1, -0.2, 0.8, ...],  // 384-dimensional vector
    "title_vector": [0.3, 0.1, -0.4, ...]     // Separate vector for title
  }
}
```

Segment-Based Vector Storage:

OpenSearch leverages Lucene's segment architecture for vector storage, providing several key benefits:

1. Immutable Segments: Once written, segments don't change, enabling efficient memory mapping and caching
2. Parallel Processing: Multiple segments can be searched concurrently
3. Incremental Updates: New data creates new segments rather than modifying existing ones
4. Memory Management: Vectors stored in off-heap memory-mapped files

Vector Index Files per Segment:

```
Segment Directory:
├── vectors.vec      # Raw vector data (memory-mapped)
├── vector_meta.vem  # Vector metadata and mappings
├── hnsw_graph.hng   # HNSW graph structure (if used)
├── ivf_clusters.ivc # IVF cluster assignments (if used)
└── documents.json   # Traditional Lucene document storage
```

### Memory Management Strategy

Off-Heap Vector Storage:

OpenSearch stores vector data off-heap to avoid garbage collection pressure and enable memory mapping:

```python
# Memory allocation example for 1M vectors, 384 dimensions
vector_storage = {
    "raw_vectors": "1M × 384 × 4 bytes = 1.54GB (memory-mapped)",
    "hnsw_graph": "1M × 24 connections × 4 bytes = 96MB (direct memory)",
    "metadata": "1M × 64 bytes = 64MB (heap)",
    "total_memory": "~6GB including system overhead"
}
```

Query Processing Memory:

Temporary structures for query processing use on-heap memory:
- Query vector parsing and normalization
- Similarity score calculations
- Result ranking and aggregation

Caching Strategy:

- Vector cache: Recently accessed vectors cached in direct memory
- Graph cache: Frequently traversed graph regions kept in memory
- Query cache: Common query patterns cached for repeated execution

### Engine Architecture

Lucene Integration:

OpenSearch vector search builds on Lucene's KnnVectorField implementation while adding:

- Multiple algorithm support (HNSW, IVF)
- Advanced parameter tuning
- Production-ready optimizations

Query Execution Pipeline:

```
1. Query Parsing → Parse knn/vector query syntax
2. Vector Validation → Verify dimensions and format
3. Algorithm Selection → Choose HNSW vs IVF based on index config
4. Segment Search → Execute vector search across all segments
5. Score Aggregation → Combine results from multiple segments
6. Filter Application → Apply any additional query filters
7. Result Ranking → Final ranking and relevance scoring
```

## Index Configuration and Setup

Proper index configuration is crucial for optimal vector search performance. OpenSearch provides extensive configuration options for different algorithms and use cases.

### Basic Vector Field Configuration

Simple Vector Field:

```json
{
  "mappings": {
    "properties": {
      "content_vector": {
        "type": "knn_vector",
        "dimension": 384,
        "space_type": "cosinesimil"
      },
      "title": {"type": "text"},
      "content": {"type": "text"},
      "category": {"type": "keyword"},
      "timestamp": {"type": "date"}
    }
  }
}
```

Space Type Options:

- "cosinesimil": Cosine similarity (recommended for text embeddings)
- "l2": Euclidean distance (good for normalized embeddings)
- "l1": Manhattan distance (robust for sparse vectors)
- "linf": Maximum distance (specialized use cases)

### HNSW Configuration

Production HNSW Setup:

```json
{
  "settings": {
    "index": {
      "knn": true,
      "number_of_shards": 3,
      "number_of_replicas": 1,
      "refresh_interval": "30s"
    }
  },
  "mappings": {
    "properties": {
      "content_vector": {
        "type": "knn_vector",
        "dimension": 384,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "lucene",
          "parameters": {
            "ef_construction": 256,  # Higher = better quality, slower build
            "m": 32                  # Higher = better recall, more memory
          }
        }
      }
    }
  }
}
```

Parameter Selection Guidelines:

| Use Case | ef_construction | M | Reasoning |
|----------|----------------|---|-----------|
| Development/Testing | 128 | 16 | Fast iteration, adequate quality |
| Production (Balanced) | 256 | 24 | Good performance, manageable resources |
| High Accuracy | 512 | 32 | Maximum quality, higher resource usage |
| Memory Constrained | 128 | 12 | Reduced memory footprint |
| Large Scale (10M+) | 256 | 24 | Balanced for large datasets |

### IVF Configuration

IVF Index Setup:

```json
{
  "mappings": {
    "properties": {
      "content_vector": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "ivf",
          "space_type": "l2",
          "engine": "lucene",
          "parameters": {
            "nlist": 1024,     # Number of clusters
            "nprobes": 64      # Default search width
          }
        }
      }
    }
  }
}
```

IVF Parameter Calculation Framework:

*Cluster Count Formula:*

- Base: √expected_vector_count
- Adjusted: base × max(1.0, dimensions/512)
- Constrained: max(32, calculated_value)

*Search Width:*

- Conservative: 10% of cluster count (minimum 8)

*Memory Estimation:*

- Formula: vector_count × dimensions × 4 bytes

*Example Results:*

- 500K vectors, 384 dims → nlist=707, nprobes=71, ~0.7GB
- 5M vectors, 768 dims → nlist=3,464, nprobes=346, ~14.4GB

### Multi-Vector Field Configuration

Multiple Vector Fields for Different Purposes:

```json
{
  "mappings": {
    "properties": {
      "title": {"type": "text"},
      "content": {"type": "text"},
      "category": {"type": "keyword"},

      "title_vector": {
        "type": "knn_vector",
        "dimension": 384,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "parameters": {"ef_construction": 256, "m": 24}
        }
      },

      "content_vector": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "parameters": {"ef_construction": 256, "m": 32}
        }
      },

      "image_vector": {
        "type": "knn_vector",
        "dimension": 512,
        "method": {
          "name": "ivf",
          "space_type": "l2",
          "parameters": {"nlist": 512, "nprobes": 32}
        }
      }
    }
  }
}
```

## Query Patterns and Implementation

OpenSearch provides flexible query patterns for vector search, from simple k-nearest neighbor queries to complex hybrid searches combining text, filters, and vector similarity.

### Basic Vector Search Queries

Simple KNN Query:

```json
{
  "size": 10,
  "query": {
    "knn": {
      "content_vector": {
        "vector": [0.1, -0.2, 0.8, ...],
        "k": 10
      }
    }
  }
}
```

KNN with Filters:

```json
{
  "size": 10,
  "query": {
    "bool": {
      "must": [
        {
          "knn": {
            "content_vector": {
              "vector": [0.1, -0.2, 0.8, ...],
              "k": 100
            }
          }
        }
      ],
      "filter": [
        {"term": {"category": "technology"}},
        {"range": {"timestamp": {"gte": "2024-01-01"}}}
      ]
    }
  }
}
```

### Hybrid Search Queries

Combining Text and Vector Search:

```json
{
  "size": 10,
  "query": {
    "bool": {
      "should": [
        {
          "multi_match": {
            "query": "machine learning algorithms",
            "fields": ["title^3", "content"],
            "type": "best_fields"
          }
        },
        {
          "knn": {
            "content_vector": {
              "vector": [0.2, -0.1, 0.9, ...],
              "k": 100
            }
          }
        }
      ]
    }
  }
}
```

Query-Time Parameter Tuning:

```json
{
  "size": 10,
  "query": {
    "knn": {
      "content_vector": {
        "vector": [0.1, -0.2, 0.8, ...],
        "k": 50,
        "ef_search": 200  # HNSW-specific: higher = better accuracy
      }
    }
  }
}
```

### Reranking in OpenSearch

OpenSearch provides several built-in mechanisms for implementing reranking, from simple rescoring queries to integration with external machine learning models. Understanding these capabilities enables you to improve search relevance significantly.

Basic Rescore Query Structure:

OpenSearch's `rescore` query allows you to apply a secondary query to refine the top results from your initial search:

```json
{
  "query": {
    "bool": {
      "must": [
        {
          "multi_match": {
            "query": "wireless headphones",
            "fields": ["title^2", "description"]
          }
        }
      ]
    }
  },
  "rescore": {
    "window_size": 50,
    "query": {
      "rescore_query": {
        "function_score": {
          "functions": [
            {
              "field_value_factor": {
                "field": "rating",
                "factor": 1.2,
                "modifier": "log1p"
              }
            },
            {
              "field_value_factor": {
                "field": "review_count",
                "factor": 0.1,
                "modifier": "sqrt"
              }
            }
          ]
        }
      },
      "query_weight": 0.7,
      "rescore_query_weight": 0.3
    }
  }
}
```

Key Parameters:

- window_size: Number of top documents to rescore (typically 50-200)
- query_weight: Weight given to original query score (0.0-1.0)
- rescore_query_weight: Weight given to rescore query score (0.0-1.0)

### Advanced Function Scoring

Multi-Signal Reranking:

Combine multiple relevance signals for sophisticated ranking:

```json
{
  "query": {
    "function_score": {
      "query": {
        "bool": {
          "should": [
            {
              "match": {
                "title": {
                  "query": "machine learning",
                  "boost": 2.0
                }
              }
            },
            {
              "knn": {
                "content_vector": {
                  "vector": [0.1, -0.2, 0.8],
                  "k": 50
                }
              }
            }
          ]
        }
      },
      "functions": [
        {
          "field_value_factor": {
            "field": "popularity_score",
            "factor": 1.5,
            "modifier": "sqrt",
            "missing": 0
          }
        },
        {
          "gauss": {
            "publish_date": {
              "origin": "now",
              "scale": "30d",
              "decay": 0.5
            }
          }
        },
        {
          "script_score": {
            "script": {
              "source": "Math.log(doc['view_count'].value + 1) * params.factor",
              "params": {
                "factor": 0.2
              }
            }
          }
        }
      ],
      "score_mode": "sum",
      "boost_mode": "multiply"
    }
  }
}
```

Function Types:

- field_value_factor: Use document field values as scoring factors
- gauss/linear/exp: Distance-based decay functions for date, location, numerical ranges
- script_score: Custom scoring logic using Painless scripts
- random_score: Add controlled randomization to prevent result staleness

### Hybrid Search with Reranking

Combining Text and Vector Search with Reranking:

```json
{
  "query": {
    "bool": {
      "should": [
        {
          "multi_match": {
            "query": "sustainable energy solutions",
            "fields": ["title^3", "content", "tags^2"],
            "type": "best_fields"
          }
        },
        {
          "knn": {
            "content_vector": {
              "vector": [0.2, -0.1, 0.9],
              "k": 100
            }
          }
        }
      ]
    }
  },
  "rescore": {
    "window_size": 100,
    "query": {
      "rescore_query": {
        "function_score": {
          "functions": [
            {
              "field_value_factor": {
                "field": "authority_score",
                "factor": 2.0,
                "modifier": "log1p"
              }
            },
            {
              "field_value_factor": {
                "field": "recency_boost",
                "factor": 1.0,
                "modifier": "none"
              }
            }
          ],
          "score_mode": "multiply"
        }
      },
      "query_weight": 0.8,
      "rescore_query_weight": 0.2
    }
  }
}
```

### External Neural Reranking Integration

Pipeline Architecture for Neural Reranking:

Modern OpenSearch deployments often integrate with external reranking services for advanced neural reranking:

Step 1: Initial Retrieval
```bash
# OpenSearch returns top 100-200 candidates
curl -X POST "localhost:9200/documents/_search" \
  -H "Content-Type: application/json" \
  -d '{
    "size": 200,
    "query": {
      "bool": {
        "should": [
          {"match": {"content": "machine learning"}},
          {"knn": {"content_vector": {"vector": [...], "k": 100}}}
        ]
      }
    }
  }'
```

Step 2: Feature Extraction
```python
# Extract additional signals for reranking
features = {
    "query_document_similarity": cosine_similarity(query_vector, doc_vector),
    "user_click_score": user_interaction_data.get(doc_id, 0),
    "content_quality": quality_metrics.get(doc_id, 0.5),
    "temporal_relevance": calculate_temporal_decay(doc.publish_date)
}
```

Step 3: Neural Reranking
```python
# Apply transformer-based reranking model
reranked_scores = neural_reranker.predict(
    query_text=query,
    document_texts=[doc.content for doc in candidates],
    features=features
)
```

Step 4: Result Integration
```python
# Return reranked results to user
final_results = sorted(
    zip(candidates, reranked_scores),
    key=lambda x: x[1],
    reverse=True
)
```

### Performance Optimization

Reranking Performance Tuning:

- Window Size Optimization: Start with 50, increase to 100-200 for better quality
- Weight Balancing: Use 70-80% original query weight, 20-30% rescore weight
- Caching Strategies: Cache rescore results for popular queries
- Async Processing: Implement asynchronous reranking for real-time applications

Resource Management:

```json
{
  "search": {
    "max_buckets": 10000,
    "max_rescore_window": 10000
  },
  "indices": {
    "query": {
      "bool": {
        "max_clause_count": 2048
      }
    }
  }
}
```

## Advanced Applications

### Multi-modal Search

Multi-modal search enables searching across different content types (text, images, audio) using unified vector representations, opening new possibilities for content discovery and retrieval.

#### Understanding Multi-Modal Vector Search

Cross-Modal Understanding:

Multi-modal search transcends traditional single-content-type search by enabling queries across heterogeneous data types. This capability allows users to search for images using text descriptions, find videos using audio queries, or discover text documents using image inputs.

Key Advantages:

- Natural Query Expression: Users can express intent using the most convenient modality
- Content Discovery: Find related content across different media types
- Accessibility: Enable alternative access methods for users with different needs
- Rich Results: Provide diverse result sets combining multiple content types

Technical Foundation:

Multi-modal search relies on embedding models trained on paired data across modalities, such as CLIP (Contrastive Language-Image Pre-training) for text-image pairs, or specialized audio-text models. These models learn shared representations where semantically similar content clusters together regardless of its original format.

Common Use Cases:

- E-commerce: Search for products using text descriptions to find matching images
- Media Libraries: Find videos or images using natural language descriptions
- Educational Content: Discover learning materials across text, video, and image formats
- Research Databases: Cross-reference findings across papers, diagrams, and datasets

#### Cross-Modal Search Architecture

Unified Embedding Space:

Multi-modal search relies on embedding models that map different content types into a shared semantic space where similar concepts cluster together regardless of modality.

Shared Vector Space Design:

The core innovation of multi-modal search lies in creating a unified vector space where different content types can be meaningfully compared. This requires specialized embedding models that understand semantic relationships across modalities.

Implementation Architecture:

```json
{
  "mappings": {
    "properties": {
      "content_id": {"type": "keyword"},
      "content_type": {"type": "keyword"},
      "title": {"type": "text"},
      "description": {"type": "text"},

      "text_embedding": {
        "type": "knn_vector",
        "dimension": 512,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "parameters": {"ef_construction": 256, "m": 32}
        }
      },

      "image_embedding": {
        "type": "knn_vector",
        "dimension": 512,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "parameters": {"ef_construction": 256, "m": 32}
        }
      },

      "unified_embedding": {
        "type": "knn_vector",
        "dimension": 512,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "parameters": {"ef_construction": 256, "m": 32}
        }
      }
    }
  }
}
```

Cross-Modal Query Examples:

*Text-to-Image Search:*
```json
{
  "query": {
    "bool": {
      "must": [
        {"term": {"content_type": "image"}},
        {
          "knn": {
            "unified_embedding": {
              "vector": [0.1, -0.2, 0.8, ...],
              "k": 20
            }
          }
        }
      ]
    }
  }
}
```

*Image-to-Text Search:*
```json
{
  "query": {
    "bool": {
      "must": [
        {"term": {"content_type": "text"}},
        {
          "knn": {
            "unified_embedding": {
              "vector": [0.3, 0.1, -0.4, ...],
              "k": 20
            }
          }
        }
      ]
    }
  }
}
```

Multi-Modal Embedding Models:

- CLIP (OpenAI): Text-image understanding with 512-dimensional embeddings
- ALIGN (Google): Large-scale text-image alignment with 640-dimensional vectors
- AudioCLIP: Extension to audio-text-image modalities
- VideoCLIP: Video-text understanding for temporal content

Practical Implementation Considerations:

- Dimension Alignment: Ensure all modalities use the same vector dimensions
- Normalization: Apply consistent normalization across different embedding models
- Quality Control: Validate cross-modal similarity using human evaluation
- Performance Optimization: Use separate indexes per modality for complex queries

## Production Best Practices

### Index Optimization

Shard Configuration:

```json
{
  "settings": {
    "index": {
      "number_of_shards": 3,        // Balance based on data size
      "number_of_replicas": 1,       // High availability
      "refresh_interval": "30s",     // Reduce for better indexing throughput
      "max_result_window": 10000
    }
  }
}
```

Recommendations:

- Shard count: 1-3 shards per 50GB of data
- Replica count: At least 1 for production
- Refresh interval: 30s-60s for vector-heavy workloads

### Monitoring and Observability

Key Metrics to Monitor:

- Query latency: P50, P95, P99 percentiles
- Index size: Track growth over time
- Memory usage: JVM heap and off-heap memory
- Cache hit rates: Query cache, request cache
- Merge statistics: Segment count and merge times

Example Monitoring Query:

```bash
# Check index statistics
curl -X GET "localhost:9200/_cat/indices/my_vector_index?v&h=index,docs.count,store.size,pri,rep"

# Check node stats
curl -X GET "localhost:9200/_nodes/stats/indices,jvm?pretty"
```

### Scaling Strategies

Horizontal Scaling:

- Add more nodes to distribute vector search load
- Increase shard count for large datasets (>500GB)
- Use dedicated master nodes for cluster stability

Vertical Scaling:

- Increase memory for better vector caching
- Use faster storage (NVMe SSDs) for vector data
- Allocate more CPU cores for parallel segment search

## Summary

OpenSearch provides a powerful, production-ready platform for vector search built on the solid foundation of Apache Lucene. Key takeaways:

1. Lucene Integration: Understanding Lucene's segment architecture and inverted index model is crucial for optimizing vector search performance
2. Flexible Configuration: OpenSearch supports multiple algorithms (HNSW, IVF) with extensive tuning options
3. Hybrid Capabilities: Seamlessly combine text search, filters, and vector similarity in unified queries
4. Advanced Features: Multi-modal search, reranking, and function scoring enable sophisticated applications
5. Production Ready: Built-in monitoring, scaling, and optimization features for enterprise deployments

For related topics, see:

- [Introduction to Search Systems](intro_to_search.md) - Fundamentals of text and vector search
- [Vector Search Algorithms Deep Dive](index_deep_dive.md) - HNSW, IVF, and optimization strategies
- [Precision and Recall in Vector Search](precision_vs_recall.md) - Understanding and optimizing search quality
- [Vector Search Algorithms Deep Dive](index_deep_dive.md) - HNSW, IVF, and optimization strategies

---
