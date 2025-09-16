# OpenSearch Vector Search: Python Examples

This document contains practical Python code examples for implementing vector search with OpenSearch.

## Table of Contents

- [Index Setup and Configuration](#index-setup-and-configuration)
- [Document Indexing](#document-indexing)
- [Basic Vector Search](#basic-vector-search)
- [Advanced Search Techniques](#advanced-search-techniques)
- [Hybrid Search Implementation](#hybrid-search-implementation)
- [Multi-Modal Search](#multi-modal-search)
- [Recommendation Systems](#recommendation-systems)
- [Real-Time Applications](#real-time-applications)

---

## Index Setup and Configuration

### OpenSearch Client Setup and Index Creation

```python
from opensearchpy import OpenSearch
import json

# Client configuration
client = OpenSearch([
    {'host': 'localhost', 'port': 9200}
],
http_auth=('admin', 'admin'),  # Configure authentication
use_ssl=True,
verify_certs=False,
ssl_show_warn=False
)

def create_optimized_vector_index(index_name, vector_config):
    """Create vector index with optimized settings"""

    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": calculate_optimal_shards(vector_config['expected_docs']),
                "number_of_replicas": 1,
                "refresh_interval": "30s",  # Batch refresh for better performance
                "codec": "best_compression",  # Reduce disk usage
                "max_result_window": 50000  # Allow larger result sets
            }
        },
        "mappings": {
            "properties": create_vector_mapping(vector_config)
        }
    }

    try:
        response = client.indices.create(index=index_name, body=index_body)
        print(f"Created index {index_name}: {response}")
        return True
    except Exception as e:
        print(f"Error creating index: {e}")
        return False

def calculate_optimal_shards(expected_docs):
    """Calculate optimal shard count based on data size"""
    if expected_docs < 100_000:
        return 1
    elif expected_docs < 1_000_000:
        return 2
    elif expected_docs < 10_000_000:
        return 3
    else:
        return 5  # Balance parallelism with overhead

def create_vector_mapping(config):
    """Generate vector field mappings"""
    properties = {
        "title": {"type": "text", "analyzer": "standard"},
        "content": {"type": "text", "analyzer": "standard"},
        "category": {"type": "keyword"},
        "timestamp": {"type": "date"},
        "url": {"type": "keyword", "index": False}  # Store but don't index
    }

    # Add vector fields
    for field_name, field_config in config['vector_fields'].items():
        properties[field_name] = {
            "type": "knn_vector",
            "dimension": field_config['dimension'],
            "method": {
                "name": field_config['algorithm'],
                "space_type": field_config['space_type'],
                "engine": "lucene",
                "parameters": field_config['parameters']
            }
        }

    return properties

# Example usage
vector_config = {
    "expected_docs": 1_000_000,
    "vector_fields": {
        "content_vector": {
            "dimension": 384,
            "algorithm": "hnsw",
            "space_type": "cosinesimil",
            "parameters": {"ef_construction": 256, "m": 24}
        }
    }
}

create_optimized_vector_index("my_documents", vector_config)
```

---

## Document Indexing

### Single Document Indexing

```python
def index_document_with_vector(index_name, doc_id, title, content, vector, metadata=None):
    """Index a single document with vector"""

    document = {
        "title": title,
        "content": content,
        "content_vector": vector,
        "timestamp": datetime.utcnow().isoformat(),
        "doc_length": len(content),
        **(metadata or {})
    }

    try:
        response = client.index(
            index=index_name,
            id=doc_id,
            body=document,
            refresh=False  # Don't force immediate refresh
        )
        return response['result'] == 'created'
    except Exception as e:
        print(f"Error indexing document {doc_id}: {e}")
        return False
```

### Optimized Bulk Indexing

```python
def bulk_index_documents(index_name, documents, batch_size=1000):
    """Efficiently bulk index documents with vectors"""

    total_docs = len(documents)
    successful = 0

    for i in range(0, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        bulk_body = []

        for doc in batch:
            # Index action
            action = {
                "index": {
                    "_index": index_name,
                    "_id": doc.get("id", f"doc_{i}_{len(bulk_body)//2}")
                }
            }

            # Document body
            doc_body = {
                "title": doc["title"],
                "content": doc["content"],
                "content_vector": doc["vector"],
                "timestamp": doc.get("timestamp", datetime.utcnow().isoformat()),
                "category": doc.get("category", "general"),
                "url": doc.get("url", ""),
                "doc_length": len(doc["content"])
            }

            bulk_body.extend([action, doc_body])

        try:
            response = client.bulk(
                body=bulk_body,
                refresh=False,  # Batch refresh later
                timeout='60s'
            )

            # Count successful operations
            for item in response['items']:
                if 'index' in item and item['index']['status'] in [200, 201]:
                    successful += 1

        except Exception as e:
            print(f"Error in bulk indexing batch {i//batch_size}: {e}")

    # Force refresh after bulk operations
    client.indices.refresh(index=index_name)

    print(f"Successfully indexed {successful}/{total_docs} documents")
    return successful

# Example usage
documents = [
    {
        "id": "doc_1",
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence...",
        "vector": generate_embedding("Introduction to Machine Learning..."),
        "category": "education",
        "url": "https://example.com/ml-intro"
    },
    # ... more documents
]

bulk_index_documents("my_documents", documents)
```

---

## Basic Vector Search

### Simple k-NN Search

```python
def simple_vector_search(index_name, query_vector, k=10):
    """Perform basic vector similarity search"""

    search_body = {
        "size": k,
        "query": {
            "knn": {
                "content_vector": {
                    "vector": query_vector,
                    "k": k
                }
            }
        },
        "_source": ["title", "content", "category", "timestamp"]
    }

    response = client.search(index=index_name, body=search_body)
    return format_search_results(response)

def format_search_results(response):
    """Format OpenSearch response for easy consumption"""
    results = []

    for hit in response["hits"]["hits"]:
        result = {
            "id": hit["_id"],
            "score": hit["_score"],
            "title": hit["_source"].get("title", ""),
            "content": hit["_source"].get("content", "")[:200] + "...",
            "category": hit["_source"].get("category", ""),
            "timestamp": hit["_source"].get("timestamp", "")
        }
        results.append(result)

    return {
        "total_hits": response["hits"]["total"]["value"],
        "max_score": response["hits"]["max_score"],
        "results": results,
        "took_ms": response["took"]
    }

# Example usage
query_embedding = generate_embedding("machine learning algorithms")
results = simple_vector_search("my_documents", query_embedding, k=5)

for result in results["results"]:
    print(f"Score: {result['score']:.4f} - {result['title']}")
```

### Advanced Vector Search with Parameters

```python
def advanced_vector_search(index_name, query_vector, k=10, ef_search=None, algorithm_params=None):
    """Vector search with algorithm-specific parameter tuning"""

    knn_query = {
        "content_vector": {
            "vector": query_vector,
            "k": k
        }
    }

    # Add algorithm-specific parameters
    if ef_search:  # [HNSW](glossary.md#hnsw-hierarchical-navigable-small-world) parameter
        knn_query["content_vector"]["ef_search"] = ef_search

    if algorithm_params:  # Additional algorithm parameters
        knn_query["content_vector"].update(algorithm_params)

    search_body = {
        "size": k,
        "query": {"knn": knn_query},
        "_source": ["title", "content", "category", "timestamp"],
        "explain": False  # Set to True for debugging relevance scores
    }

    response = client.search(index=index_name, body=search_body)
    return format_search_results(response)

# Examples of parameter tuning
# High accuracy search (slower)
high_accuracy_results = advanced_vector_search(
    "my_documents",
    query_vector,
    k=10,
    ef_search=200
)

# Balanced search
balanced_results = advanced_vector_search(
    "my_documents",
    query_vector,
    k=10,
    ef_search=100
)

# Fast search (lower accuracy)
fast_results = advanced_vector_search(
    "my_documents",
    query_vector,
    k=10,
    ef_search=50
)
```

---

## Advanced Search Techniques

### Text Search Implementation

```python
def text_search(index_name, query_text, k=10):
    """Perform traditional text search"""

    search_body = {
        "size": k,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": ["title^2", "content"],
                "type": "best_fields"
            }
        }
    }

    response = client.search(index=index_name, body=search_body)
    return format_search_results(response)

def vector_search(index_name, query_vector, k=10):
    """Perform pure vector search"""
    return simple_vector_search(index_name, query_vector, k)
```

### Rank Fusion for Combined Results

```python
def rank_fusion_search(index_name, query_text, query_vector, k=10):
    """Use reciprocal rank fusion to combine text and vector search results"""

    # Perform separate text and vector searches
    text_results = text_search(index_name, query_text, k=k*2)
    vector_results = vector_search(index_name, query_vector, k=k*2)

    # Apply reciprocal rank fusion
    fused_scores = reciprocal_rank_fusion(
        text_results["results"],
        vector_results["results"],
        k_constant=60
    )

    # Get final results
    final_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    return format_fusion_results(final_results, index_name)

def reciprocal_rank_fusion(text_results, vector_results, k_constant=60):
    """Apply reciprocal rank fusion algorithm"""

    doc_scores = {}

    # Add text search scores
    for rank, result in enumerate(text_results, 1):
        doc_id = result["id"]
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k_constant + rank)

    # Add vector search scores
    for rank, result in enumerate(vector_results, 1):
        doc_id = result["id"]
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k_constant + rank)

    return doc_scores
```

---

## Hybrid Search Implementation

### Basic Hybrid Search

```python
def hybrid_search(index_name, text_query, query_vector, k=10, text_boost=1.0, vector_boost=1.0):
    """Combine text and vector search using OpenSearch hybrid query"""

    search_body = {
        "size": k,
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "match": {
                            "content": {
                                "query": text_query,
                                "boost": text_boost
                            }
                        }
                    },
                    {
                        "knn": {
                            "content_vector": {
                                "vector": query_vector,
                                "k": k,
                                "boost": vector_boost
                            }
                        }
                    }
                ]
            }
        }
    }

    response = client.search(index=index_name, body=search_body)
    return format_search_results(response)

# Example usage
results = hybrid_search(
    "my_documents",
    text_query="machine learning algorithms",
    query_vector=generate_embedding("machine learning algorithms"),
    k=10,
    text_boost=1.2,  # Slightly favor text matches
    vector_boost=1.0
)
```

### Adaptive Hybrid Search

```python
def adaptive_hybrid_search(index_name, query_text, k=10):
    """Automatically adjust hybrid search strategy based on query characteristics"""

    # Generate query vector
    query_vector = generate_embedding(query_text)

    # Analyze query characteristics
    query_analysis = analyze_query(query_text)

    # Adjust search strategy based on analysis
    if query_analysis["is_factual"]:
        # Factual queries benefit from text search
        text_boost, vector_boost = 2.0, 0.8
    elif query_analysis["is_conceptual"]:
        # Conceptual queries benefit from vector search
        text_boost, vector_boost = 0.8, 2.0
    elif query_analysis["has_specific_terms"]:
        # Queries with specific terms benefit from balanced approach
        text_boost, vector_boost = 1.5, 1.0
    else:
        # Default balanced approach
        text_boost, vector_boost = 1.0, 1.0

    return hybrid_search(index_name, query_text, query_vector, k, text_boost, vector_boost)

def analyze_query(query_text):
    """Analyze query characteristics to inform search strategy"""

    # Simple heuristics (can be replaced with ML models)
    lower_query = query_text.lower()

    factual_indicators = ["what is", "how to", "when did", "where is", "who", "definition"]
    conceptual_indicators = ["similar to", "like", "about", "related to", "concepts"]
    specific_terms = ["api", "function", "class", "method", "error", "code"]

    analysis = {
        "is_factual": any(indicator in lower_query for indicator in factual_indicators),
        "is_conceptual": any(indicator in lower_query for indicator in conceptual_indicators),
        "has_specific_terms": any(term in lower_query for term in specific_terms),
        "query_length": len(query_text.split()),
        "has_quotes": '"' in query_text
    }

    return analysis

# Example usage
results = adaptive_hybrid_search("my_documents", "what is machine learning?", k=10)
```

---

## Multi-Modal Search

### Multi-Modal Index Setup

```python
def setup_multimodal_index():
    """Create index supporting multiple content modalities"""

    mapping = {
        "settings": {
            "index": {"knn": True}
        },
        "mappings": {
            "properties": {
                # Content metadata
                "title": {"type": "text"},
                "description": {"type": "text"},
                "content_type": {"type": "keyword"},  # "text", "image", "audio", "video"
                "source_url": {"type": "keyword"},
                "timestamp": {"type": "date"},

                # Raw content storage
                "text_content": {"type": "text"},
                "image_url": {"type": "keyword", "index": False},
                "image_metadata": {
                    "properties": {
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                        "format": {"type": "keyword"},
                        "file_size": {"type": "long"}
                    }
                },

                # Unified embedding for cross-modal search
                "unified_vector": {
                    "type": "knn_vector",
                    "dimension": 512,  # Shared embedding dimension
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "parameters": {"ef_construction": 256, "m": 32}
                    }
                },

                # Modality-specific vectors for within-modal search
                "text_vector": {
                    "type": "knn_vector",
                    "dimension": 384,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "parameters": {"ef_construction": 128, "m": 24}
                    }
                },

                "image_vector": {
                    "type": "knn_vector",
                    "dimension": 512,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "parameters": {"ef_construction": 256, "m": 32}
                    }
                }
            }
        }
    }

    client.indices.create(index="multimodal_content", body=mapping)
```

### Cross-Modal Search Functions

```python
def text_to_image_search(text_query, k=10, similarity_threshold=0.7):
    """Find images using text descriptions"""

    # Generate text embedding using multimodal model
    text_embedding = generate_multimodal_embedding(text_query, modality="text")

    search_body = {
        "size": k,
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "unified_vector": {
                                "vector": text_embedding,
                                "k": k * 2  # Get extra results for filtering
                            }
                        }
                    }
                ],
                "filter": [
                    {"term": {"content_type": "image"}}
                ]
            }
        },
        "min_score": similarity_threshold  # Filter low-similarity results
    }

    response = client.search(index="multimodal_content", body=search_body)
    return format_multimodal_results(response)

# Example usage
results = text_to_image_search("sunset over mountains", k=5)
for result in results["results"]:
    print(f"Score: {result['score']:.3f} - {result['title']}")
    print(f"Image: {result['image_url']}")

def image_to_text_search(image_path, k=10):
    """Find text content similar to an image"""

    # Generate image embedding
    image_embedding = generate_multimodal_embedding(image_path, modality="image")

    search_body = {
        "size": k,
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "unified_vector": {
                                "vector": image_embedding,
                                "k": k * 2
                            }
                        }
                    }
                ],
                "filter": [
                    {"term": {"content_type": "text"}}
                ]
            }
        }
    }

    response = client.search(index="multimodal_content", body=search_body)
    return format_multimodal_results(response)

def universal_multimodal_search(query_content, query_type, target_types=None, k=10):
    """Search across all modalities with any input type"""

    # Generate embedding based on input type
    if query_type == "text":
        query_embedding = generate_multimodal_embedding(query_content, "text")
    elif query_type == "image":
        query_embedding = generate_multimodal_embedding(query_content, "image")
    elif query_type == "audio":
        query_embedding = generate_multimodal_embedding(query_content, "audio")
    else:
        raise ValueError(f"Unsupported query type: {query_type}")

    # Build search query
    search_query = {
        "knn": {
            "unified_vector": {
                "vector": query_embedding,
                "k": k * 2
            }
        }
    }

    # Filter by target content types if specified
    if target_types:
        search_query = {
            "bool": {
                "must": [search_query],
                "filter": [
                    {"terms": {"content_type": target_types}}
                ]
            }
        }

    search_body = {"size": k, "query": search_query}
    response = client.search(index="multimodal_content", body=search_body)

    return format_multimodal_results(response)

# Examples
# Find any content similar to an image
all_results = universal_multimodal_search("path/to/image.jpg", "image", k=10)

# Find only text and video content similar to audio
text_video_results = universal_multimodal_search(
    "path/to/audio.mp3",
    "audio",
    target_types=["text", "video"],
    k=10
)
```

### Modality-Specific Boosting

```python
def boosted_multimodal_search(query_embedding, query_type, k=10):
    """Apply different boosting strategies based on query modality"""

    # Define boost weights based on typical cross-modal similarities
    boost_matrix = {
        "text": {"text": 2.0, "image": 1.0, "audio": 0.8, "video": 1.2},
        "image": {"text": 1.0, "image": 2.0, "audio": 0.6, "video": 1.8},
        "audio": {"text": 0.8, "image": 0.6, "audio": 2.0, "video": 1.5}
    }

    should_queries = []
    for target_type, boost in boost_matrix[query_type].items():
        should_queries.append({
            "bool": {
                "must": [
                    {
                        "knn": {
                            "unified_vector": {
                                "vector": query_embedding,
                                "k": k
                            }
                        }
                    }
                ],
                "filter": [{"term": {"content_type": target_type}}],
                "boost": boost
            }
        })

    search_body = {
        "size": k,
        "query": {
            "bool": {
                "should": should_queries,
                "minimum_should_match": 1
            }
        }
    }

    return client.search(index="multimodal_content", body=search_body)
```

---

## Recommendation Systems

### User Profile Vector Creation

```python
def create_user_profile_vector(user_id, interaction_history, decay_factor=0.95):
    """Build user preference vector from interaction history"""

    content_vectors = []
    weights = []
    current_time = datetime.now()

    for interaction in interaction_history:
        # Get content vector
        content_doc = client.get(
            index="content_library",
            id=interaction["content_id"]
        )
        content_vector = content_doc["_source"]["content_vector"]

        # Calculate weight based on interaction type and recency
        interaction_weights = {
            "view": 1.0,
            "like": 3.0,
            "share": 5.0,
            "bookmark": 4.0,
            "download": 6.0,
            "purchase": 10.0
        }

        base_weight = interaction_weights.get(interaction["type"], 1.0)

        # Apply temporal decay
        days_ago = (current_time - interaction["timestamp"]).days
        temporal_weight = decay_factor ** days_ago

        final_weight = base_weight * temporal_weight

        content_vectors.append(content_vector)
        weights.append(final_weight)

    # Calculate weighted average
    if not content_vectors:
        return None

    user_vector = weighted_vector_average(content_vectors, weights)
    return normalize_vector(user_vector)

def weighted_vector_average(vectors, weights):
    """Calculate weighted average of vectors"""
    total_weight = sum(weights)
    if total_weight == 0:
        return [0] * len(vectors[0])

    result_vector = [0] * len(vectors[0])
    for vector, weight in zip(vectors, weights):
        for i, value in enumerate(vector):
            result_vector[i] += value * weight

    return [value / total_weight for value in result_vector]
```

### Personalized Recommendations

```python
def generate_personalized_recommendations(user_vector, exclude_content=None,
                                        diversity_factor=0.3, k=20):
    """Generate recommendations balancing relevance and diversity"""

    # Base similarity search
    search_body = {
        "size": k * 3,  # Get extra results for diversity filtering
        "query": {
            "knn": {
                "content_vector": {
                    "vector": user_vector,
                    "k": k * 3
                }
            }
        }
    }

    # Exclude already seen content
    if exclude_content:
        search_body["query"] = {
            "bool": {
                "must": [search_body["query"]],
                "must_not": [
                    {"terms": {"_id": exclude_content}}
                ]
            }
        }

    response = client.search(index="content_library", body=search_body)
    candidates = response["hits"]["hits"]

    # Apply diversity and business logic filtering
    recommendations = diversified_selection(
        candidates,
        diversity_factor=diversity_factor,
        final_count=k
    )

    return format_recommendations(recommendations)

def diversified_selection(candidates, diversity_factor=0.3, final_count=20):
    """Select diverse recommendations using maximal marginal relevance"""

    if len(candidates) <= final_count:
        return candidates

    selected = []
    remaining = list(candidates)

    # Always select the top result first
    selected.append(remaining.pop(0))

    # Select remaining items balancing relevance and diversity
    while len(selected) < final_count and remaining:
        best_score = -1
        best_idx = 0

        for i, candidate in enumerate(remaining):
            # Relevance score (similarity to user vector)
            relevance = candidate["_score"]

            # Diversity score (distance from already selected items)
            if selected:
                similarities = []
                candidate_vector = candidate["_source"]["content_vector"]

                for selected_item in selected:
                    selected_vector = selected_item["_source"]["content_vector"]
                    sim = cosine_similarity(candidate_vector, selected_vector)
                    similarities.append(sim)

                diversity = 1.0 - max(similarities)  # Distance from closest selected
            else:
                diversity = 1.0

            # Combined score
            combined_score = (1 - diversity_factor) * relevance + diversity_factor * diversity

            if combined_score > best_score:
                best_score = combined_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected
```

### Collaborative Filtering

```python
def collaborative_vector_recommendations(target_user_id, user_vectors_index, k=20):
    """Generate recommendations based on similar users' preferences"""

    # Get target user's profile vector
    target_user_doc = client.get(
        index=user_vectors_index,
        id=target_user_id
    )
    target_vector = target_user_doc["_source"]["user_vector"]

    # Find similar users
    similar_users = client.search(
        index=user_vectors_index,
        body={
            "size": 50,  # Consider top 50 similar users
            "query": {
                "knn": {
                    "user_vector": {
                        "vector": target_vector,
                        "k": 50
                    }
                }
            }
        }
    )

    # Aggregate content preferences from similar users
    content_scores = {}

    for user_hit in similar_users["hits"]["hits"]:
        user_similarity = user_hit["_score"]
        user_preferences = user_hit["_source"]["recent_interactions"]

        for content_id, interaction_strength in user_preferences.items():
            if content_id not in content_scores:
                content_scores[content_id] = 0

            # Weight by user similarity and interaction strength
            content_scores[content_id] += user_similarity * interaction_strength

    # Get top content recommendations
    top_content_ids = sorted(
        content_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:k]

    return format_collaborative_recommendations(top_content_ids)
```

### Hybrid Recommendation Systems

```python
def hybrid_recommendations(user_id, content_based_weight=0.6,
                          collaborative_weight=0.3, popularity_weight=0.1, k=20):
    """Combine multiple recommendation signals"""

    # Get different types of recommendations
    content_based = generate_content_based_recs(user_id, k=k*2)
    collaborative = generate_collaborative_recs(user_id, k=k*2)
    popularity_based = get_trending_content(k=k//2)

    # Combine scores using weighted fusion
    combined_scores = {}

    # Content-based recommendations
    for i, rec in enumerate(content_based):
        content_id = rec["content_id"]
        # Decay rank-based score
        score = content_based_weight * (1.0 / (i + 1))
        combined_scores[content_id] = combined_scores.get(content_id, 0) + score

    # Collaborative recommendations
    for i, rec in enumerate(collaborative):
        content_id = rec["content_id"]
        score = collaborative_weight * (1.0 / (i + 1))
        combined_scores[content_id] = combined_scores.get(content_id, 0) + score

    # Popularity boost
    for i, rec in enumerate(popularity_based):
        content_id = rec["content_id"]
        score = popularity_weight * (1.0 / (i + 1))
        combined_scores[content_id] = combined_scores.get(content_id, 0) + score

    # Sort and return top k
    final_recommendations = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:k]

    return enrich_recommendations(final_recommendations)
```

---

## Real-Time Applications

### Real-Time Indexing System

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue

class RealTimeVectorIndexer:
    def __init__(self, index_name, max_workers=4, batch_size=100):
        self.index_name = index_name
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.document_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = False

    async def start_indexing(self):
        """Start background indexing process"""
        self.running = True

        # Start batch processing coroutines
        tasks = []
        for _ in range(self.max_workers):
            task = asyncio.create_task(self.process_document_batches())
            tasks.append(task)

        await asyncio.gather(*tasks)

    async def process_document_batches(self):
        """Process documents in batches for efficiency"""
        batch = []

        while self.running:
            try:
                # Collect batch
                while len(batch) < self.batch_size:
                    try:
                        doc = self.document_queue.get(timeout=1.0)
                        batch.append(doc)
                    except queue.Empty:
                        break

                if batch:
                    await self.index_batch(batch)
                    batch.clear()
                else:
                    await asyncio.sleep(0.1)  # Brief pause if no documents

            except Exception as e:
                print(f"Error processing batch: {e}")
                await asyncio.sleep(1.0)

    async def index_batch(self, documents):
        """Index a batch of documents"""
        loop = asyncio.get_event_loop()

        # Run indexing in thread pool to avoid blocking
        await loop.run_in_executor(
            self.executor,
            self._sync_index_batch,
            documents
        )

    def _sync_index_batch(self, documents):
        """Synchronous batch indexing"""
        bulk_body = []

        for doc in documents:
            action = {"index": {"_index": self.index_name, "_id": doc["id"]}}
            bulk_body.extend([action, doc["content"]])

        if bulk_body:
            client.bulk(body=bulk_body, refresh=True)

    def add_document(self, document):
        """Add document to indexing queue"""
        self.document_queue.put(document)

    def stop(self):
        """Stop indexing process"""
        self.running = False

# Usage example
indexer = RealTimeVectorIndexer("live_content")

# Start indexing in background
asyncio.create_task(indexer.start_indexing())

# Add documents as they arrive
new_document = {
    "id": "doc_123",
    "content": {
        "title": "Breaking News",
        "content": "Latest developments in AI...",
        "content_vector": generate_embedding("Latest developments in AI..."),
        "timestamp": datetime.now().isoformat()
    }
}

indexer.add_document(new_document)
```

### High-Performance Search API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from typing import List, Optional

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    query_type: str = "hybrid"  # "text", "vector", "hybrid"
    k: int = 10
    filters: Optional[dict] = None
    ef_search: Optional[int] = None

class SearchResult(BaseModel):
    id: str
    title: str
    content: str
    score: float
    timestamp: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_hits: int
    took_ms: int
    query_info: dict

class HighPerformanceSearchAPI:
    def __init__(self, opensearch_client):
        self.client = opensearch_client
        self.embedding_cache = {}

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Main search endpoint with async processing"""
        start_time = asyncio.get_event_loop().time()

        try:
            if request.query_type == "vector":
                results = await self.vector_search(request)
            elif request.query_type == "text":
                results = await self.text_search(request)
            else:  # hybrid
                results = await self.hybrid_search(request)

            end_time = asyncio.get_event_loop().time()
            took_ms = int((end_time - start_time) * 1000)

            return SearchResponse(
                results=results["results"],
                total_hits=results["total_hits"],
                took_ms=took_ms,
                query_info={
                    "query_type": request.query_type,
                    "processed_query": request.query,
                    "k": request.k
                }
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def vector_search(self, request: SearchRequest):
        """Async vector search with caching"""
        # Get or generate embedding
        query_vector = await self.get_embedding_cached(request.query)

        search_body = self.build_vector_query(
            query_vector,
            request.k,
            request.filters,
            request.ef_search
        )

        # Execute search in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.search(index="live_content", body=search_body)
        )

        return self.format_results(response)

    async def get_embedding_cached(self, text: str):
        """Get embedding with LRU cache"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        # Generate embedding asynchronously
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            generate_embedding,
            text
        )

        # Cache with size limit
        if len(self.embedding_cache) > 1000:
            # Remove oldest entry
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]

        self.embedding_cache[text] = embedding
        return embedding

# FastAPI endpoints
search_api = HighPerformanceSearchAPI(client)

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    return await search_api.search(request)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# WebSocket for real-time search
@app.websocket("/ws/search")
async def websocket_search(websocket):
    await websocket.accept()

    try:
        while True:
            # Receive search request
            data = await websocket.receive_json()
            request = SearchRequest(**data)

            # Process search
            response = await search_api.search(request)

            # Send results
            await websocket.send_json(response.dict())

    except Exception as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()
```

### Performance Monitoring

```python
import time
from collections import defaultdict, deque
import statistics

class SearchPerformanceMonitor:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.query_types = defaultdict(lambda: deque(maxlen=window_size))
        self.error_counts = defaultdict(int)
        self.total_queries = 0

    def record_query(self, query_type: str, latency_ms: float, success: bool):
        """Record query performance metrics"""
        self.total_queries += 1

        if success:
            self.latencies.append(latency_ms)
            self.query_types[query_type].append(latency_ms)
        else:
            self.error_counts[query_type] += 1

    def get_performance_stats(self):
        """Get current performance statistics"""
        if not self.latencies:
            return {"status": "no_data"}

        overall_stats = {
            "total_queries": self.total_queries,
            "avg_latency_ms": statistics.mean(self.latencies),
            "p95_latency_ms": statistics.quantiles(self.latencies, n=20)[18],  # 95th percentile
            "p99_latency_ms": statistics.quantiles(self.latencies, n=100)[98],  # 99th percentile
            "error_rate": sum(self.error_counts.values()) / self.total_queries
        }

        # Per-query-type stats
        type_stats = {}
        for query_type, latencies in self.query_types.items():
            if latencies:
                type_stats[query_type] = {
                    "count": len(latencies),
                    "avg_latency_ms": statistics.mean(latencies),
                    "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
                }

        return {
            "overall": overall_stats,
            "by_type": type_stats,
            "errors": dict(self.error_counts)
        }

# Integration with search API
monitor = SearchPerformanceMonitor()

@app.middleware("http")
async def performance_middleware(request, call_next):
    if request.url.path.startswith("/search"):
        start_time = time.time()

        try:
            response = await call_next(request)
            latency_ms = (time.time() - start_time) * 1000

            # Extract query type from request or response
            query_type = "unknown"  # Extract from actual request

            monitor.record_query(query_type, latency_ms, True)
            return response

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            monitor.record_query("error", latency_ms, False)
            raise
    else:
        return await call_next(request)

@app.get("/metrics")
async def get_metrics():
    return monitor.get_performance_stats()
```

---

This collection provides comprehensive Python implementation examples for building production-ready vector search systems with OpenSearch.