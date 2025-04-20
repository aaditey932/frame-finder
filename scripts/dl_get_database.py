import os
import time
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional, Tuple


def initialize_pinecone(api_key: str) -> Pinecone:
    """
    Initialize a Pinecone client with the provided API key.
    
    Args:
        api_key: Pinecone API key
        
    Returns:
        Initialized Pinecone client
    """
    return Pinecone(api_key=api_key)


def create_pinecone_index(
    pc: Pinecone, 
    index_name: str, 
    dimension: int = 512, 
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1"
) -> Any:
    """
    Create a Pinecone index if it doesn't exist.
    
    Args:
        pc: Pinecone client
        index_name: Name of the index to create
        dimension: Dimension of the vectors
        metric: Distance metric to use
        cloud: Cloud provider
        region: Cloud region
        
    Returns:
        Pinecone index object
    """
    # Check if index already exists
    if not pc.has_index(index_name):
        print(f"Creating index: {index_name} with dimension {dimension}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            )
        )
        
        # Wait for the index to be ready
        print("Waiting for index to be ready...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
            
    else:
        print(f"Index {index_name} already exists")
        
    return pc.Index(index_name)


def prepare_vectors_from_dataframe(
    df: pd.DataFrame, 
    embedding_col: str = "image_embedding",
    id_col: str = "file",
    metadata_cols: List[str] = ["title", "artist", "style", "genre"]
) -> List[Dict[str, Any]]:
    """
    Prepare vectors for Pinecone upsert from a DataFrame.
    
    Args:
        df: DataFrame containing vectors and metadata
        embedding_col: Column name for the embedding vectors
        id_col: Column name to use as vector ID
        metadata_cols: Column names to include as metadata
        
    Returns:
        List of dictionaries formatted for Pinecone upsert
    """
    vectors = []
    
    for _, row in df.iterrows():
        # Sanitize filename for use as ID (Pinecone IDs must be ASCII)
        sanitized_id = str(row[id_col]).encode('ascii', errors='ignore').decode('ascii')
        
        # Create metadata dictionary
        metadata = {col: row[col] for col in metadata_cols if col in row}
        
        # Create vector dictionary
        vector_dict = {
            "id": sanitized_id,
            "values": row[embedding_col],
            "metadata": metadata
        }
        
        vectors.append(vector_dict)
        
    return vectors


def upsert_to_pinecone(
    index: Any, 
    vectors: List[Dict[str, Any]], 
    namespace: str = "ns1",
    batch_size: int = 100
) -> int:
    """
    Upsert vectors to Pinecone index in batches.
    
    Args:
        index: Pinecone index
        vectors: List of vector dictionaries
        namespace: Namespace to use
        batch_size: Size of batches for upserting
        
    Returns:
        Number of vectors upserted
    """
    # Upsert in batches to avoid timeouts with large datasets
    total_vectors = len(vectors)
    for i in range(0, total_vectors, batch_size):
        batch = vectors[i:min(i+batch_size, total_vectors)]
        index.upsert(
            vectors=batch,
            namespace=namespace
        )
        print(f"Upserted batch {i//batch_size + 1}/{(total_vectors-1)//batch_size + 1} ({len(batch)} vectors)")
    
    return total_vectors


def query_image(
    image_embedding: List[float],
    index: Any,
    namespace: str = "ns1",
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Query Pinecone index with an image embedding.
    
    Args:
        image_embedding: Image embedding vector
        index: Pinecone index
        namespace: Namespace to query
        top_k: Number of top matches to return
        
    Returns:
        Query response with matches
    """
    return index.query(
        namespace=namespace,
        vector=image_embedding,
        top_k=top_k,
        include_metadata=True
    )


def format_query_results(query_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Format query results for display.
    
    Args:
        query_response: Response from Pinecone query
        
    Returns:
        List of formatted results
    """
    results = []
    
    for match in query_response.get("matches", []):
        result = {
            "id": match.get("id", ""),
            "score": match.get("score", 0),
            "metadata": match.get("metadata", {})
        }
        results.append(result)
        
    return results


def main():
    """Main function to demonstrate functionality."""
    # Configuration
    PINECONE_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
    INDEX_NAME = "frame-finder-database"
    EMBEDDINGS_PATH = "./data/wikiart_export/embeddings_data.csv"
    
    # Initialize Pinecone
    pc = initialize_pinecone(PINECONE_API_KEY)
    
    # Load DataFrame with embeddings
    try:
        df = pd.read_csv(EMBEDDINGS_PATH)
        print(f"Loaded {len(df)} embeddings from {EMBEDDINGS_PATH}")
        
        # Check if DataFrame has embeddings column
        if "image_embedding" not in df.columns:
            print("Error: DataFrame does not contain 'image_embedding' column")
            return
            
    except FileNotFoundError:
        print(f"Error: Embeddings file not found: {EMBEDDINGS_PATH}")
        return
    
    # Create index
    index = create_pinecone_index(pc, INDEX_NAME)
    
    # Prepare vectors
    vectors = prepare_vectors_from_dataframe(df)
    
    # Upsert vectors
    total_upserted = upsert_to_pinecone(index, vectors)
    print(f"✅ Upserted {total_upserted} vectors to Pinecone")
    
    # Show index statistics
    stats = index.describe_index_stats()
    print(f"Index statistics: {stats}")


if __name__ == "__main__":
    main()