########################################
# STEP 1: Imports and Dependencies
########################################
import os
import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

########################################
# STEP 2: Device Setup for CUDA
########################################
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    gpu_id = "0"  # set to whichever GPU device ID is appropriate
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Using CUDA device: {gpu_id}")
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current CUDA Device:", torch.cuda.current_device())
elif device == "cpu":
    print("Using CPU for computation.")
else:
    print("Error: Unknown device:", device)
    exit()

########################################
# STEP 3: Load Sentence Transformer Model
########################################
def load_model_on_device(model_name_or_path):
    """
    Load a SentenceTransformer model on the specified device.
    """
    embedding_model = SentenceTransformer(model_name_or_path, device=device)
    print(f"Loaded embedding model on device: {embedding_model.device}")
    return embedding_model

# Example model: "all-MiniLM-L6-v2"
embedding_model_path = "all-MiniLM-L6-v2"
embedding_model = load_model_on_device(embedding_model_path)

########################################
# STEP 4: Initialize FAISS Index
########################################
def initialize_faiss_index(dimension, use_gpu=True):
    """
    Initialize a FAISS index with the specified dimension using
    Inner Product similarity (IndexFlatIP). We'll normalize
    embeddings to approximate cosine similarity.
    """
    index = faiss.IndexFlatIP(dimension)
    
    if use_gpu and torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        print("FAISS index created on GPU")
    else:
        print("FAISS index created on CPU")
    
    return index

# Determine embedding dimension by encoding a sample
sample_text = "Sample text for dimension"
sample_embedding = embedding_model.encode([sample_text])
embedding_dim = len(sample_embedding[0])

# Create the FAISS index
index = initialize_faiss_index(embedding_dim, use_gpu=(device == "cuda"))

########################################
# STEP 5: Load and Process Data
########################################
# Replace "dataset.csv" with your actual dataset path
df = pd.read_csv("dataset.csv")

# Drop any rows with NaN
df = df.dropna(subset=["text", "label"])

# Add character and word length columns
df["char_len"] = df["text"].apply(len)
df["word_len"] = df["text"].apply(lambda x: len(x.split()))

print("Data Loaded Successfully.")
print(df.head())

########################################
# STEP 6: Document Chunking
########################################
def do_chunking(text_list, chunk_size=200, overlap=10):
    """
    Splits each text in text_list into overlapping chunks.
    Returns a list of chunk strings and corresponding metadata.
    """
    chunks = []
    chunk_metadata = []
    
    for text in text_list:
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            chunk_metadata.append({
                "char_len": len(chunk),
                "word_len": len(chunk.split()),
                "pg_num": (i // (chunk_size - overlap)) + 1,
                "first_pg": (i == 0)
            })
    return chunks, chunk_metadata

# Apply chunking per row in the dataframe
df["chunks"], df["chunk_metadata"] = zip(*df["text"].apply(lambda x: do_chunking([x])))

print("Chunking completed successfully.")

########################################
# STEP 7: Create Embeddings
########################################
def vectorize_text(text_list, model):
    """
    Convert list of text into embeddings using a SentenceTransformer model.
    FAISS requires normalized vectors for cosine similarity.
    """
    embeddings = model.encode(text_list, show_progress_bar=True)
    # Normalize each vector in-place
    faiss.normalize_L2(embeddings)
    return embeddings

# Gather all chunks and metadata into lists for indexing
all_chunks = []
all_metadata = []
for i, row in df.iterrows():
    # Extend the chunk list
    all_chunks.extend(row["chunks"])
    
    # Merge each chunk's metadata with the row's label
    for meta in row["chunk_metadata"]:
        meta_with_label = {
            **meta,
            "label": row["label"]
        }
        all_metadata.append(meta_with_label)

# Create embeddings for all chunks
all_embeddings = vectorize_text(all_chunks, embedding_model)
print("Embeddings generated successfully.")

########################################
# STEP 8: Store Data in the FAISS Index
########################################
index.add(all_embeddings)
print(f"Added {len(all_embeddings)} vectors to FAISS index")

########################################
# STEP 9: Basic FAISS Search
########################################
def search_in_db(query_text, index, model, k=3):
    """
    Perform similarity search in FAISS using a given model for encoding.
    Returns top-k documents, metadata, and similarity scores.
    """
    # Encode and normalize the query
    query_embedding = model.encode([query_text])
    faiss.normalize_L2(query_embedding)
    
    # Search in FAISS
    distances, faiss_indices = index.search(query_embedding, k)
    
    # Prepare results
    # distances is shape (1, k), faiss_indices is shape (1, k)
    distances = distances[0]
    faiss_indices = faiss_indices[0]
    
    results = []
    for dist, idx in zip(distances, faiss_indices):
        # Convert inner product to a [0..1] similarity score
        similarity_score = (dist + 1.0) / 2.0
        result = {
            "document": all_chunks[idx],
            "metadata": all_metadata[idx],
            "score": similarity_score
        }
        results.append(result)
        
    return results

# Example query
example_query = "What is the White House address?"
search_results = search_in_db(example_query, index, embedding_model, k=3)

for i, r in enumerate(search_results):
    print(f"Match {i+1}: {r['document']}")
    print(f"Metadata: {r['metadata']}")
    print(f"Similarity Score: {r['score']:.4f}\n")

########################################
# STEP 10: Train/Test Split for Classification
########################################
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

true_labels = test_df["label"].tolist()
pred_labels = []

for text in test_df["text"]:
    # We'll do a top-1 retrieval and predict the label of the closest chunk
    result = search_in_db(text, index, embedding_model, k=1)
    predicted_label = result[0]["metadata"]["label"]
    pred_labels.append(predicted_label)

########################################
# STEP 11: (Optional) First Page Detection
########################################
# If needed, you can iterate through metadata and check `metadata["first_pg"]`.
# For now, we leave this step as in the ChromaDB flow.

########################################
# STEP 12: Performance Metrics
########################################
print("Classification Performance:")
print(classification_report(true_labels, pred_labels))

accuracy_val = accuracy_score(true_labels, pred_labels)
f1_val = f1_score(true_labels, pred_labels, average="weighted")

print("*** Overall Performance ***")
print(f"Accuracy: {accuracy_val:.4f}")
print(f"F1 Score (weighted): {f1_val:.4f}")

########################################
# STEP 13: Advanced Search (Optional)
########################################
def advanced_search(query_text, index, model, k=5):
    """
    Demonstrates how you could do an 'advanced' search:
      - Possibly rerank results
      - Filter by metadata
      - Combine multiple query embeddings
      - etc.
    """
    results = search_in_db(query_text, index, model, k=k)

    # Example: Sort results by highest similarity first
    # (they may already come in descending order, but we can re-check)
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)

    return results_sorted

# Example advanced search usage
example_advanced_query = "Detailed question about a specific topic"
advanced_results = advanced_search(example_advanced_query, index, embedding_model, k=5)

for i, r in enumerate(advanced_results):
    print(f"Advanced Match {i+1}: {r['document']}")
    print(f"Metadata: {r['metadata']}")
    print(f"Similarity Score: {r['score']:.4f}\n")


########################################
# STEP 11: First Page Detection
########################################

def detect_first_page_chunks(df):
    """
    Given the dataframe (df) where each row has:
      - df["chunks"] (a list of text chunks)
      - df["chunk_metadata"] (a list of dicts, each containing 'first_pg': bool)
    Return a list of all first-page chunks along with relevant information.
    """
    first_page_docs = []
    
    for idx, row in df.iterrows():
        chunk_list = row["chunks"]
        metadata_list = row["chunk_metadata"]
        
        for chunk_idx, (chunk_text, meta) in enumerate(zip(chunk_list, metadata_list)):
            if meta["first_pg"]:
                # This chunk is flagged as the first page
                first_page_docs.append({
                    "doc_index": idx,       # the row index in df
                    "chunk_index": chunk_idx,
                    "chunk_text": chunk_text,
                    "metadata": meta,
                    "label": row["label"]
                })
                
    return first_page_docs

# Example usage:
first_pages = detect_first_page_chunks(df)

# Print out a few examples
for fp in first_pages[:5]:  # just show first 5 for demonstration
    print("=== First Page Chunk ===")
    print("Document Index:", fp["doc_index"])
    print("Chunk Index:", fp["chunk_index"])
    print("Label:", fp["label"])
    print("Metadata:", fp["metadata"])
    print("Chunk Text:", fp["chunk_text"])
    print("------------------------")
