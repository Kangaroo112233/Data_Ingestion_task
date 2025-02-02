#### INSTALL REQUIRED LIBRARIES (if not already installed)
# pip install torch numpy pandas faiss-cpu sentence-transformers scikit-learn tqdm

import os
import time
import warnings
import random
import subprocess

import numpy as np
import pandas as pd
import faiss

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Set global environment & random seeds
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
random.seed(0)
np.random.seed(0)
torch_manual_seed = 0
torch_backend_deterministic = True  # (For reproducibility, if using torch.cuda)
# ------------------------------------------------------------

# ------------------------------
# Helper Functions
# ------------------------------
def check_GPU():
    """Return the uptime of the current process (as a proxy for GPU process info)."""
    import psutil
    MEM_TOT = 40960  # Example: total GPU memory in MB (if needed)
    def get_process_uptime(pid):
        p = psutil.Process(pid)
        create_time = p.create_time()
        current_time = time.time()
        uptime_seconds = current_time - create_time
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{int(hours)} hours, {int(minutes)} minutes"
    try:
        return get_process_uptime(os.getpid())
    except psutil.NoSuchProcess:
        return "Process not found"

def train_test_df_split(df, train_sample_fraction=0.5):
    """Perform a simple train/test split on the dataframe indices."""
    trn_idx, tst_idx = train_test_split(df.index, train_size=train_sample_fraction, random_state=0)
    traindf = df.loc[trn_idx]
    testdf = df.loc[tst_idx]
    return traindf, testdf

# ------------------------------
# Embedding Model Initialization
# ------------------------------
def load_model_on_device(embedding_model_name, device='cuda', gpu_id='0'):
    """
    Load a SentenceTransformer model on the specified device.
    If using CUDA, the model will be placed on device:<gpu_id>.
    """
    if device == 'cuda' and torch.cuda.is_available():
        device_with_id = f"{device}:{gpu_id}"
    else:
        device_with_id = device
    model = SentenceTransformer(embedding_model_name, device=device_with_id)
    return model

# ------------------------------
# FAISS Vector Database Functions
# ------------------------------
def initialize_faiss(vector_dim):
    """
    Initialize a FAISS index (using L2 distance) with the given embedding dimension.
    """
    index = faiss.IndexFlatL2(vector_dim)
    return index

# ------------------------------
# Text Processing and Chunking
# ------------------------------
def process_doc(txt_list):
    """
    Placeholder for document pre-processing.
    Currently returns the list unchanged.
    """
    return txt_list

def do_chunking(txt_list, CHUNK_SIZE=200, OVERLAP=0):
    """
    Splits each text document into chunks of CHUNK_SIZE words with the specified OVERLAP.
    """
    def chunk_document_with_overlap(text):
        words = text.split()
        chunks = []
        step = CHUNK_SIZE - OVERLAP if (CHUNK_SIZE - OVERLAP) > 0 else CHUNK_SIZE
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            if chunk:  # only add non-empty chunks
                chunks.append(chunk)
        return chunks

    lolo_chunks = [chunk_document_with_overlap(text) for text in txt_list]
    return lolo_chunks

def make_metadata(txt_list, lolo_chunks):
    """
    Create simple metadata for each chunk.
    (This example simply stores the chunk text as metadata.)
    """
    return [[{"text": chunk} for chunk in chunks] for chunks in lolo_chunks]

# ------------------------------
# FAISS Storage & Search Functions
# ------------------------------
def store_in_faiss(lolo_chunks, embedding_model, faiss_index):
    """
    Encode all text chunks using the embedding model and add them to the FAISS index.
    Also maintain a parallel list (text_store) that holds the original text for each embedding.
    """
    text_store = []
    for chunk_list in lolo_chunks:
        # Generate embeddings (ensure float32 for FAISS)
        embeddings = embedding_model.encode(chunk_list, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        faiss_index.add(embeddings)
        text_store.extend(chunk_list)
    return text_store

def search_in_faiss(lolo_chunks, embedding_model, faiss_index, text_store, k=1):
    """
    For each chunk in each document, compute its embedding and query the FAISS index.
    Returns a list (per document) of lists (per chunk) of the top-k retrieved texts.
    """
    lolo_results = []
    for doc_chunk_list in lolo_chunks:
        doc_results = []
        embeddings = embedding_model.encode(doc_chunk_list, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        distances, indices = faiss_index.search(embeddings, k)
        for idx_list in indices:
            retrieved_texts = [text_store[i] for i in idx_list]
            doc_results.append(retrieved_texts)
        lolo_results.append(doc_results)
    return lolo_results

def print_search_results(lolo_results):
    """
    Print the search results in a readable format.
    """
    print("\n*** Search Results:")
    for doc_idx, doc_results in enumerate(lolo_results):
        print(f"\nDocument {doc_idx}:")
        for chunk_idx, res in enumerate(doc_results):
            print(f"  Chunk {chunk_idx}: {res}")
    print("-----\n")

# ------------------------------
# Performance Evaluation Function
# ------------------------------
def generate_results(true_labels, pred_labels):
    """
    Generate and print a classification report comparing true and predicted labels.
    (Note: In a real use-case, you would map FAISS search results to label predictions.)
    """
    print("\nClassification Report:\n", classification_report(true_labels, pred_labels))

# ------------------------------
# Implementation V1: Extraction Probability using FAISS
# ------------------------------
def v1_extraction():
    """
    V1: Load and clean dataset, perform a train-test split, embed the training documents,
    store the embeddings in FAISS, then search on test documents.
    """
    # Load dataset
    data_fn = "hlntext_274K.csv"  # Adjust path as needed
    df = pd.read_csv(f"dataset/{data_fn}", encoding="utf-8", engine="python")
    df = df.dropna()
    df["char_len"] = df.text.apply(lambda x: len(x))
    df["word_len"] = df.text.apply(lambda x: len(x.split()))
    
    # Split into training and testing sets
    traindf, testdf = train_test_df_split(df, train_sample_fraction=0.5)
    
    # Initialize embedding model and FAISS index
    embedding_model = load_model_on_device("all-MiniLM-L6-v2", device="cuda", gpu_id="0")
    vector_dim = embedding_model.get_sentence_embedding_dimension()
    faiss_index = initialize_faiss(vector_dim)
    
    # Process training documents
    documents_train = traindf.text.tolist()
    processed_train = process_doc(documents_train)
    lolo_chunks_train = do_chunking(processed_train, CHUNK_SIZE=200, OVERLAP=0)
    text_store = store_in_faiss(lolo_chunks_train, embedding_model, faiss_index)
    
    # Process testing documents and perform search
    documents_test = testdf.text.tolist()
    processed_test = process_doc(documents_test)
    lolo_chunks_test = do_chunking(processed_test, CHUNK_SIZE=200, OVERLAP=0)
    lolo_results = search_in_faiss(lolo_chunks_test, embedding_model, faiss_index, text_store, k=1)
    
    # For demonstration, print the search results.
    # (In practice, you would compare predicted labels from the retrieved texts to ground truth.)
    print_search_results(lolo_results)

# ------------------------------
# Implementation V2: Extraction Probability using FAISS
# ------------------------------
def v2_extraction():
    """
    V2: Similar to V1, but demonstrates a slightly different processing (if desired).
    In this example, we reload the data and repeat the store/search process.
    """
    # Load dataset
    data_fn = "hlntext_274K.csv"  # Adjust path as needed
    df = pd.read_csv(f"dataset/{data_fn}", encoding="utf-8", engine="python")
    df = df.dropna()
    df["char_len"] = df.text.apply(lambda x: len(x))
    df["word_len"] = df.text.apply(lambda x: len(x.split()))
    traindf, testdf = train_test_df_split(df, train_sample_fraction=0.5)
    
    # Initialize embedding model and FAISS index
    embedding_model = load_model_on_device("all-MiniLM-L6-v2", device="cuda", gpu_id="0")
    vector_dim = embedding_model.get_sentence_embedding_dimension()
    faiss_index = initialize_faiss(vector_dim)
    
    # Process and store training documents
    documents_train = traindf.text.tolist()
    processed_train = process_doc(documents_train)
    lolo_chunks_train = do_chunking(processed_train, CHUNK_SIZE=200, OVERLAP=0)
    text_store = store_in_faiss(lolo_chunks_train, embedding_model, faiss_index)
    
    # Process test documents and search
    documents_test = testdf.text.tolist()
    processed_test = process_doc(documents_test)
    lolo_chunks_test = do_chunking(processed_test, CHUNK_SIZE=200, OVERLAP=0)
    lolo_results = search_in_faiss(lolo_chunks_test, embedding_model, faiss_index, text_store, k=1)
    
    # Print search results
    print_search_results(lolo_results)

# ------------------------------
# Main Execution Block
# ------------------------------
if __name__ == "__main__":
    print("Executing V1 Extraction using FAISS...\n")
    v1_extraction()
    
    print("\nExecuting V2 Extraction using FAISS...\n")
    v2_extraction()
