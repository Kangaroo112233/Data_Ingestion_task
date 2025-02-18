import os
import torch
import numpy as np
import pandas as pd
from typing import List

# ----------------------------------------
# 1) Chroma DB & SentenceTransformer
# ----------------------------------------
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ----------------------------------------
# 2) Local LLM loader (LLAMA 3.1 8B)
# ----------------------------------------
from packages.model_utils.loading import LLM

# =========================================================
# A) Final Classification Prompts
# =========================================================

DOC_CLASSIFIER_SYSTEM_PROMPT = """
You are a document classification assistant. Identify the document type of the following page.
1) Bank Statement
2) Paystub
3) other

A bank statement is a financial record that provides a detailed summary of all transactions, including deposits, withdrawals, and fees over a specific period.
A paystub is a summary of an employee’s earnings and deductions from their paycheck.

If it does not fit the above categories, classify it as 'other'.
"""

SPLIT_CLASSIFIER_SYSTEM_PROMPT = """
You are a text classification assistant. Identify whether the following page is the first page of a document or not.
Return 'True' or 'False'.
"""

CLASSIFIER_SYSTEM_PROMPT = """
You are a document classification model. Based on the document text given at the end:
[ 'Bank Statement', 'Paystub', 'W2', 'other' ]

Return the combined output with label and first page. Only respond with the exact result.
Keep your answer short and concise.
"""

# =========================================================
# B) Utility Functions (Truncation, Train/Test Split)
# =========================================================

def truncate_doc_text(doc_text: str, top_n: int = 4, bottom_n: int = 4) -> str:
    """
    Keeps only the top N lines and bottom N lines of a document's text.
    If the document is shorter than top_n + bottom_n lines, returns it unchanged.
    """
    lines = doc_text.splitlines()
    if len(lines) <= (top_n + bottom_n):
        return doc_text  # no truncation needed
    truncated_lines = lines[:top_n] + lines[-bottom_n:]
    return "\n".join(truncated_lines)

# Example train_test_split using scikit-learn
from sklearn.model_selection import train_test_split

def train_test_df_split(df: pd.DataFrame, train_fraction: float = 0.8, random_state: int = 42):
    """
    Splits a DataFrame into train/test sets by a given fraction.
    Returns (train_df, test_df).
    """
    train_df, test_df = train_test_split(df, test_size=1 - train_fraction, random_state=random_state)
    return train_df, test_df

# =========================================================
# C) Chroma DB Setup & Chunking
# =========================================================

def load_model_on_device(embedding_model_name: str, device='cuda', gpu_id='0'):
    """
    Loads a SentenceTransformer embedding model on the specified device.
    Adjust path as needed for your environment.
    """
    model_path = f"/phoenix/lib/models/{embedding_model_name}"
    if not os.path.isdir(model_path):
        raise ValueError(f"Embedding model path does not exist: {model_path}")

    # Set GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    if device == "cuda" and torch.cuda.is_available():
        device_with_id = f"cuda:{gpu_id}"
    else:
        device_with_id = "cpu"

    embedding_model = SentenceTransformer(model_path, device=device_with_id)
    embedding_model.to(device_with_id)
    print(f"Loaded embedding model '{embedding_model_name}' on device: {device_with_id}")
    return embedding_model

class MyEmbeddingFunction:
    """
    Custom embedding function to bind with Chroma DB.
    """
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model

    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

def initialize_ChromaDB(embedding_model_name: str,
                        db_name="my_collection",
                        similarity_algo="cosine",
                        custom_embedding=None):
    """
    Create a new Chroma DB collection with the specified similarity measure.
    If 'custom_embedding' is provided, it will be used for the collection's embedding function.
    """
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                      persist_directory="chromadb_data"))
    try:
        client.delete_collection(name=db_name)
    except:
        print("No existing collection found; creating a new one.")

    if custom_embedding:
        collection = client.create_collection(name=db_name,
                                              metadata={"hnsw:space": similarity_algo},
                                              embedding_function=custom_embedding)
    else:
        collection = client.create_collection(name=db_name,
                                              metadata={"hnsw:space": similarity_algo})
    print(f"Chroma DB Collection '{db_name}' created with similarity '{similarity_algo}'.")
    return collection, client

def chunk_document_with_overlap(text: str, chunk_size=200, overlap=0) -> List[str]:
    """
    Splits a single document's text into chunks of 'chunk_size' words with optional overlap.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap if chunk_size > overlap else chunk_size
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def store_in_db(chunks_list: List[List[str]],
                collection,
                metadata_list: List[List[dict]] = None):
    """
    Stores chunked text into the Chroma DB collection.
    If the collection has a custom embedding function, just add documents + metadata.
    """
    count_before = collection.count()
    for doc_idx, doc_chunks in enumerate(chunks_list):
        for chunk_idx, chunk_text in enumerate(doc_chunks):
            _id = f"doc{doc_idx}_chunk{chunk_idx}"
            _metadata = {}
            if metadata_list:
                _metadata = metadata_list[doc_idx][chunk_idx]
            collection.add(
                documents=[chunk_text],
                metadatas=[_metadata],
                ids=[_id]
            )
    count_after = collection.count()
    print(f"Stored {count_after - count_before} new chunks in Chroma DB.")

# =========================================================
# D) RAG Retrieval & Classification
# =========================================================

def rag_retrieve(query: str, collection, k: int = 3) -> str:
    """
    Retrieves top-k relevant chunks from Chroma DB for the given query text.
    Returns a single string containing the retrieved chunks.
    """
    results = collection.query(query_texts=[query], n_results=k)
    retrieved_docs = results.get("documents", [[]])[0]
    if not retrieved_docs:
        print("No relevant documents found in Chroma DB.")
        return ""
    context = "\n".join(retrieved_docs)
    return context

def doc_classification_rag(
    user_query: str,
    collection,
    llm,
    system_prompt: str = DOC_CLASSIFIER_SYSTEM_PROMPT,
    k: int = 3,
    max_length: int = 256
) -> str:
    """
    1) Retrieves context from Chroma DB.
    2) Inserts that context + user_query into the doc-classifier prompt.
    3) Generates a classification label (e.g., 'Bank Statement', 'Paystub', or 'other').
    """
    context = rag_retrieve(user_query, collection, k=k)
    final_prompt = (
        f"{system_prompt.strip()}\n\n"
        f"Document Text:\n{context}\n\n"
        f"Question: {user_query}\nAnswer:"
    )
    print("\n=== Document Classification Prompt ===\n", final_prompt, "\n======================================\n")

    answer = llm.generate_text(prompt=final_prompt, max_length=max_length)
    return answer

def first_page_classification_rag(
    user_query: str,
    collection,
    llm,
    system_prompt: str = SPLIT_CLASSIFIER_SYSTEM_PROMPT,
    k: int = 3,
    max_length: int = 128
) -> str:
    """
    1) Retrieves context from Chroma DB.
    2) Inserts that context + user_query into the "is first page" prompt.
    3) Generates 'True' or 'False'.
    """
    context = rag_retrieve(user_query, collection, k=k)
    final_prompt = (
        f"{system_prompt.strip()}\n\n"
        f"Page Text:\n{context}\n\n"
        f"Question: {user_query}\nAnswer:"
    )
    print("\n=== First-Page Classification Prompt ===\n", final_prompt, "\n=========================================\n")

    answer = llm.generate_text(prompt=final_prompt, max_length=max_length)
    return answer

def combined_classification_rag(
    user_query: str,
    collection,
    llm,
    system_prompt: str = CLASSIFIER_SYSTEM_PROMPT,
    k: int = 3,
    max_length: int = 256
) -> str:
    """
    Single call to do both doc-type classification and first-page detection in one response.
    """
    context = rag_retrieve(user_query, collection, k=k)
    final_prompt = (
        f"{system_prompt.strip()}\n\n"
        f"Document Text:\n{context}\n\n"
        f"Question: {user_query}\nAnswer:"
    )
    print("\n=== Combined Classification Prompt ===\n", final_prompt, "\n=======================================\n")

    answer = llm.generate_text(prompt=final_prompt, max_length=max_length)
    return answer

# =========================================================
# E) Main Script: Putting It All Together
# =========================================================

if __name__ == "__main__":
    # -----------------------------
    # 1) Load your CSV dataset
    # -----------------------------
    dataset_path = "/phoenix/workspaces/xtkvtbrj/Gen_AI/classification/dataset/dataset_splits_full.csv"
    df = pd.read_csv(dataset_path)
    print("Loaded dataset with shape:", df.shape)

    # Suppose your CSV has columns like: ["doc_content", "label", "is_first_pg"] (adjust if needed).
    # We'll create a truncated version of each doc:
    df["truncated_text"] = df["doc_content"].apply(lambda x: truncate_doc_text(str(x), top_n=4, bottom_n=4))

    # -----------------------------
    # 2) Train-Test Split
    # -----------------------------
    train_sample_fraction = 0.6
    traindf, testdf = train_test_df_split(df, train_fraction=train_sample_fraction, random_state=42)
    print("Train set size:", len(traindf), "Test set size:", len(testdf))
    print("Train label distribution:\n", traindf["label"].value_counts())
    print("Test label distribution:\n", testdf["label"].value_counts())

    # -----------------------------
    # 3) Initialize Chroma DB
    # -----------------------------
    embedding_model_name = "all-MiniLM-L6-v2"  # or your local embedding model name
    device = "cuda"
    gpu_id = "0"

    embedding_model = load_model_on_device(embedding_model_name, device=device, gpu_id=gpu_id)
    my_embedding_function = MyEmbeddingFunction(embedding_model)

    db_name = "my_doc_classification_db"
    similarity_algo = "cosine"
    collection, client = initialize_ChromaDB(embedding_model_name, db_name, similarity_algo, custom_embedding=my_embedding_function)

    # -----------------------------
    # 4) Chunk & Store Documents in Chroma
    #    (Here we store only the TRAIN docs, or both if desired)
    # -----------------------------
    train_texts = traindf["truncated_text"].tolist()
    chunked_docs = [chunk_document_with_overlap(txt, chunk_size=200, overlap=0) for txt in train_texts]

    # If you want to store metadata (like label/is_first_pg) for each chunk, create a matching structure.
    # We'll skip metadata here for simplicity:
    metadata_list = None

    store_in_db(chunked_docs, collection, metadata_list=metadata_list)

    # -----------------------------
    # 5) Load LLAMA 3.1 8B
    # -----------------------------
    llama_model_name = "Meta-llama-3.1-8B-Instruct"
    model_cfg = {
        "model_name": llama_model_name,
        "device": f"cuda:{gpu_id}",
        "model_type": "transformer_prompt",
        "temperature": 0.2,
        "top_p": 0.92,
        "repetition_penalty": 1.0,
        "use_cache": True
    }
    llm = LLM(**model_cfg)
    print("LLAMA model loaded on device:", llm.device)

    # -----------------------------
    # 6) Example RAG Classification Queries
    # -----------------------------
    # We'll pick a sample from the test set
    sample_test_doc = testdf.iloc[0]
    test_doc_text = sample_test_doc["truncated_text"]
    test_label = sample_test_doc["label"]
    test_is_first = sample_test_doc["is_first_pg"]

    print("\n--- Sample test doc text ---\n", test_doc_text)

    # (A) Document Classification
    user_query_doc = test_doc_text  # or a short query about the doc
    doc_class_label = doc_classification_rag(user_query_doc, collection, llm, k=3, max_length=256)
    print("** Document Classification Result:", doc_class_label)

    # (B) First-Page Classification
    user_query_first_page = test_doc_text
    first_page_result = first_page_classification_rag(user_query_first_page, collection, llm, k=3, max_length=128)
    print("** First-Page Detection Result:", first_page_result)

    # (C) Combined Classification
    user_query_combined = test_doc_text
    combined_result = combined_classification_rag(user_query_combined, collection, llm, k=3, max_length=256)
    print("** Combined Classification Result:", combined_result)
