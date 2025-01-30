# Import Libraries
import subprocess
import torch
import os
import numpy as np
import pandas as pd
import time
import warnings
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
import re
import subprocess
import faiss
from sentence_transformers import SentenceTransformer

# Suppress warnings
warnings.filterwarnings("ignore")

# Define constants and configurations
bs_t = '9200000' # 224250
w2_t = '9205000' # 14285
ps_t = '9204005' # 4467

tup_to_name = {'9200000': 'Bank Statement',
               '9205000': 'W2',
               '9204005': 'Paystub'}

# Set global variable to limit cuda memory split size
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

def check_gpu():
    """Check GPU memory and device information"""
    print("nbk lookup site: https://smartportal.bankofamerica.com/SSO/LookupID/Default.aspx")
    
    MEM_TOT = 40960
    def get_process_uptime(pid):
        try:
            p = psutil.Process(pid)
            create_time = p.create_time()
            current_time = time.time()
            uptime_seconds = current_time - create_time
            hours, remainder = divmod(uptime_seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            return f"{int(hours)} hours, {int(minutes)} minutes"
        except psutil.NoSuchProcess:
            return "Process not found"

    def format_mem(mem_used_txt):
        mem_used = int(mem_used_txt)
        mem_free = MEM_TOT - mem_used
        mem_used_f = f'{mem_used:,}'  # thousands format with comma
        mem_free_f = f'{mem_free:,}'  # thousands format with comma
        mem_used_msg = '{MB ({:.2f}%)'.format(mem_used_f, mem_used*100/MEM_TOT)
        mem_free_msg = '{MB ({:.2f}%)'.format(mem_free_f, mem_free*100/MEM_TOT)
        return mem_used_msg, mem_free_msg

class MyEmbeddingFunction:
    def __init__(self, embedding_model):
        """Store the SentenceTransformer model"""
        self.embedding_model = embedding_model
        
    def __call__(self, texts: Documents) -> Embeddings:
        """Use embedding model to generate embeddings"""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

def initialize_faiss(embedding_dimension, similarity_metric='cosine'):
    """Initialize FAISS index with specified similarity metric"""
    if similarity_metric == 'cosine':
        # Normalize vectors and use L2 for cosine similarity
        index = faiss.IndexFlatIP(embedding_dimension)  # Inner product for normalized vectors
    elif similarity_metric == 'l2':
        index = faiss.IndexFlatL2(embedding_dimension)
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
    return index

def process_doc(txt_list):
    """For any cleaning, preprocessing, special processing of text.
    input: list of text docs (text_list)
    output: list of text docs (txt_list)
    """
    if VERBOSE_ON:
        print('***process_doc()')
        print('    Number of words per document:')
    word_cnt_l = [len(txt.split()) for txt in txt_list]
    avg_word_cnt, max_word_cnt, min_word_cnt = np.mean(word_cnt_l), max(word_cnt_l), min(word_cnt_l)
    if VERBOSE_ON:
        print(f'    avg_word_cnt:{int(avg_word_cnt)}    max_word_cnt:{max_word_cnt}    min_word_cnt:{min_word_cnt}')
    
    return txt_list

def chunk_document_with_overlap(text):
    """Chunk text into pieces of 'CHUNK_SIZE' words with 'overlap' words between chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
        chunk = ' '.join(words[i:i + CHUNK_SIZE])
        chunks.append(chunk)
    
    return chunks

def do_chunking(txt_list):
    """Takes each text doc and chunks based on CHUNK_SIZE and OVERLAP"""
    lolo_chunks = [chunk_document_with_overlap(text) for text in txt_list]
    
    if VERBOSE_ON:
        print('***do_chunking()')
        print('    Number of chunks per doc:')
        chunks_per_doc_cnt_l = [len(chunk_list) for chunk_list in lolo_chunks]
        avg_chunk_cnt = np.mean(chunks_per_doc_cnt_l)
        print(f'    avg_chunk_cnt:{int(avg_chunk_cnt)}    max_chunk_cnt:{max(chunks_per_doc_cnt_l)}    min_chunk_cnt:{min(chunks_per_doc_cnt_l)}')
        
        print('    Number of words per chunk:')
        flat_chunk_list = [x for xs in lolo_chunks for x in xs]
        words_per_chunk_cnt_l = [len(chunk.split()) for chunk in flat_chunk_list]
        avg_word_cnt = np.mean(words_per_chunk_cnt_l)
        print(f'    avg_word_cnt:{int(avg_word_cnt)}    max_word_cnt:{max(words_per_chunk_cnt_l)}    min_word_cnt:{min(words_per_chunk_cnt_l)}')
    
    return lolo_chunks

def make_metadata_clf(txt_list, lolo_chunks, df=None):
    """Create metadata that needs to be stored alongside each chunk's vector in
    the vectordb. Need the txt_list and chunks to create metadata.
    Each chunk must have associated metadata.
    input: list of text docs (text_list), list of list of metadata (lolo_metadata)
    output: list of list of metadata (lolo_metadata)
    """
    if df is None:
        return None
        
    lolo_metadata = list()
    for chunk_list, (txt_idx, doc_info) in zip(lolo_chunks, df.iterrows()):
        metadata_list = list()
        for chunk in chunk_list:
            dd = {"label": doc_info['label'],
                  "first_pg": doc_info['first_pg'],
                  "is_split": doc_info['is_split'],
                  "pg_num": doc_info['pg_num'],
                  "st_pg": doc_info['st_pg'],
                  "en_pg": doc_info['en_pg'],
                 }
            metadata_list.append(dd)
        lolo_metadata.append(metadata_list)
        
    if VERBOSE_ON:
        print('***make_metadata_clf()')
        print('    Info on metadata:')
        print(f'len(lolo_metadata):', len(lolo_metadata))
        print('Total chunks for all docs:', sum([len(meta_list) for meta_list in lolo_metadata]))
    
    return lolo_metadata

def vectorize_text(lolo_chunks, embedding_model) -> list:
    """Return a list of vectors has embedding model bound"""
    lolo_vectors = [embedding_model.encode(chunk_list, show_progress_bar=False) for chunk_list in lolo_chunks]
    
    if VERBOSE_ON:
        print('***vectorize_text()')
        print('    Vector size:', embedding_model.get_sentence_embedding_dimension())
    
    return lolo_vectors

def store_in_faiss(lolo_chunks, lolo_vectors, lolo_metadata, faiss_index):
    """Store vectors and associated metadata (if any) in FAISS
    input: list of list of vectors (lolo_vectors),
           list of list of metadata (lolo_metadata),
           faiss_index
    output: Nothing
    """
    cnt_before_loading = faiss_index.ntotal
    
    # Flatten vectors and track metadata
    all_vectors = []
    all_metadata = []
    for doc_idx, (doc_chunk_list, doc_vectors, metadata_list) in enumerate(zip(lolo_chunks, lolo_vectors, lolo_metadata)):
        for chunk_idx, (chunk, vector, metadata) in enumerate(zip(doc_chunk_list, doc_vectors, metadata_list)):
            all_vectors.append(vector)
            all_metadata.append({
                'doc_idx': doc_idx,
                'chunk_idx': chunk_idx,
                'text': chunk,
                **metadata
            })
    
    # Convert to numpy array and add to index
    vectors_array = np.array(all_vectors).astype('float32')
    if vectors_array.shape[0] > 0:
        faiss_index.add(vectors_array)
    
    cnt_after_loading = faiss_index.ntotal
    if VERBOSE_ON:
        print('***store_in_faiss()')
        print(f'    {cnt_before_loading}:{cnt_after_loading}')
    
    return all_metadata

def search_in_faiss(search_vectors, faiss_index, metadata_list, k=1):
    """Perform similarity search
    input: search_vectors, faiss_index, k=1
    output: results_list
    """
    # Convert query vectors to numpy array
    query_array = np.array(search_vectors).astype('float32')
    
    # Search
    distances, indices = faiss_index.search(query_array, k)
    
    # Format results
    results = []
    for query_distances, query_indices in zip(distances, indices):
        query_results = []
        for dist, idx in zip(query_distances, query_indices):
            if idx != -1:  # FAISS returns -1 for not found
                query_results.append({
                    'distance': float(dist),
                    'metadata': metadata_list[idx]
                })
        results.append(query_results)
    
    return results

def print_search_results(lolo_results, k=1):
    """Print formatted search results"""
    print()
    print(f'***Results: Total search docs={len(lolo_results)}. Each search doc has k={k} results\\n')
    
    for idx, results in enumerate(lolo_results):
        print(f'*** Result for document: {idx}')
        for result in results:
            cos_sim = 1 - max(0, result['distance'])  # Convert L2 distance to cosine similarity
            print(f'distance: {result["distance"]:.6f}')
            print(f'similarity score: {cos_sim:.6f}')
            print(f'metadata: {result["metadata"]}')
            print('-' * 40)

# Main execution example
if __name__ == "__main__":
    # Configuration
    embedding_model_name = "all-MiniLM-L6-v2"
    similarity_algo = "cosine"
    device = "cuda"
    gpu_id = "3"
    VERBOSE_ON = True
    CHUNK_SIZE = 20
    OVERLAP = 0

    # Initialize embedding model
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    embedding_dimension = embedding_model.get_sentence_embedding_dimension()
    
    # Initialize FAISS index
    faiss_index = initialize_faiss(embedding_dimension, similarity_algo)
    
    # Example usage with sample data
    print("*** Loading and processing documents...")
    # Add your document loading and processing code here
    
    # Example search
    search_text = ["Sample search document"]
    search_vectors = embedding_model.encode(search_text)
    results = search_in_faiss(search_vectors, faiss_index, metadata_list=[], k=2)
    print_search_results(results)
