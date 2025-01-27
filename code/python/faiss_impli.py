import os
import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import time
import warnings
import re
import psutil
from pathlib import Path

# Constants
CHUNK_SIZE = 20
OVERLAP = 0
VERBOSE_ON = False
MEM_TOT = 40960  # Total memory in MB

@dataclass
class SearchResult:
    document: str
    distance: float
    metadata: Dict
    similarity: float

def check_gpu():
    """Check GPU availability and memory"""
    pat = r'/envs/(.*?)/bin/python'
    out1 = subprocess.check_output(['nvidia-smi']).decode()
    out = out1.split('\n')
    procs = [x for x in out if x.endswith('MiB |')]
    
    # Print GPU memory info
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB
    available_memory = psutil.virtual_memory().available / (1024 ** 3)
    used_memory = psutil.virtual_memory().used / (1024 ** 3)
    memory_percent = psutil.virtual_memory().percent
    
    print(f"Total Memory    : {total_memory:.2f} GB")
    print(f"Available Memory: {available_memory:.2f} GB")
    print(f"Used Memory     : {used_memory:.2f} GB")
    print(f"Memory Usage    : {memory_percent}%")
    
    return procs

def get_process_uptime(pid):
    """Get process uptime in hours:minutes format"""
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
    """Format memory usage information"""
    mem_used = int(mem_used_txt)
    mem_free = MEM_TOT - mem_used
    
    mem_used_f = f"{mem_used:,}"  # thousands format with comma
    mem_free_f = f"{mem_free:,}"  # thousands format with comma
    
    mem_used_msg = f"{MEM_TOT:,}MB :{2.0f}%".format(mem_used_f, mem_used*100/MEM_TOT)
    mem_free_msg = f"{MEM_TOT:,}MB :{2.0f}%".format(mem_free_f, mem_free*100/MEM_TOT)
    
    return mem_used_msg, mem_free_msg

class DocumentProcessor:
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_doc(self, txt_list: List[str]) -> List[str]:
        """Basic document preprocessing"""
        if VERBOSE_ON:
            print('***process_doc()')
            word_cnt_l = [len(txt.split()) for txt in txt_list]
            stats = self._calculate_stats(word_cnt_l)
            print(f"    avg_word_cnt:{int(stats['mean'])}  "
                  f"max_word_cnt:{stats['max']}  "
                  f"min_word_cnt:{stats['min']}")
        return txt_list

    def keep_lines(self, txt_list: List[str], num_lines: int = 4) -> List[str]:
        """Keep first and last n lines of documents"""
        return [' '.join(doc.splitlines(keepends=True)[:num_lines] + 
                        doc.splitlines(keepends=True)[-num_lines:]) 
                for doc in txt_list]

    def chunk_document(self, text: str) -> List[str]:
        """Chunk text into pieces with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
        return chunks

    def get_chunks(self, txt_list: List[str]) -> List[List[str]]:
        """Get chunks for all documents with statistics"""
        lolo_chunks = [self.chunk_document(txt) for txt in txt_list]
        
        if VERBOSE_ON:
            self._print_chunking_stats(lolo_chunks)
            
        return lolo_chunks

    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics"""
        return {
            'mean': np.mean(values),
            'max': max(values),
            'min': min(values)
        }

    def _print_chunking_stats(self, lolo_chunks: List[List[str]]):
        """Print chunking statistics"""
        print('***do_chunking()')
        
        # Chunks per document stats
        chunks_per_doc = [len(chunk_list) for chunk_list in lolo_chunks]
        chunk_stats = self._calculate_stats(chunks_per_doc)
        print(' *Number of chunks per doc:')
        print(f"    avg_chunk_cnt:{int(chunk_stats['mean'])}  "
              f"max_chunk_cnt:{chunk_stats['max']}  "
              f"min_chunk_cnt:{chunk_stats['min']}")
        
        # Words per chunk stats
        flat_chunks = [x for xs in lolo_chunks for x in xs]
        words_per_chunk = [len(chunk.split()) for chunk in flat_chunks]
        word_stats = self._calculate_stats(words_per_chunk)
        print(' *Number of words per chunk:')
        print(f"    avg_word_cnt:{int(word_stats['mean'])}  "
              f"max_word_cnt:{word_stats['max']}  "
              f"min_word_cnt:{word_stats['min']}")

class FAISSVectorDB:
    def __init__(self,
                 embedding_dimension: int,
                 gpu_id: str = '0',
                 similarity_algo: str = 'cosine'):
        """Initialize FAISS vector database
        Args:
            embedding_dimension: Dimension of embeddings
            gpu_id: GPU device ID
            similarity_algo: 'cosine' or 'l2'
        """
        self.dimension = embedding_dimension
        self.gpu_id = gpu_id
        self.similarity_algo = similarity_algo
        self.index = None
        self.metadata_store = []
        self.document_store = []
        self.initialize_index()

    def initialize_index(self):
        """Initialize FAISS index based on similarity algorithm"""
        if self.similarity_algo == 'cosine':
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
        else:
            self.index = faiss.IndexFlatL2(self.dimension)  # L2 distance
            
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, int(self.gpu_id), self.index)

    def add_documents(self, 
                     documents: List[str],
                     embeddings: np.ndarray,
                     metadata: List[Dict] = None):
        """Add documents with their embeddings and metadata"""
        if metadata is None:
            metadata = [{} for _ in documents]
            
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
            
        self.index.add(embeddings.astype(np.float32))
        self.document_store.extend(documents)
        self.metadata_store.extend(metadata)

    def search(self, 
              query_embedding: np.ndarray,
              k: int = 1,
              return_embeddings: bool = False) -> List[Dict]:
        """Search for similar documents
        Args:
            query_embedding: Query vector
            k: Number of results to return
            return_embeddings: Whether to return embeddings in results
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for dist_array, idx_array in zip(distances, indices):
            result = {
                'distances': dist_array.tolist(),
                'documents': [self.document_store[idx] for idx in idx_array],
                'metadatas': [self.metadata_store[idx] for idx in idx_array],
            }
            if return_embeddings:
                result['embeddings'] = [self.get_embedding(idx) for idx in idx_array]
            results.append(result)
            
        return results

    def get_embedding(self, idx: int) -> np.ndarray:
        """Get embedding vector for document at index"""
        if not 0 <= idx < len(self.document_store):
            raise ValueError(f"Index {idx} out of range")
        return faiss.vector_to_array(self.index.reconstruct(idx))

class DocumentClassifier:
    def __init__(self,
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 device: str = 'cuda',
                 gpu_id: str = '0'):
        """Initialize document classifier
        Args:
            embedding_model_name: Name of the sentence transformer model
            device: 'cuda' or 'cpu'
            gpu_id: GPU device ID
        """
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        self.doc_processor = DocumentProcessor()
        self.vectordb = FAISSVectorDB(
            embedding_dimension=self.embedding_model.get_sentence_embedding_dimension(),
            gpu_id=gpu_id
        )

    def prepare_data(self, df: pd.DataFrame, count: int = 100) -> pd.DataFrame:
        """Create balanced dataset with 'count' samples per label"""
        return pd.concat([df[df.label == label].sample(count) 
                         for label in df.label.unique()])

    def process_and_index(self,
                         documents: List[str],
                         metadata: List[Dict] = None,
                         batch_size: int = 32):
        """Process documents and add to index"""
        # Process documents
        processed_docs = self.doc_processor.process_doc(documents)
        chunks = self.doc_processor.get_chunks(processed_docs)
        
        # Flatten chunks and maintain mapping
        flat_chunks = []
        chunk_metadata = []
        
        for doc_idx, (doc_chunks, doc_meta) in enumerate(zip(chunks, metadata or [{}] * len(chunks))):
            for chunk_idx, chunk in enumerate(doc_chunks):
                flat_chunks.append(chunk)
                chunk_metadata.append({
                    'doc_idx': doc_idx,
                    'chunk_idx': chunk_idx,
                    **doc_meta
                })

        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(flat_chunks), batch_size):
            batch = flat_chunks[i:i + batch_size]
            embeddings = self.embedding_model.encode(batch, convert_to_tensor=True)
            all_embeddings.append(embeddings.cpu().numpy())

        # Add to vector database
        embeddings = np.vstack(all_embeddings)
        self.vectordb.add_documents(
            documents=flat_chunks,
            embeddings=embeddings,
            metadata=chunk_metadata
        )

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        results = self.vectordb.search(query_embedding, k=k)
        
        if VERBOSE_ON:
            self._print_search_results(results)
            
        return results

    def batch_search(self, queries: List[str], k: int = 5) -> List[List[Dict]]:
        """Perform batch search for multiple queries"""
        query_embeddings = self.embedding_model.encode(queries, convert_to_numpy=True)
        all_results = []
        
        for query_embedding in query_embeddings:
            results = self.vectordb.search(query_embedding, k=k)
            all_results.append(results)
            
        return all_results

    def _print_search_results(self, results: List[Dict]):
        """Print detailed search results"""
        print('\n***Search results:')
        print(f'Found {len(results)} matches')
        
        for i, result in enumerate(results):
            print(f'\nResult {i+1}:')
            for j, (dist, doc, meta) in enumerate(zip(
                result['distances'], 
                result['documents'], 
                result['metadatas']
            )):
                similarity = 1 - dist if self.vectordb.similarity_algo == 'l2' else dist
                print(f'\nMatch {j+1}:')
                print(f'Similarity: {similarity:.2%}')
                print(f'Distance: {dist:.4f}')
                print(f'Metadata: {meta}')
                print(f'Document excerpt: {doc[:200]}...')

def main():
    # Configuration
    embedding_model_name = 'all-MiniLM-L6-v2'
    db_name = 'document_classification'
    gpu_id = '3'
    device = 'cuda'
    
    # Initialize classifier
    classifier = DocumentClassifier(
        embedding_model_name=embedding_model_name,
        device=device,
        gpu_id=gpu_id
    )
    
    # Load and process data
    print('\n***Loading data into df and cleaning:')
    df = pd.read_csv('dataset/data_fn.csv', encoding='utf-8', engine='python')
    df = df.dropna()
    
    # Sample for training
    tdf = classifier.prepare_data(df, count=100)
    txt_list = tdf.text.tolist()
    
    # Process documents
    txt_list = classifier.doc_processor.keep_lines(txt_list, num_lines=4)
    chunks = classifier.doc_processor.get_chunks(txt_list)
    
    # Create metadata
    metadata = [{
        'label': row.label,
        'doc_id': idx,
        'first_pg': row.get('first_pg', False),
        'is_split': row.get('is_split', False)
    } for idx, row in tdf.iterrows()]
    
    # Index documents
    classifier.process_and_index(txt_list, metadata=metadata)
    
    #
