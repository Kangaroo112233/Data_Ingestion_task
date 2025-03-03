import subprocess
import torch
import os
import numpy as np
import pandas as pd
import time
import warnings
import logging
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
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Any, Tuple, Optional

# Set global variable to limit cuda memory split size
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# Set random seeds for reproducibility
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('document_classifier_rag')

warnings.filterwarnings("ignore")

# Constants
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Can be changed to other embedding models
LLM_MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"  # Using Llama 3.3 70B as requested
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = "0"  # Set appropriate GPU ID
CHUNK_SIZE = 250  # Default chunk size
OVERLAP = 50  # Default overlap size
TOP_K_RETRIEVAL = 5  # Number of chunks to retrieve from vector DB

class DocumentClassifierRAG:
    """
    RAG system specialized for document classification with ChromaDB
    - Classifies document types (Bank Statement, Paystub, W2, Other)
    - Detects if a page is the first page of a document
    - Extracts information from documents using RAG
    """
    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        llm_model_name: str = LLM_MODEL_NAME,
        device: str = DEVICE,
        gpu_id: str = GPU_ID,
        db_name: str = "document_classifier_db",
        similarity_algo: str = "cosine"
    ):
        """Initialize the RAG system with embedding model, LLM, and vector database."""
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.device = device
        self.gpu_id = gpu_id
        self.db_name = db_name
        self.similarity_algo = similarity_algo
        
        # Setup environment
        if device == 'cuda' and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            print(f"Using GPU: {gpu_id}")
            print(f'CUDA available: {torch.cuda.is_available()}')
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            self.device_with_id = f"{device}:{torch.cuda.current_device()}"
        else:
            self.device_with_id = "cpu"
            print("Using CPU")
            
        # Initialize models and database
        self._init_embedding_model()
        self._init_llm()
        self._init_vector_db()
        
        # Set default parameters
        self.chunk_size = CHUNK_SIZE
        self.overlap = OVERLAP
        self.top_k = TOP_K_RETRIEVAL
    
    def _init_embedding_model(self):
        """Initialize the embedding model."""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        try:
            # Try to use phoenix_util if available (for secure environments)
            from phoenix_util.model_util import getModel
            embedding_model_path = getModel(project_name="DCRS_LLM", model_name=self.embedding_model_name)()
            assert os.path.isdir(embedding_model_path)
            logger.info(f"Using Phoenix model path: {embedding_model_path}")
        except Exception as e:
            logger.warning(f"Phoenix model util not available or error: {e}")
            # Fallback to direct path
            embedding_model_path = f"/phoenix/lib/models/{self.embedding_model_name}"
            if not os.path.isdir(embedding_model_path):
                # Use the model name directly from HuggingFace if local path not found
                logger.info(f"Local path not found, using model name directly: {self.embedding_model_name}")
                embedding_model_path = self.embedding_model_name
        
        self.embedding_model = SentenceTransformer(embedding_model_path, device=self.device_with_id)
        self.embedding_model.to(self.device_with_id)
        logger.info(f"Embedding model loaded to device: {self.embedding_model.device}")
    
    def _init_llm(self):
        """Initialize the LLM model."""
        print(f"Loading LLM: {self.llm_model_name}")
        try:
            # Try to use phoenix_util if available
            from phoenix_util.model_util import getModel
            llm_model_path = getModel(project_name="DCRS_LLM", model_name=self.llm_model_name)()
            assert os.path.isdir(llm_model_path)
        except:
            # Fallback to direct path or HuggingFace
            llm_model_path = self.llm_model_name
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        
        # Load model with appropriate quantization/settings based on available resources
        if self.device == "cuda":
            # Use BF16 for better performance on newer GPUs
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # Load with int8 quantization for CPU
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                device_map="auto",
                load_in_8bit=True
            )
        
        # Create a text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
            max_length=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            device_map="auto"
        )
        
        print(f"LLM loaded successfully")
    
    def _init_vector_db(self):
        """Initialize the vector database."""
        print(f"Initializing ChromaDB")
        self.client = chromadb.Client()
        
        # Create a custom embedding function for ChromaDB
        self.embedding_function = self._create_embedding_function()
        
        # Try to delete existing collection if it exists
        try:
            self.client.delete_collection(self.db_name)
            print(f"Deleted existing collection: {self.db_name}")
        except:
            print(f"Creating new collection: {self.db_name}")
        
        # Create a new collection
        self.vector_db = self.client.create_collection(
            name=self.db_name,
            metadata={'hnsw:space': self.similarity_algo},
            embedding_function=self.embedding_function
        )
        
        print(f"Vector DB created: {self.vector_db.name}")
    
    def _create_embedding_function(self):
        """Create a custom embedding function for ChromaDB."""
        from chromadb import Documents, EmbeddingFunction, Embeddings
        
        class CustomEmbeddingFunction(EmbeddingFunction):
            def __init__(self, embedding_model):
                self.embedding_model = embedding_model
            
            def __call__(self, input: Documents) -> Embeddings:
                embeddings = self.embedding_model.encode(input, show_progress_bar=False)
                return embeddings.tolist()
        
        return CustomEmbeddingFunction(self.embedding_model)
    
    def chunk_document(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Chunk document into smaller pieces with overlap."""
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
            
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            # Break if we've processed all words
            if i + chunk_size >= len(words):
                break
                
        return chunks
    
    def index_documents(self, documents: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None):
        """Index documents into the vector database."""
        print(f"Indexing {len(documents)} documents")
        
        # Process one document at a time
        for idx, document in tqdm(enumerate(documents), total=len(documents)):
            # Chunk the document
            chunks = self.chunk_document(document)
            
            # Add metadata to each chunk if provided
            if metadata_list and idx < len(metadata_list):
                doc_metadata = metadata_list[idx]
            else:
                doc_metadata = {"document_id": str(idx)}
            
            # Add each chunk to the vector DB
            for chunk_idx, chunk in enumerate(chunks):
                # Create unique ID for each chunk
                chunk_id = f"doc_{idx}_chunk_{chunk_idx}"
                
                # Add metadata about the chunk position
                chunk_metadata = doc_metadata.copy()
                chunk_metadata.update({
                    "chunk_id": chunk_idx,
                    "document_id": str(idx),
                    "chunk_count": len(chunks)
                })
                
                # Store in vector DB
                self.vector_db.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    metadatas=[chunk_metadata]
                )
        
        print(f"Indexed {self.vector_db.count()} chunks from {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query."""
        if top_k is None:
            top_k = self.top_k
            
        # Query the vector DB
        results = self.vector_db.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format results
        retrieved_chunks = []
        for i in range(len(results['ids'][0])):
            retrieved_chunks.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
            
        return retrieved_chunks
    
    def generate_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]], task: str = "general") -> str:
        """Generate a prompt for the LLM using the retrieved chunks."""
        # Combine retrieved chunks into context
        context = "\n\n".join([f"[Document {i+1}]: {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)])
        
        if task == "doc_classification":
            # Document type classification prompt
            prompt = f"""
            You are a document classification assistant. Given the retrieved text below, classify the document type:
            1) Bank Statement
            2) Paystub
            3) W2
            4) Other

            A bank statement provides a detailed summary of transactions, deposits, and withdrawals over a specific period.
            A paystub details an employee's earnings and deductions.
            A W2 is a tax form showing an employee's annual wages and taxes withheld.

            Use only the retrieved context to make your classification. If the context does not contain enough information, classify as "Other."

            ### Retrieved Context ###
            {context}

            ### User Query ###
            {query}

            ### Answer (One of: Bank Statement, Paystub, W2, Other) ###
            """
        
        elif task == "first_page_detection":
            # First page detection prompt
            prompt = f"""
            You are a text classification assistant. Based on the retrieved document content, determine if this is the first page of a document.

            Indicators of a first page include:
            - Presence of a title or header
            - Introduction or overview text
            - Absence of "continued" or page numbers indicating mid-document content

            Use only the retrieved text to make your decision.

            ### Retrieved Context ###
            {context}

            ### User Query ###
            {query}

            ### Answer (True/False) ###
            """
            
        elif task == "combined_classification":
            # Combined document type and first page classification
            prompt = f"""
            You are a document classification assistant. Given the retrieved document content, determine both:
            1) The document type (Bank Statement, Paystub, W2, or Other).
            2) Whether this is the first page of the document (True/False).

            Use only the retrieved text to make your decision.

            ### Retrieved Context ###
            {context}

            ### User Query ###
            {query}

            ### Answer Format (Document Type: First Page Status) ###
            Example: Bank Statement:True
            """
            
        else:
            # General RAG query prompt
            prompt = f"""
            You are an advanced AI assistant. Please answer the following question based on the provided context.

            Context:
            {context}

            Question: {query}

            Answer:
            """
        
        return prompt
    
    def query(self, query: str, top_k: int = None, task: str = "general") -> str:
        """End-to-end RAG query pipeline."""
        # 1. Retrieve relevant chunks
        retrieved_chunks = self.retrieve(query, top_k)
        
        # 2. Generate prompt with context based on task
        prompt = self.generate_prompt(query, retrieved_chunks, task)
        
        # 3. Generate response with LLM
        response = self.generator(prompt, max_length=2048)[0]['generated_text']
        
        # 4. Extract just the answer part
        if task == "doc_classification":
            # Extract document type classification
            if "### Answer" in response:
                answer = response.split("### Answer")[1].strip()
                # Further clean up to get just the document type
                for doc_type in ["Bank Statement", "Paystub", "W2", "Other"]:
                    if doc_type.lower() in answer.lower():
                        return doc_type
                return "Other"  # Default if no clear match
            return response.split("Answer:")[-1].strip()
            
        elif task == "first_page_detection":
            # Extract True/False for first page detection
            if "### Answer" in response:
                answer = response.split("### Answer")[1].strip()
                if "true" in answer.lower():
                    return "True"
                elif "false" in answer.lower():
                    return "False"
                return answer
            
            answer = response.split("Answer:")[-1].strip()
            # Further clean up to get just True/False
            if "true" in answer.lower():
                return "True"
            elif "false" in answer.lower():
                return "False"
            return answer
            
        elif task == "combined_classification":
            # Extract combined classification (Document Type:First Page Status)
            if "### Answer" in response:
                answer = response.split("### Answer")[1].strip()
                # Look for pattern like "Bank Statement:True"
                for line in answer.split("\n"):
                    if ":" in line and any(doc_type.lower() in line.lower() for doc_type in ["Bank Statement", "Paystub", "W2", "Other"]):
                        return line.strip()
                
                # If we don't find the right pattern, try to construct it
                doc_type = None
                is_first = None
                
                for doc_type_option in ["Bank Statement", "Paystub", "W2", "Other"]:
                    if doc_type_option.lower() in answer.lower():
                        doc_type = doc_type_option
                        break
                
                if "true" in answer.lower():
                    is_first = "True"
                elif "false" in answer.lower():
                    is_first = "False"
                    
                if doc_type and is_first:
                    return f"{doc_type}:{is_first}"
                
                return answer
            
            answer = response.split("Answer:")[-1].strip()
            # Try to find or construct the pattern
            for line in answer.split("\n"):
                if ":" in line and any(doc_type.lower() in line.lower() for doc_type in ["Bank Statement", "Paystub", "W2", "Other"]):
                    return line.strip()
            
            # If we don't find the right pattern, try to construct it
            doc_type = None
            is_first = None
            
            for doc_type_option in ["Bank Statement", "Paystub", "W2", "Other"]:
                if doc_type_option.lower() in answer.lower():
                    doc_type = doc_type_option
                    break
            
            if "true" in answer.lower():
                is_first = "True"
            elif "false" in answer.lower():
                is_first = "False"
                
            if doc_type and is_first:
                return f"{doc_type}:{is_first}"
            
            return answer
            
        else:
            # General query answer extraction
            return response.split("Answer:")[-1].strip()
    
    def evaluate_document_classification(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate document classification performance using the test dataframe
        
        Args:
            test_df: Dataframe with columns 'text', 'label', and 'first_pg'
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Ensure required columns exist
        required_cols = ['text', 'label', 'first_pg']
        for col in required_cols:
            if col not in test_df.columns:
                logger.error(f"Required column '{col}' not found in test dataframe")
                return {"error": f"Missing column: {col}"}
        
        # Limit evaluation to a reasonable number of samples if needed
        sample_size = min(len(test_df), 100)  # Adjust based on computation resources
        if len(test_df) > sample_size:
            logger.info(f"Sampling {sample_size} documents for evaluation")
            eval_df = test_df.sample(sample_size, random_state=42)
        else:
            eval_df = test_df
            
        # Initialize results storage
        results = {
            'document_type': {
                'true': [],
                'pred': []
            },
            'first_page': {
                'true': [],
                'pred': []
            },
            'combined': {
                'true': [],
                'pred': []
            }
        }
        
        # Run predictions
        logger.info(f"Evaluating on {len(eval_df)} samples")
        for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating"):
            text = row['text']
            true_label = row['label']
            true_first_page = str(row['first_pg'])
            true_combined = f"{true_label}:{true_first_page}"
            
            # Run document type classification
            pred_label = self.query(f"Classify this document: {text}", task="doc_classification")
            
            # Run first page detection
            pred_first_page = self.query(f"Is this the first page: {text}", task="first_page_detection")
            
            # Combined classification
            pred_combined = self.query(f"Classify this document and determine if it's a first page: {text}", 
                                        task="combined_classification")
            
            # Store results
            results['document_type']['true'].append(true_label)
            results['document_type']['pred'].append(pred_label)
            
            results['first_page']['true'].append(true_first_page)
            results['first_page']['pred'].append(pred_first_page)
            
            results['combined']['true'].append(true_combined)
            results['combined']['pred'].append(pred_combined)
        
        # Calculate metrics for document type classification
        doc_type_accuracy = accuracy_score(
            results['document_type']['true'], 
            results['document_type']['pred']
        )
        
        doc_type_f1 = f1_score(
            results['document_type']['true'], 
            results['document_type']['pred'],
            average='weighted'
        )
        
        # Calculate metrics for first page detection
        first_page_accuracy = accuracy_score(
            results['first_page']['true'], 
            results['first_page']['pred']
        )
        
        first_page_f1 = f1_score(
            [True if x.lower() == 'true' else False for x in results['first_page']['true']], 
            [True if x.lower() == 'true' else False for x in results['first_page']['pred']],
            average='binary'
        )
        
        # Calculate metrics for combined classification
        combined_accuracy = accuracy_score(
            results['combined']['true'], 
            results['combined']['pred']
        )
        
        # Return metrics
        metrics = {
            'document_type_accuracy': doc_type_accuracy,
            'document_type_f1': doc_type_f1,
            'first_page_accuracy': first_page_accuracy,
            'first_page_f1': first_page_f1,
            'combined_accuracy': combined_accuracy,
            'sample_size': len(eval_df)
        }
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics

# Example usage
def process_test_dataframe(testdf, rag_system):
    """
    Process the test dataframe with the RAG system to add prediction columns and metrics
    
    Args:
        testdf: Test dataframe with document data
        rag_system: Initialized RAG system
        
    Returns:
        Dataframe with added prediction columns and metrics dictionary
    """
    logger.info(f"Processing test dataframe with {len(testdf)} rows")
    
    # Copy the dataframe to avoid modifying the original
    results_df = testdf.copy()
    
    # Initialize new columns
    results_df['predicted_label'] = None
    results_df['predicted_first_pg'] = None
    results_df['combined_prediction'] = None
    
    # Process each document in the test set
    for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Classifying documents"):
        # Get document text - use truncated_text if available, otherwise use text
        doc_text = row.get('truncated_text', row.get('text', ''))
        
        if not doc_text or doc_text == '':
            logger.warning(f"Empty document text for row {idx}")
            continue
            
        # Document type classification
        doc_type = rag_system.query(f"Classify this document: {doc_text}", task="doc_classification")
        
        # First page detection
        is_first_page = rag_system.query(f"Is this the first page: {doc_text}", task="first_page_detection")
        
        # Combined classification
        combined = rag_system.query(f"Classify this document and determine if it's a first page: {doc_text}", 
                                    task="combined_classification")
        
        # Store predictions
        results_df.at[idx, 'predicted_label'] = doc_type
        results_df.at[idx, 'predicted_first_pg'] = is_first_page
        results_df.at[idx, 'combined_prediction'] = combined
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    return results_df, metrics

def calculate_metrics(results_df):
    """
    Calculate classification metrics based on predictions
    
    Args:
        results_df: Dataframe with predictions and ground truth
        
    Returns:
        Dictionary with classification metrics
    """
    # Document type classification metrics
    doc_type_accuracy = accuracy_score(
        results_df['label'], 
        results_df['predicted_label']
    )
    
    doc_type_f1 = f1_score(
        results_df['label'], 
        results_df['predicted_label'],
        average='weighted'
    )
    
    doc_type_precision = precision_score(
        results_df['label'], 
        results_df['predicted_label'],
        average='weighted'
    )
    
    doc_type_recall = recall_score(
        results_df['label'], 
        results_df['predicted_label'],
        average='weighted'
    )
    
    # First page detection metrics
    # Convert to boolean for metrics calculation
    true_first_pg = results_df['first_pg'].astype(str).apply(lambda x: x.lower() == 'true')
    pred_first_pg = results_df['predicted_first_pg'].astype(str).apply(lambda x: x.lower() == 'true')
    
    first_page_accuracy = accuracy_score(true_first_pg, pred_first_pg)
    
    first_page_f1 = f1_score(true_first_pg, pred_first_pg)
    
    first_page_precision = precision_score(true_first_pg, pred_first_pg)
    
    first_page_recall = recall_score(true_first_pg, pred_first_pg)
    
    # Combined metrics
    # Create combined ground truth
    results_df['combined_truth'] = results_df['label'] + ':' + results_df['first_pg'].astype(str)
    
    combined_accuracy = accuracy_score(
        results_df['combined_truth'], 
        results_df['combined_prediction']
    )
    
    # Generate detailed classification report
    label_report = classification_report(
        results_df['label'],
        results_df['predicted_label'],
        output_dict=True
    )
    
    first_page_report = classification_report(
        true_first_pg,
        pred_first_pg,
        output_dict=True
    )
    
    # Create metrics dictionary
    metrics = {
        'document_type': {
            'accuracy': doc_type_accuracy,
            'f1': doc_type_f1,
            'precision': doc_type_precision,
            'recall': doc_type_recall,
            'report': label_report
        },
        'first_page': {
            'accuracy': first_page_accuracy,
            'f1': first_page_f1,
            'precision': first_page_precision,
            'recall': first_page_recall,
            'report': first_page_report
        },
        'combined': {
            'accuracy': combined_accuracy
        },
        'sample_size': len(results_df)
    }
    
    return metrics

def main():
    # Initialize the document classifier RAG system
    rag = DocumentClassifierRAG(
        embedding_model_name=EMBEDDING_MODEL_NAME,
        llm_model_name=LLM_MODEL_NAME,
        device=DEVICE,
        gpu_id=GPU_ID
    )
    
    # Load dataset from CSV (similar to what's shown in the images)
    try:
        dataset_path = '/phoenix/workspaces/zktvbrj/Gen_AI/classification/dataset/dataset_splits_full.csv'
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Add necessary columns if not present
        if 'truncated_text' not in df.columns:
            logger.info("Adding truncated_text column")
            # Generate truncated text (first few lines + last few lines)
            def truncate_text(text, num_lines=4):
                if isinstance(text, str):
                    lines = text.splitlines()
                    if len(lines) <= num_lines * 2:
                        return text
                    return "\n".join(lines[:num_lines] + lines[-num_lines:])
                return text
            
            df['truncated_text'] = df['text'].apply(lambda x: truncate_text(x))
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Use sample data instead
        df = None
    
    # If unable to load the real dataset, use sample documents
    if df is None:
        sample_docs = [
            "CHASE PRIVATE CLIENT July 01, 2021 through July 30, 2021 JPMorgan Chase Bank, N.A. Account Number: 0000291 3672644 P O Box 182051 Columbus, OH 43218-2051 CUSTOMER SERVICE INFORMATION HELIS ZULJIANI-BOYE Service Center: 50 RIVERSIDE BLVD APT 10P Deaf and Hard of Hearing: NEW YORK NY 10069 International Calls: SAVINGS SUMMARY",
            "W2 tax form for 2022. Employee: Jane Smith. SSN: XXX-XX-1234. Employer: ACME Corp. Employer ID: 12-3456789. Wages: $75,000. Federal income tax withheld: $15,000.",
            "ACME CORPORATION EARNINGS STATEMENT Employee: Alice Johnson Employee ID: 987654 Pay Period: 01/01/2023 - 01/15/2023 Gross Pay: $3,200.00 Federal Tax: $640.00 Social Security: $198.40 Medicare: $46.40 State Tax: $160.00 Net Pay: $2,155.20",
            "Page 2 of 5 May 6, 2022 CHASE PRIVATE CLIENT Account Number: 0000291 3672644 TRANSACTION DETAIL DATE DESCRIPTION AMOUNT BALANCE 07/15 ATM Withdrawal -$300.00 $9,700.00 07/20 Direct Deposit Payroll +$2,500.00 $12,200.00"
        ]
        
        sample_metadata = [
            {"label": "Bank Statement", "first_pg": True, "document_date": "2021-07-01"},
            {"label": "W2", "first_pg": True, "document_date": "2022-12-31"},
            {"label": "Paystub", "first_pg": True, "document_date": "2023-01-15"},
            {"label": "Bank Statement", "first_pg": False, "document_date": "2022-05-06"}
        ]
        
        # Index the sample documents
        rag.index_documents(sample_docs, sample_metadata)
    else:
        # Train-test split (60-40 split as shown in the images)
        train_sample_fraction = 0.6
        
        # Split the dataframe at the document level to avoid data leakage
        def train_test_df_split(df, train_fraction):
            fn_unique = df.fn.unique().tolist() if 'fn' in df.columns else [i for i in range(len(df))]
            train_idx, test_idx = train_test_split(fn_unique, train_size=train_fraction, random_state=42)
            
            traindf = pd.DataFrame()
            testdf = pd.DataFrame()
            
            if 'fn' in df.columns:
                for file in train_idx:
                    xdf = df.loc[df['fn'] == file].copy(deep=True)
                    traindf = pd.concat([traindf, xdf], ignore_index=True)
                
                for file in test_idx:
                    ydf = df.loc[df['fn'] == file].copy(deep=True)
                    testdf = pd.concat([testdf, ydf], ignore_index=True)
            else:
                traindf = df.iloc[train_idx].copy(deep=True)
                testdf = df.iloc[test_idx].copy(deep=True)
            
            return traindf, testdf
        
        try:
            traindf, testdf = train_test_df_split(df, train_sample_fraction)
            print(f"Train set size: {len(traindf)}, Test set size: {len(testdf)}")
            
            # Print label distribution
            print("Train label distribution:")
            if 'label' in traindf.columns:
                print(traindf['label'].value_counts())
            
            print("Test label distribution:")
            if 'label' in testdf.columns:
                print(testdf['label'].value_counts())
            
            # Index the training documents
            train_docs = traindf['text'].tolist() if 'text' in traindf.columns else []
            
            # Create metadata dictionary for each document
            train_metadata = []
            for _, row in traindf.iterrows():
                metadata = {}
                if 'label' in row:
                    metadata['label'] = row['label']
                if 'first_pg' in row:
                    metadata['first_pg'] = row['first_pg']
                train_metadata.append(metadata)
            
            # Index the training documents
            if train_docs:
                rag.index_documents(train_docs, train_metadata)
                print(f"Indexed {len(train_docs)} training documents")
        except Exception as e:
            print(f"Error processing dataset: {e}")
            # Fall back to sample documents
            rag.index_documents(sample_docs, sample_metadata)
    
    # Example document classification query
    doc_text = "CHASE PRIVATE CLIENT July 01, 2021 through July 30, 2021 JPMorgan Chase Bank, N.A. Account Number: 0000291 3672644"
    
    # 1. Document type classification
    doc_type = rag.query(f"Classify this document: {doc_text}", task="doc_classification")
    print(f"\nDocument Classification:")
    print(f"Text: {doc_text[:100]}...")
    print(f"Document Type: {doc_type}")
    
    # 2. First page detection
    first_page = rag.query(f"Is this the first page: {doc_text}", task="first_page_detection")
    print(f"\nFirst Page Detection:")
    print(f"Text: {doc_text[:100]}...")
    print(f"Is First Page: {first_page}")
    
    # 3. Combined classification
    combined = rag.query(f"Classify this document and determine if it's a first page: {doc_text}", task="combined_classification")
    print(f"\nCombined Classification:")
    print(f"Text: {doc_text[:100]}...")
    print(f"Result: {combined}")
    
    # 4. Test with a non-first page example
    non_first_page = "Page 2 of 5 May 6, 2022 CHASE PRIVATE CLIENT Account Number: 0000291 3672644 TRANSACTION DETAIL"
    combined = rag.query(f"Classify this document and determine if it's a first page: {non_first_page}", task="combined_classification")
    print(f"\nNon-First Page Classification:")
    print(f"Text: {non_first_page[:100]}...")
    print(f"Result: {combined}")
    
    # 5. Information extraction example (general RAG)
    extract_query = "What is the account number in the document?"
    answer = rag.query(extract_query)
    print(f"\nInformation Extraction:")
    print(f"Query: {extract_query}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
