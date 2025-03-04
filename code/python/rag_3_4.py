#### Setup and Imports ####

#### Install in this order to avoid exception when retrieving dataset
# !pip show datasets
# !pip install -U -q bitsandbytes peft trl accelerate evaluate rouge-score
# !pip install -U -q datasets
# !pip show datasets

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
# --- Remove chromodb imports ---
# import chromadb
# from chromadb import Documents, EmbeddingFunction, Embeddings

# Instead, we will use FAISS for our vector DB:
#import faiss

from model_service_api_util import prompt_model

#### Dataset Configuration ####

bs_t = '920000'  # 224250
w2_t = '920500'  # 14285
ps_t = '920405'  # 4467

tup_to_name = {
    '920000': 'Bank Statement',
    '920500': 'W2',
    '920405': 'Paystub'
}

warnings.filterwarnings("ignore")

# Set global variable to limit cuda memory split size
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

time_start = time.time()

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

torch.backends.cudnn.deterministic = True

#### Utility Functions ####

def check_gpu():
    '''nbk lookup site: https://smartportal.bankofamerica.com/SSO/LookupID/Default.aspx'''
    
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
        
        mem_used_msg = '{}MB {:.2f}%'.format(mem_used_f, mem_used*100/MEM_TOT)
        mem_free_msg = '{}MB {:.2f}%'.format(mem_free_f, mem_free*100/MEM_TOT)
        return mem_used_msg, mem_free_msg
    
    pat = r'/envs/(.+?)/bin/python'
    out1 = subprocess.check_output(['nvidia-smi']).decode()
    out = out1.split('\n')
    procs = [x for x in out if x.endswith('MiB |')]
    
    mmap = {'zk0x08r': ['Aftab', 'DCRS'],
            'zktvbrj': ['Sai', 'DCRS'],
            'zkw1emy': ['Vishal', 'DCRS'],
            'zkx0ek2': ['Venkat', 'DCRS'],
            'zkp3ca8': ['Garreth', 'Global_Pmts'],
            'nbk3vq0': ['Freddy', 'Global_Pmts'],
            'zkjohj2': ['Tony', 'Global_Pmts'],
            'zkiapj4': ['Ammar', 'DCRS']
    }
    
    proc_info = list()
    for proc in procs:
        gpu_num = proc.split()[1]
        pid = proc.split()[4]
        # memory
        mem_used_txt = proc.split()[-2][:-3]
        mem_used_msg, mem_free_msg = format_mem(mem_used_txt)
        
        envir_path = proc.split()[-3]
        envir = re.search(pat, envir_path).group(1) if re.search(pat, envir_path) else 'Unknown'
        oot = subprocess.check_output(['ps', '-f', '-p', pid]).decode()
        nbk = oot.split('\n')[1].split()[0]
        name = mmap.get(nbk, ['Unknown'])[0]
        org = mmap.get(nbk, 'Unknown')[1]
        duration = get_process_uptime(int(pid))
        proc_info.append([name, nbk, org, envir, gpu_num, pid, duration, f'{MEM_TOT:,}MB', mem_used_msg])
    
    # print(out1)
    df = pd.DataFrame(proc_info, columns = ['name', 'nbk', 'org', 'envir', 'gpu', 'pid', 'duration', 'tot_GPU_mem', 'mem_used'])
    display(df)
    
    df['mem_used_int'] = df.mem_used.apply(lambda x: int(x.split('MB')[0].replace(',', '')))
    tdf = df.groupby('gpu')['mem_used_int'].sum().reset_index()
    tdf['mem_free'] = tdf.mem_used_int.apply(lambda x: f'{(MEM_TOT - x):,}MB {((MEM_TOT-x)*100/MEM_TOT):.2f}%')
    
    print()
    print('***GPUs Availability***')
    display(tdf[['gpu', 'mem_free']])

# Load Phoenix model path
def get_model_path(model_name):
    '''Takes model name returns Phoenix encrypted model path.
    model_name = "Mistral-7B-Instruct-v0.1"
    model_name = "llama-2-7b-chat-hf"
    model_name = "llama-2-7b-hf"
    model_name = "Meta-Llama-3-8B-Instruct"
    model_name = "Meta-Llama-3-8B"
    '''
    from os.path import isdir
    try:
        from phoenix_util.model_util import getModel
        model_path = getModel(project_name='DCRS_LLM', model_name=model_name)()
        assert isdir(model_path)
    except:
        model_path=f'/phoenix/lib/models/{model_name}'
        assert isdir(model_path)
    
    print('model_name:', model_name)
    print('model_path:', model_path)
    return model_path

def df_info(df, comment='df info'):
    print(f'******************{comment}******************')
    print('***df len (pages):', len(df))
    print('***page distribution:')
    display(df.label.value_counts())
    
    u_fn_count = df.fn.nunique()
    print('***unique fn count:', u_fn_count)
    print('***fn distribution:')
    display(df.groupby('label')['fn'].nunique().reset_index(name='fn count'))
    fn_count = df.groupby('label')['fn'].nunique().reset_index(name='fn count')['fn count'].sum()
    print('***total fn count:', fn_count)
    print(f'>>>Unique fn count {u_fn_count} and document wise fn count {fn_count} are same: {u_fn_count == fn_count}<<<')
    print('***char & word info:')
    print(' -character info')
    print(df.char_len.describe())
    print(' -word info')
    print(df.word_len.describe())
    
    print('\n**** First page distribution *****')
    d={}
    for label in df.label.unique().tolist():
        vc = df.first_pg.loc[df.label==label].value_counts()
        vc_idx, vc_val = vc.index.tolist(), vc.values
        d[label] = {str(vc_idx[0]): vc_val[0], str(vc_idx[1]): vc_val[1]}
    
    print('Doc      True    False')
    print('-----------------------')
    for label in df.label.unique().tolist():
        print("{:7s} {:7d}    {}".format(label[:7], d[label]['True'], d[label]['False']))

import psutil
def check_ram():
    # Linux: !free -h
    # Total memory
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert from bytes to GB
    # Available memory
    available_memory = psutil.virtual_memory().available / (1024 ** 3)
    # Memory usage
    used_memory = psutil.virtual_memory().used / (1024 ** 3)
    # Memory usage percentage
    memory_percent = psutil.virtual_memory().percent
    
    print(f"Total Memory    : {total_memory:.2f} GB")
    print(f"Available Memory: {available_memory:.2f} GB")
    print(f"Used Memory     : {used_memory:.2f} GB")
    print(f"Memory Usage    : {memory_percent}%")

def make_data(df_, count=100):
    '''Creates and returns a dataframe with
    'count' number of pages for each label.'''
    
    tdf = pd.DataFrame()
    
    for label in df_.label.unique():
        xdf = df_.loc[df_.label == label].sample(count)
        tdf = pd.concat([tdf, xdf])
    return tdf

### train test split tdf
def trn_tst_split(tdf, test_size=0.2):
    trn_idx, tst_idx = train_test_split(tdf.index, test_size = test_size, random_state=123)
    trndf = tdf.loc[trn_idx]
    tstdf = tdf.loc[tst_idx]
    return trndf, tstdf

# Example train_test_split using scikit-learn
from sklearn.model_selection import train_test_split

def train_test_df_split(df: pd.DataFrame, train_fraction: float = 0.8, random_state: int = 42):
    """
    Splits a DataFrame into train/test sets by a given fraction.
    Returns (train_df, test_df).
    """
    train_df, test_df = train_test_split(df, test_size=1 - train_fraction, random_state=random_state)
    return train_df, test_df

#### ChromaDB Setup & Chunking ####

from os.path import isdir

def load_model_on_device(embedding_model_name, device='cuda', gpu_id='2'):
    ### get model encrypted name
    # embedding_model_name = EMBEDDING_MODEL_NAME
    try:
        from phoenix_util.model_util import getModel
        embedding_model_path = getModel(project_name='DCRS_LLM', model_name=embedding_model_name)()
        assert isdir(embedding_model_path)
    except:
        embedding_model_path=f'/phoenix/lib/models/{embedding_model_name}'
        assert isdir(embedding_model_path)
    print(embedding_model_path)
    
    ### device setup
    # device = DEVICE
    if device == 'cuda' and torch.cuda.is_available():
        # if torch.cuda.is_available():
        # gpu_id = GPU_ID
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print('gpu_id:', gpu_id)
        print('os.environ["CUDA_VISIBLE_DEVICES"]:', os.environ["CUDA_VISIBLE_DEVICES"])
        
        print('torch.cuda.is_available():', torch.cuda.is_available())
        
        # Only one GPU should be visible when gpu_id is set
        print('torch.cuda.device_count():',torch.cuda.device_count())
        
        # Torch will assign gpu_id to 0 by default
        print('torch.cuda.current_device():', torch.cuda.current_device())
        
        # device = f"cuda:{torch.cuda.current_device()}"
        device_with_id = device + ':' + gpu_id
    elif device == 'cpu':
        device_with_id = device
    else:
        print('Error: Unknown device:', device)
        return
    
    ### Initialize the SentenceTransformer model
    embedding_model = SentenceTransformer(embedding_model_path, device = device_with_id)
    embedding_model.to(device_with_id)
    print('Loaded embedding model to device:', embedding_model.device)
    return embedding_model

from chromadb import Documents, EmbeddingFunction, Embeddings

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embedding_model):
        # Store the SentenceTransformer model
        self.embedding_model = embedding_model
    
    def __call__(self, input: Documents) -> Embeddings:
        # Use the embedding model to generate embeddings
        # texts = [doc.text for doc in input]
        embeddings = self.embedding_model.encode(sentences=input, show_progress_bar=False)  # returns list of embeddings
        return embeddings.tolist()

def initialize_ChromaDB(embedding_model_name, db_name="document_classification", similarity_algo='cosine', custom_embedding_function=None):
    '''
    Valid options for hnsw:space are "l2", "ip", or "cosine". The default is "l2" which is the squared L2 norm.
    If embedding_function is not None, meaning an embedding is provided and we just have to provide
    the text (chunks) when adding to or querying the vector_db.
    '''
    try:
        global client
        client.delete_collection(db_name)
    except:
        print('No db. Define new one.')
    
    client = chromadb.Client()
    if custom_embedding_function: # an embedding model is provided, bind it to chromadb
        vector_db = client.create_collection(name = db_name,
                                            metadata={'hnsw:space': similarity_algo},
                                            embedding_function=custom_embedding_function)
        
        print(f'Vector_db: {vector_db.name} created successfully\n')
        new_name = vector_db.name + '_' + embedding_model_name +'_emb'
        vector_db.name = new_name
        print('Vector db name updated to:', new_name)
    else:
        vector_db = client.create_collection(name = db_name,
                                            metadata={'hnsw:space': similarity_algo})
        print(f'Vector_db: {vector_db.name} created successfully\n')
    
    return vector_db, client

# a collection of above functions to do all in one functiono call
def load_embed_model_and_chromadb(embedding_model_name, db_name, similarity_algo, device, gpu_id):
    print('***Loading embedding model on device:')
    embedding_model = load_model_on_device(embedding_model_name = embedding_model_name, device=device, gpu_id=gpu_id)
    display(embedding_model)
    
    print()
    print('***Binding embedding model to ChromaDB:')
    #### Bind embedding model to ChromaDB & create DB
    # Create an instance of the custom embedding function
    my_embedding_function = MyEmbeddingFunction(embedding_model)
    
    print()
    print('***Creating a new collection (vectorDB):')
    # Create a new collection (vectorDB)
    vector_db, client = initialize_ChromaDB(embedding_model_name, db_name=db_name, similarity_algo=similarity_algo, custom_embedding_function=my_embedding_function)
    display(vector_db.model_dump())
    return embedding_model, vector_db, client, my_embedding_function

def chunk_document_with_overlap(text: str, chunk_size=200, overlap=0) -> List[str]:
    """
    Splits a single document's text into chunks of 'chunk_size' words with optional overlap
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
            unique_id = f"doc{doc_idx}_chunk{chunk_idx}"
            
            if metadata_list is not None:
                doc_metadata = metadata_list[doc_idx][chunk_idx]
                
                if not doc_metadata:
                    doc_metadata = {"placeholder": "N/A"}
                
                collection.add(
                    documents=[chunk_text],
                    metadatas=[doc_metadata],
                    ids=[unique_id]
                )
            else:
                collection.add(
                    documents=[chunk_text],
                    ids=[unique_id]
                )
    count_after = collection.count()
    print(f"Stored {count_after - count_before} new chunks in Chroma DB.")

### RAG Retrieval & Classification ###

def rag_retrieve(query: str, vector_db, k: int = 3) -> str:
    """
    Retrieves top-k relevant chunks from Chroma DB for the given query text.
    Returns a single string containing the retrieved chunks.
    """
    results = vector_db.query(query_texts=[query], n_results=k)
    retrieved_docs = results.get("documents", [[]])[0]
    if not retrieved_docs:
        print("No relevant documents found in Chroma DB.")
        return ""
    context = "\n".join(retrieved_docs)
    return context

### Document Classification System Prompts ###

DOC_CLASSIFIER_SYSTEM_PROMPT = """
You are a document classification assistant. Given the retrieved text below, classify the document type:
1) Bank Statement
2) Paystub
3) Other

A bank statement provides a detailed summary of transactions, deposits, and withdrawals over a specific period.
A paystub details an employee's earnings and deductions.

Use only the retrieved context to make your classification. If the context does not contain enough information, classify as "Other."

### Retrieved Context ###
{retrieved_chunks}

### User Query ###
{user_query}

### Answer (One of: Bank Statement, Paystub, Other) ###
"""

SPLIT_CLASSIFIER_SYSTEM_PROMPT = """
You are a text classification assistant. Based on the retrieved document content, determine if this is the first page of a document.

Indicators of a first page include:
- Presence of a title or header
- Introduction or overview text
- Absence of "continued" or page numbers indicating mid-document content

Use only the retrieved text to make your decision.

### Retrieved Context ###
{retrieved_chunks}
"""

CLASSIFIER_SYSTEM_PROMPT = """
You are a document classification assistant. Given the retrieved document content, determine both:
1) The document type (Bank Statement, Paystub, W2, or Other).
2) Whether this is the first page of the document (True/False).

Use only the retrieved text to make your decision.

### Retrieved Context ###
{retrieved_chunks}

### User Query ###
{user_query}

### Answer Format ###
Return a JSON object with exactly two fields:
- "document_type": The type of document (Bank Statement, Paystub, W2, or Other)
- "is_first_page": Boolean value (true or false, not "True" or "False")

Example:
{"document_type": "Bank Statement", "is_first_page": true}
"""

#### Document Classification RAG Functions ####

async def doc_classification_rag(
    user_query: str,
    vector_db,
    model_name: str,
    model_params: dict,
    top_k: int = 3
) -> str:
    """
    1) Retrieves context from Chroma DB.
    2) Constructs final prompt with DOC_CLASSIFIER_SYSTEM_PROMPT + context.
    3) Calls the remote model via prompt_model(...) to classify doc type.
    """
    # Step 1: Retrieve relevant chunks
    context = rag_retrieve(user_query, vector_db, k=top_k)
    
    # Step 2: Build the user portion of the prompt
    # In this approach, 'system_prompt' and 'query' are separate parameters to prompt_model(...).
    # The system prompt is the classification instructions, and the query consists of context + user question.
    final_user_query = f"Document Text:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    
    # Step 3: Call the model service API
    # 'prompt_model' is presumably asynchronous, so we use 'await'.
    result = await prompt_model(
        query=final_user_query,
        system_prompt=DOC_CLASSIFIER_SYSTEM_PROMPT,
        model_name=model_name,
        model_params=model_params,
        top_k=top_k
    )
    return result

async def first_page_classification_rag(
    user_query: str,
    vector_db,
    model_name: str,
    model_params: dict,
    top_k: int = 3
) -> str:
    """
    1) Retrieves context from Chroma DB.
    2) Constructs final prompt with SPLIT_CLASSIFIER_SYSTEM_PROMPT + context.
    3) Calls the remote model to classify if it's first page (True/False).
    """
    context = rag_retrieve(user_query, vector_db, k=top_k)
    final_user_query = f"Page Text:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    
    result = await prompt_model(
        query=final_user_query,
        system_prompt=SPLIT_CLASSIFIER_SYSTEM_PROMPT,
        model_name=model_name,
        model_params=model_params,
        top_k=top_k
    )
    return result

async def combined_classification_rag(
    user_query: str,
    vector_db,
    model_name: str,
    model_params: dict,
    top_k: int = 3
) -> dict:
    """
    Single call to do both doc-type classification and first-page detection.
    1) Retrieves context from Chroma DB.
    2) Uses CLASSIFIER_SYSTEM_PROMPT + context + user_query.
    3) Calls the remote model for a combined classification result.
    4) Returns a clean JSON object with only document_type and is_first_page
    """
    context = rag_retrieve(user_query, vector_db, k=top_k)
    final_user_query = f"Document Text:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    
    result = await prompt_model(
        query=final_user_query,
        system_prompt=CLASSIFIER_SYSTEM_PROMPT,
        model_name=model_name,
        model_params=model_params,
        top_k=top_k
    )
    
    # If the model doesn't return a proper JSON, process the result
    # This handles cases where the model might not follow instructions exactly
    try:
        # If result is already a dict with our fields, use it directly
        if isinstance(result, dict) and "document_type" in result and "is_first_page" in result:
            return {
                "document_type": result["document_type"],
                "is_first_page": result["is_first_page"]
            }
        
        # If it's JSON string, parse it
        import json
        import re
        
        # Try to extract a JSON object from the string using regex
        # This handles cases where the model might add extra text
        json_pattern = r'\{.*?\}'
        json_match = re.search(json_pattern, str(result), re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            json_data = json.loads(json_str)
            return {
                "document_type": json_data.get("document_type", "Unknown"),
                "is_first_page": json_data.get("is_first_page", False)
            }
        
        # If still not parsed, try to extract from text
        # This is a fallback for cases where the model doesn't return JSON
        doc_type = "Unknown"
        is_first = False
        
        if "Bank Statement" in str(result):
            doc_type = "Bank Statement"
        elif "Paystub" in str(result):
            doc_type = "Paystub"
        elif "W2" in str(result):
            doc_type = "W2"
        elif "Other" in str(result):
            doc_type = "Other"
            
        is_first = "true" in str(result).lower() or "first page: true" in str(result).lower()
        
        return {
            "document_type": doc_type,
            "is_first_page": is_first
        }
        
    except Exception as e:
        print(f"Error processing classification result: {e}")
        # Return a default response in case of parsing errors
        return {
            "document_type": "Unknown",
            "is_first_page": False,
            "error": "Failed to parse model response"
        }
    
    return result

#### Main Pipeline Implementation ####

# 1) Load CSV dataset
df = load_dataset('ESO_train_15k_pages_top_20_plus_other.csv')
print("Loaded dataset with shape:", df.shape)

# 2) Load CSV dataset
dataset_path = '/phoenix/workspaces/zktvbrj/Gen_AI/classification/dataset/dataset_splits_full.csv'
df = load_dataset(dataset_path)
print("Loaded dataset with shape:", df.shape)

# 3) Create truncated document text
# Suppose CSV has columns like: ["doc_content", "label", "first_pg"] (adjust if needed).
# We'll create a truncated version of each doc:
df["truncated_text"] = df["doc_content"].apply(lambda x: truncate_doc_text(str(x), top_n=4, bottom_n=4))

# 4) Train-Test Split
train_sample_fraction = 0.6
traindf, testdf = train_test_df_split(df, train_fraction=train_sample_fraction, random_state=42)
print("Train set size:", len(traindf), "Test set size:", len(testdf))
print("Train label distribution:\n", traindf["label"].value_counts())
print("Test label distribution:\n", testdf["label"].value_counts())

# 5) Initialize ChromaDB with embedding model
embedding_model_name = "all-MiniLM-L6-v2"  # embedding_models = "all-MiniLM-L6-v2", "all-mpnet-base-v2", "bert-base-uncased", "jina-embeddings-v2-base-en"
db_name = "label_split_classification"
similarity_algo='cosine' #cosine or L2
device = 'cuda'  # 'cpu' or 'cuda'
gpu_id = '2'  # '2' or '3' - currently for DCRS

embedding_model, vector_db, client, my_embedding_function = load_embed_model_and_chromadb(embedding_model_name, db_name, similarity_algo, device, gpu_id)

# 6) Chunk & Store Documents in Chroma
# ---------------------------------
# 4) Chunk & Store Documents in Chroma
#    (Here we store only the TRAIN docs, or both if desired)
# ---------------------------------
train_texts = traindf["truncated_text"].tolist()
chunked_docs = [chunk_document_with_overlap(txt, chunk_size=200, overlap=0) for txt in train_texts]

# If you want to store metadata (like label/is_first_pg) for each chunk, create a matching structure.
# We'll skip metadata here for simplicity:
#metadata_list = None

metadata_list = []
for (idx, row), doc_chunks in zip(traindf.iterrows(),chunked_docs):
    doc_metadata = {
        "label": row["label"],
        "first_pg": row["first_pg"],
        "doc_id": idx,
        "pg_range": row.get("pg_rng", "N/A"),
        "is_split": row.get("is_split", False)
    }
    
    doc_chunks_metadata = [doc_metadata] * len(chunked_docs)
    metadata_list.append(doc_chunks_metadata)

store_in_db(chunked_docs, vector_db, metadata_list=metadata_list)

# 7) Load LLM for classification
model_name = "Meta-Llama-3.3-70B-Instruct"  # Default model

# 8) Example RAG Classification Queries
# We'll pick a sample from the test set
sample_test_doc = testdf.iloc[1]
test_doc_text = sample_test_doc["truncated_text"]
test_label = sample_test_doc["label"]
test_is_first = sample_test_doc["first_pg"]

print("\n--- Sample test doc text ---\n", test_doc_text)

# 9) Document Classification
result = await doc_classification_rag(
    user_query=test_doc_text,
    vector_db=vector_db,  # your Chroma DB collection
    model_name = "Meta-Llama-3.3-70B-Instruct",
    model_params={
        "temperature": 0.2,
        "top_p": 0.95,
        "logprobs": True,
        "prompt_type": "instruction"
    },
    top_k=3
)
print("Document Classification Result:", result)

# 10) First-Page Classification
result = await first_page_classification_rag(
    user_query=test_doc_text,
    vector_db=vector_db,  # your Chroma DB collection
    model_name = "Meta-Llama-3.3-70B-Instruct",
    model_params={
        "temperature": 0.2,
        "top_p": 0.95
    },
    top_k=3
)
print("First-Page Classification Result:", result)

# 11) Combined Classification
result = await combined_classification_rag(
    user_query=test_doc_text,
    vector_db=vector_db,  # your Chroma DB collection
    model_name = "Meta-Llama-3.3-70B-Instruct",
    model_params={
        "temperature": 0.2,
        "top_p": 0.95,
        "response_format": {"type": "json_object"}  # Request JSON format if the API supports it
    },
    top_k=3
)
print("Combined Classification Result:", result)
print("Document Type:", result["document_type"])
print("Is First Page:", result["is_first_page"])
