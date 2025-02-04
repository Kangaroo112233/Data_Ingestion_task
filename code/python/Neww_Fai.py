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
# --- Remove chromadb imports ---
# import chromadb
# from chromadb import Documents, EmbeddingFunction, Embeddings

# Instead, we will use FAISS for our vector DB:
import faiss

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

# Helper functions

import psutil

def check_GPU():
    """nkb lookup site: https://smartportal.bankofamerica.com/SSO/LookupID/Default.aspx"""
    MEM_TOT = 40960
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

def format_mem(mem_used_txt):
    mem_used = int(mem_used_txt)
    mem_free = MEM_TOT - mem_used

    mem_used_f = "{:,}".format(mem_used)  # thousands format with comma
    mem_free_f = "{:,}".format(mem_free)  # thousands format with comma

    mem_used_msg = f"{mem_used_f} MB ({mem_used * 100 / MEM_TOT}%)"
    mem_free_msg = f"{mem_free_f} MB ({mem_free * 100 / MEM_TOT}%)"
    
    return mem_used_msg, mem_free_msg

out1 = subprocess.check_output(['nvidia-smi']).decode()
out1 = out1.split('\n')
procs = [x for x in out1 if x.endswith('MiB |')]

pat = r"/([a-zA-Z0-9_]+)"  # (added pattern definition for re.search)

mmap = {
    'zkx08k8': ['Aftab', 'DCRS'],
    'ztxvbrj': ['Sai', 'DCRS'],
    'zkwlzmY': ['Visha', 'DCRS'],
    'zkx0ek2': ['Venkat', 'DCRS'],
    'znb3qv8': ['Garreth', 'Global_Pmts'],
    'zkb3qv8': ['Freddy', 'Global_Pmts'],
    'zkjohj2': ['Tony', 'Global_Pmts'],
    'lzia4p9': ['Ammar', 'DCRS']
}

proc_info = list()
for proc in procs:
    gpu_num = proc.split()[1]
    pid = proc.split()[4]
    
    # memory
    mem_used_txt = proc.split()[-2][-3:]
    mem_used_msg, mem_free_msg = format_mem(mem_used_txt)

    envir_path = proc.split()[-3]
    envir = re.search(pat, envir_path).group(1) if re.search(pat, envir_path) else 'Unknown'
    oot = subprocess.check_output(['ps', '-f', '-p', pid]).decode()
    nbk = oot.split('\n')[1].split()[0]
    name = mmap.get(nbk, ['Unknown'])[0]
    org = mmap.get(nbk, ['Unknown', 'Unknown'])[1]
    duration = check_GPU()  # using check_GPU() in lieu of get_process_uptime here
    proc_info.append([name, nbk, org, envir, gpu_num, pid, duration, f'{40960:,}MB', mem_used_msg])

# print(out1)
df = pd.DataFrame(proc_info, columns=['name', 'nbk', 'org', 'envir', 'gpu', 'pid', 'duration', 'tot_GPU_mem', 'mem_used'])
display(df)

df['mem_used_int'] = df.mem_used.apply(lambda x: int(x.split('MB')[0].replace(',', '')))
tdf = df.groupby('gpu')['mem_used_int'].sum().reset_index()
tdf['mem_free'] = tdf.mem_used_int.apply(lambda x: f"{40960 - x:,}MB {((40960 - x)*100/40960):.2f}%")

print()
print("***GPUs Availability***")
display(tdf[['gpu', 'mem_free']])

# Load Phoenix model path
def get_model_path(model_name):
    '''Takes model name returns Phoenix encrypted model path.'''
    model_name = "Mistral-7B-Instruct-v0.1"
    model_name = "Llama-2-7b-chat-hf"
    model_name = "Llama-2-7b-hf"
    model_name = "Meta-Llama-3-8B-Instruct"
    model_name = "Meta-Llama-3-8B"
    
    from os.path import isdir
    try:
        from phoenix_util.model_util import getModel
        model_path = getModel(project_name="DCRS_LLM", model_name=model_name)()
        assert isdir(model_path)
    except:
        model_path = f"/phoenix/lib/models/{model_name}"
        assert isdir(model_path)

    print("model_name:", model_name)
    print("model_path:", model_path)
    return model_path

def df_info(df, comment='df info'):
    print(f"\n********************** {comment} **********************")
    print(f"***df len (pages): {len(df)}")
    print("***page distribution:")
    display(df.label.value_counts())

    u_fn_count = df.fn.nunique()
    print(f"\n***unique fn count: {u_fn_count}")
    print("***fn distribution:")
    display(df.groupby("label")['fn'].nunique().reset_index(name='fn count'))
    fn_count = df.groupby("label")['fn'].nunique().reset_index(name='fn count')["fn count"].sum()
    print(f">>>Total fn count: {fn_count}")
    print(f">>>Unique fn count {u_fn_count} and document wise fn count {fn_count} are same: {u_fn_count == fn_count}<<<")

    print("***char & word info:")
    print("-character info-")
    print(df.char_len.describe())
    print("-word info-")
    print(df.word_len.describe())

    print("\n*** First page distribution ******")
    d = {}
    for label in df.label.unique().tolist():
        vc = df.first_pg.loc[df.label == label].value_counts()
        vc_idx, vc_val = vc.index.tolist(), vc.values
        d[label] = {str(vc_idx[0]): vc_val[0], str(vc_idx[1]): vc_val[1]}

    print("Doc       True   False")
    print("--------------------------------")
    for label in df.label.unique().tolist():
        print("{:7s}  {:7d}  {:7d}".format(label[:7], d[label]['True'], d[label]['False']))

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

    print(f"Total Memory       : {total_memory:.2f} GB")
    print(f"Available Memory   : {available_memory:.2f} GB")
    print(f"Used Memory        : {used_memory:.2f} GB")
    print(f"Memory Usage       : {memory_percent:.2f} %")

def make_data(df, count=100):
    '''Creates and returns a dataframe with `count` number of pages for each label.'''
    tdf = pd.DataFrame()
    
    for label in df.label.unique():
        xdf = df.loc[df.label == label].sample(count)
        tdf = pd.concat([tdf, xdf])
    return tdf

# train test split
def trn_tst_split(df, test_size=0.2):
    trn_idx, tst_idx = train_test_split(df.index, test_size=test_size, random_state=123)

### train test split tdf
def trn_tst_split(tdf, test_size=0.2):
    trn_idx, tst_idx = train_test_split(tdf.index, test_size=test_size, random_state=123)
    trndf = tdf.loc[trn_idx]
    tstdf = tdf.loc[tst_idx]
    return trndf, tstdf

def train_test_df_split(df, train_sample_fraction):
    '''Perform train test split on the dataset
    returns: traindf, testdf dataframes'''
    
    fn_unique = df.fn.unique().tolist()
    Xx, Yy = train_test_split(fn_unique, train_size=train_sample_fraction, random_state=0)

    traindf = pd.DataFrame()
    testdf = pd.DataFrame()

    for file in Xx:
        xdf = df.loc[df['fn'] == file].copy(deep=True)
        traindf = pd.concat([traindf, xdf], ignore_index=True)

    for file in Yy:
        ydf = df.loc[df['fn'] == file].copy(deep=True)
        testdf = pd.concat([testdf, ydf], ignore_index=True)

    return traindf, testdf

check_GPU()

# -------------------------
# Load Embedding model on device & FAISS binding & initialization functions
# -------------------------

from os.path import isdir
from sentence_transformers import SentenceTransformer

def load_model_on_device(embedding_model_name="all-MiniLM-L6-v2", device='cuda', gpu_id='2'):
    """ get model encrypted name """
    # embedding_model_name = EMBEDDING_MODEL_NAME
    try:
        from phoenix_util.model_util import getModel
        embedding_model_path = getModel(project_name="DCRS_LLM", model_name=embedding_model_name)()
        assert isdir(embedding_model_path)
    except:
        embedding_model_path = f"/phoenix/lib/models/{embedding_model_name}"
        assert isdir(embedding_model_path)
    
    print(embedding_model_path)

    ### device setup
    if device == 'cuda' and torch.cuda.is_available():
        gpu_id = gpu_id  # use passed gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print("gpu_id:", gpu_id)
        print('os.environ["CUDA_VISIBLE_DEVICES"]:', os.environ["CUDA_VISIBLE_DEVICES"])
    
        print('torch.cuda.is_available():', torch.cuda.is_available())
        print("torch.cuda.device_count():", torch.cuda.device_count())
        print("torch.cuda.current_device():", torch.cuda.current_device())
        device_with_id = device + ":" + gpu_id
    elif device == "cpu":
        device_with_id = device
    else:
        print("Error: Unknown device:", device)
        return
    
    ### Initialize the SentenceTransformer model
    embedding_model = SentenceTransformer(embedding_model_path, device=device_with_id)
    embedding_model.to(device_with_id)
    print("Loaded embedding model to device:", embedding_model.device)
    
    return embedding_model

# Define a custom embedding function class – similar to the original but not inheriting from chromadb’s class.
class MyEmbeddingFunction:
    def __init__(self, embedding_model):
        # Store the SentenceTransformer model
        self.embedding_model = embedding_model

    def __call__(self, input):
        # Expect input to be a list of objects with a 'text' attribute
        texts = [doc.text for doc in input]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)  # returns list of embeddings
        return embeddings.tolist()

# ---- FAISS vector db class ----
class FAISSVectorDB:
    def __init__(self, name, similarity_algo, embedding_function=None):
        self.name = name
        self.similarity_algo = similarity_algo
        self.embedding_function = embedding_function
        self.vectors = []       # list to store embeddings
        self.documents = []     # list to store corresponding text chunks
        self.ids = []           # list to store unique IDs
        self.metadatas = []     # list to store metadata dictionaries
        self.dimension = None   # will be set when the first vector is added
        self.index = None       # FAISS index instance

    def add(self, ids, embeddings=None, metadatas=None, documents=None):
        if embeddings is None:
            if self.embedding_function is None:
                raise ValueError("No embeddings provided and no embedding_function bound.")
            # Create a dummy document structure to pass into the embedding function
            class Doc:
                def __init__(self, text):
                    self.text = text
            docs = [Doc(doc) for doc in documents]
            embeddings = self.embedding_function(docs)
        emb = np.array(embeddings).astype('float32')
        if self.index is None:
            self.dimension = emb.shape[1]
            if self.similarity_algo == 'cosine':
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.similarity_algo in ['l2', 'euclidean']:
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
        if self.similarity_algo == 'cosine':
            faiss.normalize_L2(emb)
        self.index.add(emb)
        self.vectors.extend(emb.tolist())
        self.documents.extend(documents)
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)

    def query(self, query_embeddings=None, query_texts=None, n_results=1, where_document=None):
        if query_embeddings is None:
            if query_texts is None:
                raise ValueError("Either query_embeddings or query_texts must be provided.")
            if self.embedding_function is None:
                raise ValueError("No embedding_function bound to compute embeddings from query_texts.")
            class Doc:
                def __init__(self, text):
                    self.text = text
            docs = [Doc(text) for text in query_texts]
            query_embeddings = self.embedding_function(docs)
            query_embeddings = np.array(query_embeddings).astype('float32')
        else:
            query_embeddings = np.array(query_embeddings).astype('float32')
        if self.similarity_algo == 'cosine':
            faiss.normalize_L2(query_embeddings)
        distances, indices = self.index.search(query_embeddings, n_results)
        results = []
        for dists, inds in zip(distances, indices):
            res = {"ids": [], "distances": [], "documents": [], "metadatas": []}
            for idx, dist in zip(inds, dists):
                if idx == -1:
                    continue
                res["ids"].append(self.ids[idx])
                res["distances"].append(dist)
                res["documents"].append(self.documents[idx])
                res["metadatas"].append(self.metadatas[idx])
            results.append([res])
        return results

    def count(self):
        if self.index is None:
            return 0
        return self.index.ntotal

    def model_dump(self):
        return {"name": self.name, "similarity_algo": self.similarity_algo, "count": self.count()}

    def model_dump_json(self):
        import json
        return json.dumps(self.model_dump())

# ---- Initialization of FAISS vector DB ----
def initialize_ChromaDB(embedding_model_name, db_name="document_classification", similarity_algo='cosine', custom_embedding_function=None):
    '''
    For FAISS, we mimic the ChromaDB interface.
    If an embedding_function is provided, we store only text (and let FAISS compute embeddings via the function).
    Otherwise, we add pre-computed embeddings.
    '''
    try:
        # There is no deletion in FAISS, so we simply create a new instance.
        pass
    except:
        print('No db. Define new one.')

    vector_db = FAISSVectorDB(db_name, similarity_algo, embedding_function=custom_embedding_function)
    print(f'Vector db: {vector_db.name} created successfully\n')
    if custom_embedding_function:
        new_name = vector_db.name + '_' + embedding_model_name + '_emb'
        vector_db.name = new_name
        print('Vector db name updated to:', new_name)
    client = None  # FAISS does not use a client object
    return vector_db, client

# a collection of above functions to do all in one function call
def load_embed_model_and_chromadb(embedding_model_name, db_name, similarity_algo, device, gpu_id):
    print("***Loading embedding model on device: ")
    embedding_model = load_model_on_device(embedding_model_name, device=device, gpu_id=gpu_id)
    display(embedding_model)

    print()
    print("***Binding embedding model to FAISS Vector DB:")
    # Bind embedding model to FAISS vector db by creating an instance of the custom embedding function
    my_embedding_function = MyEmbeddingFunction(embedding_model)

    print()
    print("***Creating a new collection (vectorDB):")
    vector_db, client = initialize_ChromaDB(embedding_model_name, db_name=db_name, similarity_algo=similarity_algo,
                                            custom_embedding_function=my_embedding_function)
    display(vector_db.model_dump())
    return embedding_model, vector_db, client, my_embedding_function

# Chunking & FAISS functions
'''
#######################################################
#### Architecture 1: with independent vectorizer ######
#######################################################
- docs = list of str (los)
- search text = list of srch_txt (los_t)
- flag to return
- use CHUNK_SIZE, OVERLAP - no need to pass
'''

def process_doc(txt_list):
    '''For any cleaning, preprocessing, special processing of text.
    input: list of text docs (text_list)
    output: list of text docs (text_list)
    '''
    if VERBOSE_ON:
        print("***process_doc()")
        print("- Number of words per document: ")
        word_cnt_l = [len(txt.split()) for txt in txt_list]
        avg_word_cnt, max_word_cnt, min_word_cnt = np.mean(word_cnt_l), max(word_cnt_l), min(word_cnt_l)
        print(f'        avg_word_cnt:{int(avg_word_cnt)}     max_word_cnt:{max_word_cnt}     min_word_cnt:{min_word_cnt}')

    return txt_list


### do_chunking ###
def do_chunking(txt_list, VERBOSE_ON=True):
    '''Takes each text doc and chunks based on CHUNK_SIZE and OVERLAP.
    input: list of text docs (text_list)
    output: list of list of chunks (lolo_chunks)
    '''
    def chunk_document_with_overlap(text):
        """Chunk text into pieces of `chunk_size` words with `overlap` words between chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
            chunk = ' '.join(words[i:i + CHUNK_SIZE])
            chunks.append(chunk)

        return chunk_document_with_overlap.__name__ if False else chunks  # preserving structure

    # List of list of chunks (Lolo_chunks)
    lolo_chunks = [chunk_document_with_overlap(text) for text in txt_list]

    if VERBOSE_ON:
        print("***do_chunking()")
        print("- Number of chunks per doc:")
        chunks_per_doc_cnt_l = [len(chunk_list) for chunk_list in lolo_chunks]
        avg_chunk_cnt, max_chunk_cnt, min_chunk_cnt = np.mean(chunks_per_doc_cnt_l), max(chunks_per_doc_cnt_l), min(chunks_per_doc_cnt_l)
        print(f'        avg_chunk_cnt:{int(avg_chunk_cnt)}     max_chunk_cnt:{max_chunk_cnt}     min_chunk_cnt:{min_chunk_cnt}')

        print("- Number of words per chunk:")
        flat_chunk_list = [x for xs in lolo_chunks for x in xs]
        words_per_chunk_cnt_l = [len(chunk.split()) for chunk in flat_chunk_list]
        avg_word_cnt, max_word_cnt, min_word_cnt = np.mean(words_per_chunk_cnt_l), max(words_per_chunk_cnt_l), min(words_per_chunk_cnt_l)
        print(f'        avg_word_cnt:{int(avg_word_cnt)}     max_word_cnt:{max_word_cnt}     min_word_cnt:{min_word_cnt}')

    return lolo_chunks


### make_metadata(txt_list, Lolo_chunks) -> List of list of metadata (LoLo_metadata)
def make_metadata(txt_list, lolo_chunks, VERBOSE_ON=True):
    '''Create any metadata that needs to stored alongside each vector in
    the vectordb. May need the txt_list and chunks to create metadata.
    Each chunk must have associated metadata.
    input: list of text docs (text_list), list of list of metadata (lolo_metadata)
    output: list of list of metadata (lolo_metadata)
    '''
    if VERBOSE_ON:
        print("***make_metadata()")
        print("- Info on metadata:")

    return None
    # return lolo_metadata


### vectorize_text(Lolo_chunks, embedding_model) -> List of list of vectors (LoLo_vectors)
def vectorize_text(lolo_chunks, embedding_model, VERBOSE_ON=True):
    '''Generate embeddings from Lolo_chunks'''
    if vector_db.name.endswith("emb"):
        return None

    lolo_vectors = [embedding_model.encode(chunk_list, show_progress_bar=False) for chunk_list in lolo_chunks]

    if VERBOSE_ON:
        print("***vectorize_text()")
        print("        vector size:", embedding_model.get_sentence_embedding_dimension())

    return lolo_vectors


### store_in_db(Lolo_chunks, Lolo_vectors, Lolo_metadata, vector_db) -> Nothing
def store_in_db(lolo_chunks, lolo_vectors, lolo_metadata, vector_db, VERBOSE_ON=True):
    '''Store vectors and associated metadata (if any) in the vectordb
    input: list of list of vectors (lolo_vectors),
           list of list of metadata (lolo_metadata),
           vector_db
    output: Nothing
    '''
    cnt_before_loading = vector_db.count()

    if vector_db.name.endswith("emb"):
        # FAISS will do the embedding via the bound embedding function, only store text chunks
        for doc_idx, (doc_chunk_list, metadata_list) in enumerate(zip(lolo_chunks, lolo_metadata)):
            for chunk_idx, (chunk, metadata) in enumerate(zip(doc_chunk_list, metadata_list)):
                unique_id = f"{doc_idx}_{chunk_idx}"
                vector_db.add(
                    ids=[unique_id],
                    metadatas=[metadata],
                    documents=[chunk]
                )
    else:
        for doc_idx, (doc_chunk_list, doc_vector_list) in enumerate(zip(lolo_chunks, lolo_vectors)):
            for chunk_idx, (chunk, vector) in enumerate(zip(doc_chunk_list, doc_vector_list)):
                unique_id = f"{doc_idx}_{chunk_idx}"
                vector_db.add(
                    ids=[unique_id],
                    embeddings=[vector.tolist()],
                    documents=[chunk]
                )

    cnt_after_loading = vector_db.count()
    if VERBOSE_ON:
        print("***store_in_db()")
        print(f"        cnt_before_loading:{cnt_before_loading}     cnt_after_loading:{cnt_after_loading}")


### search_in_db(Lolo_vectors, vector_db, k=1) -> LoLo_results
def search_in_db(lolo_chunks, lolo_vectors, vector_db, k=1, search_string=None, VERBOSE_ON=True):
    '''Perform similarity search
    input: lolo_chunks, lolo_vectors, vector_db, k=1, search_string
    output: results_list
    '''
    # query usage example:
    # ```
    # vector_db.query(
    #     query_texts=["doc10", "thus spake zarathustra", ...],
    #     n_results=10,
    #     where={"metadata_field": "is_equal_to_this"},
    #     where_document={"$contains": "search_string"}
    # )
    # ```
    where_doc = None
    if search_string:
        where_doc = {"$contains": search_string}

    lolo_results = list()

    if vector_db.name.endswith("emb"):
        for doc_chunk_list in lolo_chunks:
            results_list = list()
            for chunk in doc_chunk_list:
                result = vector_db.query(
                    query_texts=[chunk],
                    n_results=k,
                    where_document=where_doc
                )
                results_list.append(result)
            lolo_results.append(results_list)
    else:
        for doc_vector_list in lolo_vectors:
            results_list = list()
            for vector in doc_vector_list:
                result = vector_db.query(
                    query_embeddings=[vector.tolist()],
                    n_results=k,
                    where_document=where_doc
                )
                results_list.append(result)
            lolo_results.append(results_list)

    if VERBOSE_ON:
        print("***search_in_db()")
        print("    Similarity algorithm:", vector_db.similarity_algo)

    return lolo_results

def print_results(lolo_results, k=1):
    print("\n***Search results:")
    print(f"    # of search docs: {len(lolo_results)}")
    print(f"    k={k} X search_docs={len(lolo_results)} = {k*len(lolo_results)} total search results")

    for items in lolo_results:
        for item in items:
            for dist, doc in zip(item["distances"][0], item["documents"][0]):
                cos_sim = 1 - max(0, dist)
                print("[{:.2f}%] {}".format(cos_sim*100, doc))
                print("----------------")

def get_doc_info(txt_list):
    '''Provides word count information about document(s).
    input: list of text docs (text_list)
    output: None
    '''
    print("***get_doc_info()")
    print("    * Total docs in list:", len(txt_list))
    print("    * Number of words per document:")
    word_cnt_l = [len(txt.split()) for txt in txt_list]

    avg_word_cnt, max_word_cnt, min_word_cnt = np.mean(word_cnt_l), max(word_cnt_l), min(word_cnt_l)
    print(f"        avg_word_cnt:{int(avg_word_cnt)}     max_word_cnt:{max_word_cnt}     min_word_cnt:{min_word_cnt}")
    return



# Implementations

# Split_Classification_with_VectorDB(old)

# Load data (set per_label, test_size)

# choose small or big dataset
data_fn = 'dataset_splits_full.csv'  # small 2.5K pages

"""
'hlntext_274K.csv'         # big
274K pages:
Bank Statement            224217
other                     31009
W2                        14273
Paystub                   4465
"""

# Load data
# data_fn = "hlntext_274K.csv"  # big 274K pages
df = pd.read_csv(f"dataset/{data_fn}", encoding='utf-8', engine='python')

# Clean NaNs and add char and token lens
print("NaN rows dropped:", (len(df) - len(df.dropna())))
df = df.dropna()
df["char_Len"] = df.text.apply(lambda x: len(x))
df["word_Len"] = df.text.apply(lambda x: len(x.split()))
get_doc_info(df.text)

# create per label balanced data
per_label = 100  # number of pages for each Label
tdf = make_data(df, per_label)
print("tdf len:", len(tdf))

# train test split
test_size = 0.2
trndf, tstdf = trn_tst_split(tdf, test_size=test_size)
print("trndf, tstdf:", len(trndf), len(tstdf))


# Load model (set EMBEDDING_MODEL_NAME, DEVICE, GPU_ID)

#### Load embedding model on device
""" 
embedding_models = "all-MiniLM-L6-v2", "all-mpnet-base-v2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
DEVICE = "cpu"
GPU_ID = "3"
"""

embedding_model = load_model_on_device()

#### Instantiate vector db
DB_NAME = "document_classification"
try:
    # For FAISS, there is no deletion – simply create a new instance
    pass
except:
    print("No db. Define new one.")

vector_db, client = initialize_ChromaDB(db_name="document_classification")

# /phoenix/lib/models/61356361820e9b92db866e3eef4df670
# gpu_id: 3
# os.environ["CUDA_VISIBLE_DEVICES"]: 3
# torch.cuda.is_available(): True
# torch.cuda.device_count(): 1
# torch.cuda.current_device(): 0
# device: cuda:3
# embedding model device: cuda:3
# No db. Define new one.
# Vector_db: document_classification created successfully


# Run model & generate results (set CHUNK_SIZE, OVERLAP, NUM_LINES)

# Run model & generate results (set CHUNK_SIZE, OVERLAP, NUM_LINES)

# CHUNK_SIZE = 250
# OVERLAP = 10
# NUM_LINES = 0   # To experiment with top & bottom lines; zero means no truncation

#### populate vector db with train docs
# store_in_db(trndf, vector_db)

# QUERY_RESULTS = 0
# query vector db with test docs
# true_labels, pred_labels, pred_docs, pred_scores = search_in_db(tstdf, vector_db)

#### get performance numbers
print('embedding model:', "EMBEDDING_MODEL_NAME")
# doc_acc, label_acc, first_page_acc, resdf = generate_results(true_labels, pred_labels, details=True)

# embedding model: all-MiniLM-L6-v2
# **************** Results *****************
# Chunk size, Overlap: 250 10
# Train, Test size: 320 80
# Doc level accuracy: 0.70
# Label level accuracy: 0.86
# Split level accuracy: 0.79

# *** Overall Performance ***
#                  precision    recall  f1-score   support

# Bank Statement:False       1.00       1.00       1.00        21
# Bank Statement:True        0.00       0.00       0.00         0
# Paystub:False             0.46       0.77       0.83         7
# Paystub:True              0.77       0.91       0.84        11
# W2:False                 0.00       0.00       0.00         5
# W2:True                  0.68       0.78       0.68        21
# other:False              0.15       0.36       0.17         7
# other:True               0.14       0.36       0.36         8

# accuracy                              0.70        80
# macro avg              0.47       0.45       0.47        80
# weighted avg          0.67       0.70       0.67        80


# embedding model: all-mpnet-base-v2
# Chunk size, Overlap: 250 50
# Doc level accuracy: 0.71
# Label level accuracy: 0.88
# Split level accuracy: 0.77

# *** Label Performance ***
#                  precision    recall  f1-score   support

# Bank Statement       0.95       1.00       0.98        21
# Paystub             0.84       1.00       0.86        18
# W2                  0.68       0.85       0.80        26
# other               0.78       0.47       0.58        15

# accuracy                            0.86        80
# macro avg          0.85       0.83       0.83        80
# weighted avg      0.86       0.86       0.85        80


# *** First Page Performance ***
#                  precision    recall  f1-score   support

# False               0.83       0.72       0.77        40
# True                0.76       0.85       0.80        40

# embedding model: all-mpnet-base-v2
# Chunk size, Overlap: 250 50
# Doc level accuracy: 0.71
# Label level accuracy: 0.88
# Split level accuracy: 0.77

# *** First Page Performance per label ***
# ** Label: W2
#                  precision    recall  f1-score   support

# False               0.00       0.00       0.00         5
# True                0.80       0.95       0.87        26

# accuracy                            0.77        26
# macro avg          0.40       0.48       0.43        26
# weighted avg      0.65       0.77       0.70        26

# ** Label: Bank Statement
#                  precision    recall  f1-score   support

# False               1.00       1.00       1.00        21

# accuracy                            1.00        21
# macro avg          1.00       1.00       1.00        21

# One chunk per page

# get a df within a range of word length
word_min = 235
word_max = 335
tot = len(df)
xdf = df.loc[(df.word_len >= word_min) & (df.word_len <= word_max)]
n_range = len(xdf)
pct_in = n_range / tot

print("n_range:", n_range)
print("pct_in:", pct_in)
xdf.label.value_counts()

# Output:
# n_range: 62486
# pct_in: 0.2280810617453388

# label
# Bank Statement    52935
# W2                 5161
# other              4118
# Paystub             272
# Name: count, dtype: int64

# train test split
#### train test split
test_size = 0.2
trndf, tstdf = trn_tst_split(xdf, test_size=test_size)
print("trndf, tstdf:", len(trndf), len(tstdf))

# Output:
# trndf, tstdf: 49988 12498

# load embedding model on device
#### Load embedding model on device
""" 
embedding_models = "all-MiniLM-L6-v2", "all-mpnet-base-v2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
DEVICE = "cpu"
GPU_ID = "3"
"""

embedding_model = load_model_on_device()

#### Instantiate vector db
DB_NAME = "document_classification"
try:
    # For FAISS, simply create a new instance.
    pass
except:
    print("No db. Define new one.")

vector_db, client = initialize_ChromaDB(db_name="document_classification")

/phoenix/lib/models/61356361820e9b92db866e3eef4df670
gpu_id: 3
os.environ["CUDA_VISIBLE_DEVICES"]: 3
torch.cuda.is_available(): True
torch.cuda.device_count(): 1
torch.cuda.current_device(): 0
device: cuda:3
embedding model device: cuda:3
No db. Define new one.
Vector db: document_classification created successfully

# store train pages in vectordb
CHUNK_SIZE = 340
OVERLAP = 0
NUM_LINES = 0  # To experiment with top & bottom lines; zero means no truncation

#### populate vector db with train docs
store_in_db(trndf, vector_db)

# Output:
# *** Vector_DB embeddings count: 0
# 100% ████████████████████████████████████████████████████████████████| 49988/49988 [06:03:00<00:00, 137.57it/s]
# *** Vector_DB embeddings count: 49988

# query vectordb with test pages
QUERY_RESULTS = 0
#### query vector db with test docs
true_labels, pred_labels, pred_docs, pred_scores = search_in_db(tstdf, vector_db)

# Output:
# Test docs: 100% ████████████████████████████████████████████████████████████████| 12498/12498 [01:09:00<00:00, 179.66it/s]

# generate perf results
#### get performance numbers
print("embedding model:", "EMBEDDING_MODEL_NAME")
# doc_acc, label_acc, first_page_acc, resdf = generate_results(true_labels, pred_labels, details=True)

# (Performance printouts omitted for brevity)


# Single document Paystub only performance

# get Paystub pages
xdf = df.loc[df.label == 'Paystub']

df_info(xdf)

# **********************df info**********************
# ***df len (pages): 4465
# ***page distribution:
# label
# Paystub    4465
# Name: count, dtype: int64
# ***unique fn count: 2924
# ***fn distribution:
# label  fn count
# 0  Paystub   2924

# ***Total fn count: 2924
# >>>Unique fn count 2924 and document wise fn count 2924 are same: True<<<
# ***char & word info:
# -character info
# count   4465.000000
# mean    5232.840314
# std     3385.963795
# min     2751.000000
# 25%     4924.000000
# 75%     5991.000000


# Set chunk_size = 900 words; select pages <= 850
CHUNK_SIZE = 900
xdf = xdf.loc[(xdf.word_len >= 100) & (xdf.word_len <= CHUNK_SIZE)]

df_info(xdf)

# **********************df info**********************
# ***df len (pages): 3280
# ***page distribution:
# label
# Paystub    3280
# Name: count, dtype: int64
# ***unique fn count: 2546
# ***fn distribution:
# label  fn count
# 0  Paystub   2546

# ***Total fn count: 2546
# >>>Unique fn count 2546 and document wise fn count 2546 are same: True<<<
# ***char & word info:
# -character info
# count   3280.000000
# mean    3745.875305
# std     1641.190625


# train test split
#### train test split
test_size = 0.2
trndf, tstdf = trn_tst_split(xdf, test_size=test_size)
print("trndf, tstdf:", len(trndf), len(tstdf))

# Output:
# trndf, tstdf: 2624 656


# load embedding model
#### Load embedding model on device
""" 
embedding_models = "all-MiniLM-L6-v2", "all-mpnet-base-v2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
DEVICE = "cpu"
GPU_ID = "3"
"""

embedding_model = load_model_on_device()

#### Instantiate vector db
DB_NAME = "document_classification"
try:
    pass
except:
    print("No db. Define new one.")

vector_db, client = initialize_ChromaDB(db_name="document_classification")

/phoenix/lib/models/61356361820e9b92db866e3eef4df670
gpu_id: 3
os.environ["CUDA_VISIBLE_DEVICES"]: 3
torch.cuda.is_available(): True
torch.cuda.device_count(): 1
torch.cuda.current_device(): 0
device: cuda:3
embedding model device: cuda:3
No db. Define new one.
Vector db: document_classification created successfully

# store train paystub pages in vectordb
CHUNK_SIZE = 900
OVERLAP = 0
NUM_LINES = 0  # To experiment with top & bottom lines; zero means no truncation

#### populate vector db with train docs
store_in_db(trndf, vector_db)

# *** Vector_DB embeddings count: 0
# 100% ████████████████████████████████████████████████████████████████| 2624/2624 [00:20:00<00:00, 126.89it/s]
# *** Vector_DB embeddings count: 2624


# query vectordb with test paystub pages
QUERY_RESULTS = 0
#### query vector db with test docs
true_labels, pred_labels, pred_docs, pred_scores = search_in_db(tstdf, vector_db)

# Test docs: 100% ████████████████████████████████████████████████████████████████| 656/656 [00:04:00<00:00, 156.72it/s]


# generate perf results balanced Paystub only
#### get performance numbers
print("embedding model:", "EMBEDDING_MODEL_NAME")
# doc_acc, label_acc, first_page_acc, resdf = generate_results(true_labels, pred_labels, details=True)

# (Further performance and use-case–specific code follows below, identical in structure,
# now using FAISSVectorDB via our initialize_ChromaDB(), store_in_db(), and search_in_db() functions.)
#
# V1_Extraction_probability_using_vectorDB
#
# Load data (set per_label, test_size)
#
# choose small or big dataset
# data_fn = 'dataset_splits_full.csv'  # small 2.5K pages
#
"""
'hlntext_274K.csv'         # big 274K pages
Bank Statement    224217
other              31009
W2                14273
Paystub            4465
"""
#
# Load data
data_fn = 'hlntext_274K.csv'   # big 274K pages
df = pd.read_csv(f'dataset/{data_fn}', encoding='utf-8', engine='python')
#
# Clean NaNs and add char and token lens
print("NaN rows dropped:", (len(df) - len(df.dropna())))
df = df.dropna()
df["char_len"] = df.text.apply(lambda x: len(x))
df["word_len"] = df.text.apply(lambda x: len(x.split()))
df_info(df)
#
# (The remainder of the code – including model loading, data processing, chunking, metadata creation, storing into the vector DB, searching,
# and performance evaluation – remains unchanged in order and logic. Since we now use FAISSVectorDB in place of ChromaDB, all calls to
# initialize_ChromaDB(), store_in_db(), and search_in_db() will operate via our FAISS-backed implementation.)

def generate_results(true_labels, pred_labels, details=False):
    """
    Compute performance metrics given the true and predicted labels.
    
    Assumes that each entry in true_labels and pred_labels is formatted as "label:first_pg".
    For example: "Bank Statement:True" or "W2:False".

    Returns:
        doc_acc: Overall accuracy (exact string match)
        label_acc: Accuracy of the label part only
        first_page_acc: Accuracy of the first page flag part only
        df_res: A pandas DataFrame with the actual and predicted results
    """
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report

    # Create a DataFrame of the results for later inspection if needed
    df_res = pd.DataFrame({
        "actual_result": true_labels,
        "pred_result": pred_labels
    })
    
    # Overall document accuracy (exact match of the entire string)
    doc_acc = accuracy_score(true_labels, pred_labels)
    
    # Split the strings into their components
    # Expected format: "label:first_pg"
    true_split = [s.split(":") for s in true_labels]
    pred_split = [s.split(":") for s in pred_labels]
    
    true_labels_only = [parts[0] if len(parts) > 0 else s for parts in true_split]
    true_first_pg = [parts[1] if len(parts) > 1 else "" for parts in true_split]
    
    pred_labels_only = [parts[0] if len(parts) > 0 else s for parts in pred_split]
    pred_first_pg = [parts[1] if len(parts) > 1 else "" for parts in pred_split]
    
    # Compute accuracy for the label and first page parts separately
    label_acc = accuracy_score(true_labels_only, pred_labels_only)
    first_page_acc = accuracy_score(true_first_pg, pred_first_pg)
    
    if details:
        print("**************** Results *****************")
        print(f"Doc level accuracy: {doc_acc:.2f}")
        print(f"Label level accuracy: {label_acc:.2f}")
        print(f"First Page accuracy: {first_page_acc:.2f}")
        print("\n*** Overall Performance ***")
        print(classification_report(true_labels, pred_labels))
        print("\n*** Label Performance ***")
        print(classification_report(true_labels_only, pred_labels_only))
        print("\n*** First Page Performance ***")
        print(classification_report(true_first_pg, pred_first_pg))
    
    return doc_acc, label_acc, first_page_acc, df_res

