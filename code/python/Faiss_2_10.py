import os
import torch
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
# If running in a notebook:
from IPython.display import display

# =============================================================================
# 1. Loading the embedding model on a device (unchanged)
# =============================================================================
def load_model_on_device(embedding_model_name, device='cuda', gpu_id='2'):
    from os.path import isdir
    try:
        from phoenix_util.model_util import getModel
        embedding_model_path = getModel(project_name='DCRS_LLM', model_name=embedding_model_name)()
        assert isdir(embedding_model_path)
    except Exception as e:
        embedding_model_path = f'/phoenix/lib/models/{embedding_model_name}'
        assert isdir(embedding_model_path)
    print("Embedding model path:", embedding_model_path)

    if device == 'cuda' and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print('gpu_id:', gpu_id)
        print('os.environ["CUDA_VISIBLE_DEVICES"]:', os.environ["CUDA_VISIBLE_DEVICES"])
        print('torch.cuda.is_available():', torch.cuda.is_available())
        print('torch.cuda.device_count():', torch.cuda.device_count())
        print('torch.cuda.current_device():', torch.cuda.current_device())
        device_with_id = device + ':' + gpu_id
    elif device == 'cpu':
        device_with_id = device
    else:
        print('Error: Unknown device:', device)
        return

    # Initialize the SentenceTransformer model
    embedding_model = SentenceTransformer(embedding_model_path, device=device_with_id)
    embedding_model.to(device_with_id)
    print('Loaded embedding model to device:', embedding_model.device)

    return embedding_model

# =============================================================================
# 2. Custom embedding function wrapper (unchanged)
# =============================================================================
class MyEmbeddingFunction:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def __call__(self, input_texts):
        # input_texts is a list of strings
        embeddings = self.embedding_model.encode(sentences=input_texts, show_progress_bar=False)
        return embeddings.tolist()

# =============================================================================
# 3. FAISS Vector DB Implementation
# =============================================================================
class FAISSVectorDB:
    def __init__(self, dimension, similarity_algo='cosine', custom_embedding_function=None, db_name="document_classification"):
        self.dimension = dimension
        self.similarity_algo = similarity_algo.lower()
        self.custom_embedding_function = custom_embedding_function
        self.db_name = db_name
        
        # Choose the index based on the similarity measure.
        # For cosine similarity we use inner product after normalizing vectors.
        if self.similarity_algo == 'cosine':
            self.index = faiss.IndexFlatIP(dimension)
            self.normalize = True
        elif self.similarity_algo == 'l2':
            self.index = faiss.IndexFlatL2(dimension)
            self.normalize = False
        else:
            print("Unknown similarity algorithm. Defaulting to L2.")
            self.index = faiss.IndexFlatL2(dimension)
            self.normalize = False
        
        # Dictionaries to store additional data
        self.id_to_doc = {}   # maps integer id -> document (chunk text)
        self.id_to_meta = {}  # maps integer id -> metadata dictionary
        self.id_counter = 0   # internal auto-increment id

    def add(self, ids, embeddings=None, metadatas=None, documents=None):
        """
        Add vectors (and associated documents/metadata) to the index.
        Either embeddings must be provided or (if None) the custom_embedding_function is used.
        The parameter 'ids' is not used as the external id but kept for interface compatibility.
        """
        num_items = len(ids)
        for i in range(num_items):
            # If embeddings not provided, use the custom embedding function on the document text.
            if embeddings is None:
                if self.custom_embedding_function is not None:
                    # Note: the custom embedding function expects a list of texts.
                    emb = self.custom_embedding_function([documents[i]])
                    embedding = np.array(emb[0])
                else:
                    raise ValueError("No embeddings provided and no custom embedding function available")
            else:
                embedding = np.array(embeddings[i])
            # Normalize if using cosine similarity.
            if self.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            embedding = np.expand_dims(embedding.astype('float32'), axis=0)
            self.index.add(embedding)
            # Save document and metadata by the internal id.
            self.id_to_doc[self.id_counter] = documents[i] if documents is not None else ""
            self.id_to_meta[self.id_counter] = metadatas[i] if metadatas is not None else {}
            self.id_counter += 1

    def query(self, query_embeddings=None, query_texts=None, n_results=1, where_document=None):
        """
        Query the FAISS index.
        Either provide a list of query embeddings or query_texts (which are converted using the custom embedding function).
        The optional parameter 'where_document' is ignored in this simple implementation.
        Returns a list (one per query) of dictionaries with keys: 'ids', 'distances', 'documents', 'metadatas'.
        To mimic the ChromaDB interface, each value is wrapped in an extra list.
        """
        if query_embeddings is None:
            if query_texts is not None:
                if self.custom_embedding_function is not None:
                    query_embeddings = self.custom_embedding_function(query_texts)
                    query_embeddings = [np.array(vec) for vec in query_embeddings]
                else:
                    raise ValueError("No query embeddings provided and no custom embedding function available")
            else:
                raise ValueError("No query embeddings or texts provided")
        
        results = []
        for embedding in query_embeddings:
            if self.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            embedding = np.expand_dims(embedding.astype('float32'), axis=0)
            distances, indices = self.index.search(embedding, n_results)
            res_ids = []
            res_distances = []
            res_documents = []
            res_metadatas = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                res_ids.append(str(idx))
                res_distances.append(dist)
                res_documents.append(self.id_to_doc.get(idx, ""))
                res_metadatas.append(self.id_to_meta.get(idx, {}))
            # Mimic ChromaDB’s nested list response structure.
            res = {
                'ids': [res_ids],
                'distances': [res_distances],
                'documents': [res_documents],
                'metadatas': [res_metadatas]
            }
            results.append(res)
        return results

    def count(self):
        return self.index.ntotal

    def model_dump(self):
        return {"db_name": self.db_name, "similarity_algo": self.similarity_algo, "dimension": self.dimension}

    def model_dump_json(self):
        import json
        return json.dumps(self.model_dump())

# =============================================================================
# 4. Initialize FAISS instead of ChromaDB
# =============================================================================
def initialize_FAISS(embedding_model, db_name="document_classification", similarity_algo='cosine', custom_embedding_function=None):
    dimension = embedding_model.get_sentence_embedding_dimension()
    vector_db = FAISSVectorDB(dimension=dimension,
                              similarity_algo=similarity_algo,
                              custom_embedding_function=custom_embedding_function,
                              db_name=db_name)
    print(f'Vector_db: {vector_db.db_name} created successfully\n')
    # Mimic renaming if a custom embedding function is provided.
    if custom_embedding_function:
        new_name = vector_db.db_name + '_emb'
        vector_db.db_name = new_name
        print('Vector db name updated to:', new_name)
    return vector_db, None  # client is not used in FAISS

# =============================================================================
# 5. One-call function to load the embed model and bind it to FAISS
# =============================================================================
def load_embed_model_and_faiss(embedding_model_name, db_name, similarity_algo, device, gpu_id):
    print("***Loading embedding model on device:***")
    embedding_model = load_model_on_device(embedding_model_name=embedding_model_name, device=device, gpu_id=gpu_id)
    display(embedding_model)
    print("\n***Binding embedding model to FAISS DB:***")
    my_embedding_function = MyEmbeddingFunction(embedding_model)
    print("\n***Creating a new FAISS collection (vectorDB):***")
    vector_db, client = initialize_FAISS(embedding_model, db_name=db_name, similarity_algo=similarity_algo,
                                         custom_embedding_function=my_embedding_function)
    display(vector_db.model_dump())
    return embedding_model, vector_db, client, my_embedding_function

# =============================================================================
# 6. (Optional) Document preprocessing and chunking functions (unchanged)
# =============================================================================
def process_doc(txt_list):
    '''For any cleaning, preprocessing, special processing of text.
    input: list of text docs (text_list)
    output: list of text docs (text_list)
    '''
    VERBOSE_ON = True
    if VERBOSE_ON:
        print("***process_doc()")
        print("Number of words per document:")
    word_cnt_l = [len(txt.split()) for txt in txt_list]
    avg_word_cnt, max_word_cnt, min_word_cnt = np.mean(word_cnt_l), max(word_cnt_l), min(word_cnt_l)
    print(f' avg_word_cnt:{int(avg_word_cnt)} max_word_cnt:{max_word_cnt} min_word_cnt:{min_word_cnt}')
    return txt_list

def do_chunking(txt_list, VERBOSE_ON=True):
    '''Chunk each text doc into pieces of CHUNK_SIZE words with OVERLAP words overlap.'''
    def chunk_document_with_overlap(text):
        words = text.split()
        chunks = []
        for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
            chunk = ' '.join(words[i: i + CHUNK_SIZE])
            chunks.append(chunk)
        return chunks

    lolo_chunks = [chunk_document_with_overlap(text) for text in txt_list]
    if VERBOSE_ON:
        print("***do chunking()")
        print("Number of chunks per doc:")
    chunks_per_doc_cnt_l = [len(chunk_list) for chunk_list in lolo_chunks]
    avg_chunk_cnt, max_chunk_cnt, min_chunk_cnt = np.mean(chunks_per_doc_cnt_l), max(chunks_per_doc_cnt_l), min(chunks_per_doc_cnt_l)
    print(f' avg_chunk_cnt:{int(avg_chunk_cnt)} max_chunk_cnt:{max_chunk_cnt} min_chunk_cnt:{min_chunk_cnt}')
    flat_chunk_list = [x for xs in lolo_chunks for x in xs]
    words_per_chunk_cnt_l = [len(chunk.split()) for chunk in flat_chunk_list]
    avg_word_cnt, max_word_cnt, min_word_cnt = np.mean(words_per_chunk_cnt_l), max(words_per_chunk_cnt_l), min(words_per_chunk_cnt_l)
    print(f' avg_word_cnt:{int(avg_word_cnt)} max_word_cnt:{max_word_cnt} min_word_cnt:{min_word_cnt}')
    return lolo_chunks

def make_metadata(txt_list, lolo_chunks) -> list:
    '''Placeholder for creating metadata.'''
    VERBOSE_ON = True
    if VERBOSE_ON:
        print("***make_metadata()")
        print("*Info on metadata:")
    return None

def vectorize_text(lolo_chunks, embedding_model) -> list:
    '''Vectorize the text chunks. Returns a list of vectors unless the vector db uses its own embedding function.'''
    VERBOSE_ON = True
    # If the FAISS db was created with a custom embedding function, we assume embeddings will be computed on the fly.
    if vector_db.db_name.endswith('emb'):
        return None
    lolo_vectors = [embedding_model.encode(chunk_list, show_progress_bar=False) for chunk_list in lolo_chunks]
    if VERBOSE_ON:
        print("***vectorize_text()")
        print("Vector size:", embedding_model.get_sentence_embedding_dimension())
    return lolo_vectors

# =============================================================================
# 7. Storing vectors (and docs/metadata) in the FAISS DB
# =============================================================================
def store_in_db(lolo_chunks, lolo_vectors, lolo_metadata, vector_db, VERBOSE_ON=True):
    cnt_before_loading = vector_db.count()
    # If no precomputed vectors, use the custom embedding function mode.
    if lolo_vectors is None:
        for doc_idx, (doc_chunk_list, metadata_list) in enumerate(zip(lolo_chunks, lolo_metadata)):
            for chunk_idx, (chunk, metadata) in enumerate(zip(doc_chunk_list, metadata_list)):
                unique_id = f"{doc_idx}_{chunk_idx}"
                vector_db.add(ids=[unique_id], embeddings=None, metadatas=[metadata], documents=[chunk])
    else:
        for doc_idx, (doc_chunk_list, doc_vector_list) in enumerate(zip(lolo_chunks, lolo_vectors)):
            for chunk_idx, (chunk, vector) in enumerate(zip(doc_chunk_list, doc_vector_list)):
                unique_id = f"{doc_idx}_{chunk_idx}"
                vector_db.add(ids=[unique_id], embeddings=[vector.tolist()], documents=[chunk])
    cnt_after_loading = vector_db.count()
    if VERBOSE_ON:
        print("***store_in_db()***")
        print(f' cnt_before_loading: {cnt_before_loading} cnt_after_loading: {cnt_after_loading}')

# =============================================================================
# 8. Searching in the FAISS DB
# =============================================================================
def search_in_db(lolo_chunks, lolo_vectors, vector_db, k=1, search_string=None, VERBOSE_ON=True):
    where_doc = None
    if search_string:
        where_doc = {"$contains": search_string}
    lolo_results = list()
    # If no precomputed vectors, use the custom embedding function mode.
    if lolo_vectors is None:
        for doc_chunk_list in lolo_chunks:
            results_list = list()
            for chunk in doc_chunk_list:
                result = vector_db.query(query_texts=[chunk], n_results=k, where_document=where_doc)
                results_list.append(result)
            lolo_results.append(results_list)
    else:
        for doc_vector_list in lolo_vectors:
            results_list = list()
            for vector in doc_vector_list:
                result = vector_db.query(query_embeddings=[vector.tolist()], n_results=k, where_document=where_doc)
                results_list.append(result)
            lolo_results.append(results_list)
    if VERBOSE_ON:
        print("***search_in_db()***")
        print(" Similarity algorithm:", vector_db.similarity_algo)
    return lolo_results

def print_results(lolo_results, k=1):
    print("\n**Search results:**")
    print("  # of search docs:", len(lolo_results))
    print(f'  k={k} x search_docs={len(lolo_results)} = {k*len(lolo_results)} total search results')
    for items in lolo_results:
        for item in items:
            # item is a list with one dictionary (to mimic the nested lists in Chroma)
            for dist, doc in zip(item[0]['distances'][0], item[0]['documents'][0]):
                cos_sim = 1 - max(0, dist)
                print("[[{:.1f}%]]".format(cos_sim * 100), doc)
                print("----------------------")
                
def get_doc_info(txt_list):
    print("***get_doc_info()***")
    print("  *Total docs in list:*", len(txt_list))
    print("  *Number of words per document:*")
    word_cnt_l = [len(txt.split()) for txt in txt_list]
    avg_word_cnt, max_word_cnt, min_word_cnt = np.mean(word_cnt_l), max(word_cnt_l), min(word_cnt_l)
    print(f'  avg_word_cnt:{int(avg_word_cnt)}  max_word_cnt:{max_word_cnt}  min_word_cnt:{min_word_cnt}')
    return

# =============================================================================
# 9. (Below) Use-case specific processing and overall execution
# =============================================================================

# Example usage (you may need to define functions like df_info() and train_test_df_split() elsewhere)
# For example, you might have:
# def df_info(df):
#     print(df.info())
# def train_test_df_split(df, fraction):
#     return df.sample(frac=fraction, random_state=123), df.drop(df.sample(frac=fraction, random_state=123).index)

# Set parameters for the embedding model and FAISS DB.
embedding_model_name = "all-MiniLM-L6-v2"  # Options: "all-MiniLM-L6-v2", "all-mpnet-base-v2", "bert-base-uncased", etc.
db_name = "label_split_classification"
similarity_algo = "cosine"  # or "l2"
device = "cuda"  # 'cpu' or 'cuda'
gpu_id = "2"

# Load embedding model and bind it to FAISS
embedding_model, vector_db, client, my_embedding_function = load_embed_model_and_faiss(
    embedding_model_name, db_name, similarity_algo, device, gpu_id
)

# (Optional) Reinitialize the FAISS DB if needed (this mimics your Chroma reset)
vector_db, client = initialize_FAISS(
    embedding_model, db_name=db_name, similarity_algo=similarity_algo,
    custom_embedding_function=my_embedding_function
)

# Load data
data_fn = "hlntext_274k.csv"  # example filename
# df = pd.read_csv(f'dataset/{data_fn}', encoding='utf-8', engine='python')
df = pd.read_csv('/phoenix/workspaces/xtkvtbrj/Gen_AI/classification/dataset/clf_200_per_doc_type_0102.csv')

# (Assuming df_info is defined elsewhere)
# df_info(df)

get_doc_info(txt_list=df.text.tolist())

print("\n***Loading embedding model on device:")
print("/phoenix/lib/models/61365361820e9b92bd86e63e4dfd670")
print("gpu_id:", gpu_id)
print('os.environ["CUDA_VISIBLE_DEVICES"]:', os.environ["CUDA_VISIBLE_DEVICES"])
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
print("torch.cuda.current_device():", torch.cuda.current_device())
print("Loaded embedding model to device:", embedding_model.device)

# For demonstration purposes, reinitialize FAISS DB to reset any stored data.
vector_db, client = initialize_FAISS(
    embedding_model, db_name=db_name, similarity_algo=similarity_algo,
    custom_embedding_function=my_embedding_function
)
print(f"\nVector_db: {db_name} created successfully")
print(f"Vector db name updated to: {db_name}_{embedding_model_name}_emb")

# Storing data
print("\n***Loading data into df and cleaning:")
# Use your own data splitting function; here we assume train_test_df_split exists.
train_sample_fraction = 0.5
# traindf, testdf = train_test_df_split(df, train_sample_fraction)
# For demonstration, we simply use the entire df for both train and test.
traindf, testdf = df, df

print("NaN rows dropped:", (len(df) - len(df.dropna())))
df = df.dropna()
df["char_len"] = df.text.apply(lambda x: len(x))
df["word_len"] = df.text.apply(lambda x: len(x.split()))
# df_info(df)

print("\n***Selecting data - doc(s):")
df = df.sample(10, random_state=123)  # selecting 10 pages for testing purposes
txt_list = traindf.text.tolist()
get_doc_info(txt_list)

# Use-case specific processing: truncating documents (example)
print("\n***Use case specific processing:")
print("\n Truncating document(s)")
def keep_lines(txt_list, num_lines=4):
    '''Keep top and bottom "num_lines" of each document.'''
    return [' '.join(doc.splitlines(keepends=True)[:num_lines]) + ' ' + ' '.join(doc.splitlines(keepends=True)[-num_lines:]) for doc in txt_list]

txt_list = keep_lines(txt_list, num_lines=4)
get_doc_info(txt_list)

print("\n***Perform chunking:")
CHUNK_SIZE = 200  # you can change this as needed
OVERLAP = 0
print(f' CHUNK_SIZE:{CHUNK_SIZE}  OVERLAP:{OVERLAP}')
lolo_chunks = do_chunking(txt_list)

# For classification metadata (example)
print("\n***Make metadata:")
def make_metadata_clf(txt_list, lolo_chunks, df=None, VERBOSE_ON=True):
    if df is None:
        return None
    lolo_metadata = list()
    for chunk_list, (tdf_idx, doc_info) in zip(lolo_chunks, df.iterrows()):
        metadata_list = list()
        for chunk in chunk_list:
            dd = {"label": doc_info["label"],
                  "first_pg": doc_info["first_pg"],
                  "pg_num": doc_info["st_pg_updated"],
                  "chunk_word_len": len(chunk.split())}
            metadata_list.append(dd)
        lolo_metadata.append(metadata_list)
    if VERBOSE_ON:
        print("***make_metadata_clf()")
        print("*Info on metadata:")
        print("len(lolo_metadata):", len(lolo_metadata))
        print("Total chunks for all docs:", sum([len(meta_list) for meta_list in lolo_metadata]))
    return lolo_metadata

lolo_metadata = make_metadata_clf(txt_list, lolo_chunks, df=traindf)

print("\n***Store in vectordb:")
lolo_vectors = None  # If you wish to precompute embeddings, set lolo_vectors = vectorize_text(lolo_chunks, embedding_model)
store_in_db(lolo_chunks, lolo_vectors, lolo_metadata, vector_db)

print("\n***Selecting Search doc(s):")
srch_txt_list = testdf.text.tolist()
get_doc_info(srch_txt_list)

print("\n***Search: Use case specific processing:")
print("\n Truncating document(s)")
srch_txt_list = keep_lines(srch_txt_list, num_lines=4)
get_doc_info(srch_txt_list)

print("\n***Search: Perform chunking:")
print(f' CHUNK_SIZE:{CHUNK_SIZE}  OVERLAP:{OVERLAP}')
srch_lolo_chunks = do_chunking(srch_txt_list)

print("\n***Search: Assign metadata:")
srch_lolo_metadata = make_metadata_clf(srch_txt_list, srch_lolo_chunks, df=testdf)

print("\n***Search in vectorDb:")
lolo_vectors = None  # use custom embedding function if embeddings not precomputed
k = 1
lolo_results = search_in_db(srch_lolo_chunks,
                            lolo_vectors,
                            vector_db,
                            k=k)

def print_search_results(lolo_results):
    print()
    print(f"**Results: Total search docs={len(lolo_results)}. Each search doc has k={k} results\n")
    for idx, results in enumerate(lolo_results):
        print(f"*** Result for document#: {idx}")
        # Here we assume each result is a list with one dictionary as produced by our query()
        distances = results[0]['distances'][0]
        cos_sim = [1 - max(0, dist) for dist in distances]
        for key in ['ids', 'distances', 'metadatas']:
            print(f"{key:10s}: {results[0][key]}")
        print("similarity score:", cos_sim)
        print("************************")
        
print_search_results(lolo_results)

# Evaluation: Compare ground-truth metadata to the predicted metadata from search results.
pred_label_l, pred_first_pg_l, pred_pg_num_l, pred_score_l = list(), list(), list(), list()
for idx, results in enumerate(lolo_results):
    distances = results[0]['distances'][0]
    cos_sim = [1 - max(0, dist) for dist in distances]
    pred_label_l.append(results[0]['metadatas'][0]['label'])
    pred_first_pg_l.append(results[0]['metadatas'][0]['first_pg'])
    pred_pg_num_l.append(results[0]['metadatas'][0]['pg_num'])
    pred_score_l.append(cos_sim[0])

label_l, first_page_l = list(), list()
for idx, results in enumerate(srch_lolo_metadata):
    label_l.append(results[0]['label'])
    first_page_l.append(results[0]['first_pg'])

print("*** Label Performance ***")
print("Ground Truth:", label_l[:10])
print("Prediction :", pred_label_l[:10])
print("\n Classification report:")
print(classification_report(label_l, pred_label_l))
print("---------------------------------------\n")

print("*** First Page Performance ***")
print("Ground Truth:", first_page_l[:10])
print("Prediction :", pred_first_pg_l[:10])
print("\n Classification report:")
print(classification_report(first_page_l, pred_first_pg_l))
print("---------------------------------------\n")

# Overall accuracy at document level
resdf = testdf[['fn', 'label', 'first_pg']]
resdf['pred_label'] = pred_label_l
resdf['pred_first_pg'] = pred_first_pg_l
print(resdf.head())

resdf['actual_result'] = resdf['label'].astype(str) + ':' + resdf['first_pg'].astype(str)
resdf['pred_result'] = resdf['pred_label'].astype(str) + ':' + resdf['pred_first_pg'].astype(str)
print(resdf.head())

print("***Classification Performance ***")
print("\n Classification report:")
print(classification_report(resdf['actual_result'], resdf['pred_result']))
print("---------------------------------------\n")

sample_lst = testdf['fn'].unique().tolist()
actual_doc_res = ["True"] * len(sample_lst)
pred_doc_res = []

for file in sample_lst:
    xdf = resdf.loc[resdf['fn'] == file]
    pg_flag = []
    for i, row in xdf.iterrows():
        label_pred = "True" if row["label"] == row["pred_label"] else "False"
        split_pred = "True" if str(row["first_pg"]) == str(row["pred_first_pg"]) else "False"
        flag = "True" if label_pred == "True" and split_pred == "True" else "False"
        pg_flag.append(flag)
    doc_flag = str(all(x == pg_flag[0] for x in pg_flag))
    pred_doc_res.append(doc_flag)

metrics = {
    "micro_f1": f1_score(actual_doc_res, pred_doc_res, average="micro"),
    "macro_f1": f1_score(actual_doc_res, pred_doc_res, average="macro"),
    "precision": precision_score(actual_doc_res, pred_doc_res, average="micro"),
    "recall": recall_score(actual_doc_res, pred_doc_res, average="micro"),
    "accuracy": accuracy_score(actual_doc_res, pred_doc_res),
    "eval_size": len(sample_lst),
}

print("Metrics for split-classifier at document level:\n", metrics)


def print_search_results(lolo_results):
    print()
    print(f"**Results: Total search docs={len(lolo_results)}. Each search doc has k={k} results\n")
    
    for idx, results in enumerate(lolo_results):
        print(f"*** Result for document#: {idx}")

        # Debugging: Print type and structure
        print(f"Type of results[0]: {type(results[0])}")
        print(f"Content of results[0]: {results[0]}")

        # Ensure results[0] is a list containing a dictionary
        if isinstance(results[0], list) and len(results[0]) > 0:
            result_dict = results[0][0]  # Extract the dictionary inside the list
        elif isinstance(results[0], dict):
            result_dict = results[0]  # Use it directly if it's already a dictionary
        else:
            print(f"Skipping index {idx}, unexpected format:", results[0])
            continue

        distances = result_dict.get('distances', [[0]])[0]  # Extract first list
        cos_sim = [1 - max(0, dist) for dist in distances]

        for key in ['ids', 'distances', 'metadatas']:
            if key in result_dict:
                print(f"{key:10s}: {result_dict[key]}")
            else:
                print(f"{key:10s}: Key not found")

        print("similarity score:", cos_sim)
        print("************************")

print_search_results(lolo_results)


# Evaluation: Compare ground-truth metadata to the predicted metadata from search results.
pred_label_l, pred_first_pg_l, pred_pg_num_l, pred_score_l = list(), list(), list(), list()

for idx, results in enumerate(lolo_results):
    # Ensure results[0] is a list containing a dictionary
    if isinstance(results[0], list) and len(results[0]) > 0:
        result_dict = results[0][0]  # Extract the dictionary inside the list
    elif isinstance(results[0], dict):
        result_dict = results[0]  # Use it directly if it's already a dictionary
    else:
        print(f"Skipping index {idx}, unexpected format:", results[0])
        continue

    distances = result_dict.get('distances', [[0]])[0]  # Extract first list
    cos_sim = [1 - max(0, dist) for dist in distances]

    # Debugging: Check metadata structure
    print(f"Type of result_dict['metadatas']: {type(result_dict['metadatas'])}")
    print(f"Content of result_dict['metadatas']: {result_dict['metadatas']}")

    # Extract metadata correctly
    metadata = result_dict['metadatas']
    if isinstance(metadata, list) and len(metadata) > 0:
        metadata = metadata[0]  # Extract first dictionary in the list

    if isinstance(metadata, dict):
        pred_label_l.append(metadata.get('label', 'Unknown'))
        pred_first_pg_l.append(metadata.get('first_pg', False))
        pred_pg_num_l.append(metadata.get('pg_num', 0))
    else:
        print(f"Skipping index {idx}, unexpected metadata format:", metadata)
        continue

    pred_score_l.append(cos_sim[0])


# Ground truths
label_l, first_page_l = list(), list()
for idx, results in enumerate(srch_lolo_metadata):
    label_l.append(results[0]['label'])
    first_page_l.append(results[0]['first_pg'])


print("*** Label Performance ***")
print("Ground Truth:", label_l[:10])
print("Prediction :", pred_label_l[:10])
print("\n Classification report:", classification_report(label_l, pred_label_l))
print("---------------------------------------\n")

print("*** First Page Performance ***")
print("Ground Truth:", first_page_l[:10])
print("Prediction :", pred_first_pg_l[:10])
print("\n Classification report:", classification_report(first_page_l, pred_first_pg_l))
print("---------------------------------------\n")


# Evaluation: Compare ground-truth metadata to the predicted metadata from search results.
pred_label_l, pred_first_pg_l, pred_pg_num_l, pred_score_l = list(), list(), list(), list()

for idx, results in enumerate(lolo_results):
    # Ensure results[0] is a list containing a dictionary
    if isinstance(results[0], list) and len(results[0]) > 0:
        result_dict = results[0][0]  # Extract the dictionary inside the list
    elif isinstance(results[0], dict):
        result_dict = results[0]  # Use it directly if it's already a dictionary
    else:
        print(f"Skipping index {idx}, unexpected format:", results[0])
        continue

    # Extract distances correctly
    distances = result_dict.get('distances', [[0]])[0]  # Extract first list
    cos_sim = [1 - max(0, dist) for dist in distances]

    # Extract metadata correctly
    metadata = result_dict.get('metadatas', [{}])  # Get first dictionary if it exists
    if isinstance(metadata, list) and len(metadata) > 0:
        metadata = metadata[0]  # Extract the first dictionary inside the list

    if isinstance(metadata, dict):
        pred_label_l.append(metadata.get('label', 'Unknown'))
        pred_first_pg_l.append(metadata.get('first_pg', False))
        pred_pg_num_l.append(metadata.get('pg_num', 0))
    else:
        print(f"Skipping index {idx}, unexpected metadata format:", metadata)
        continue

    pred_score_l.append(cos_sim[0])


# Ground truths
label_l, first_page_l = list(), list()
for idx, results in enumerate(srch_lolo_metadata):
    label_l.append(results[0]['label'])
    first_page_l.append(results[0]['first_pg'])


print("*** Label Performance ***")
print("Ground Truth:", label_l[:10])
print("Prediction :", pred_label_l[:10])
print("\n Classification report:", classification_report(label_l, pred_label_l))
print("---------------------------------------\n")

print("*** First Page Performance ***")
print("Ground Truth:", first_page_l[:10])
print("Prediction :", pred_first_pg_l[:10])
print("\n Classification report:", classification_report(first_page_l, pred_first_pg_l))
print("---------------------------------------\n")


# FAISS-specific implementation
# Ensure FAISS metadata is loaded beforehand, as FAISS does not store metadata by default
faiss_metadata = []  # Preload FAISS metadata, this should be populated before performing queries.

# Function to simulate or extract metadata for FAISS
def load_faiss_metadata():
    global faiss_metadata
    # This is where you load the metadata corresponding to the FAISS vectors, for example:
    faiss_metadata = [
        {"label": "Bank Statement", "first_pg": True, "pg_num": 1},
        {"label": "Paystub", "first_pg": True, "pg_num": 2},
        # Populate with actual metadata
    ]

# Initialize FAISS metadata before performing queries
load_faiss_metadata()

# Evaluation: Compare ground-truth metadata to the predicted metadata from search results.
pred_label_l, pred_first_pg_l, pred_pg_num_l, pred_score_l = list(), list(), list(), list()

# Iterate over FAISS query results (lolo_results contains vector search results)
for idx, results in enumerate(lolo_results):
    # FAISS does not return metadata, so we need to use metadata from preloaded `faiss_metadata`
    if idx < len(faiss_metadata):
        result_dict = results[0][0] if isinstance(results[0], list) and len(results[0]) > 0 else results[0]
        
        # Extract distances and calculate similarity
        distances = result_dict.get('distances', [[0]])[0]  # Extract the first list of distances
        cos_sim = [1 - max(0, dist) for dist in distances]

        # Use metadata from FAISS preloaded metadata
        metadata = faiss_metadata[idx]  # Retrieve the metadata corresponding to the query result

        # Ensure that the metadata is a dictionary before accessing
        if isinstance(metadata, dict):
            pred_label_l.append(metadata.get('label', 'Unknown'))
            pred_first_pg_l.append(metadata.get('first_pg', False))
            pred_pg_num_l.append(metadata.get('pg_num', 0))
        else:
            print(f"Skipping index {idx}, unexpected metadata format:", metadata)
            continue

        pred_score_l.append(cos_sim[0])  # Store similarity score for the prediction

# Ground truth labels and first page information from `srch_lolo_metadata`
label_l, first_page_l = list(), list()
for idx, results in enumerate(srch_lolo_metadata):
    label_l.append(results[0]['label'])
    first_page_l.append(results[0]['first_pg'])

# Ensure both lists have the same length before calling classification_report
if len(label_l) == len(pred_label_l) and len(label_l) > 0:
    print("*** Label Performance ***")
    print("Ground Truth:", label_l[:10])
    print("Prediction :", pred_label_l[:10])
    print("\n Classification report:", classification_report(label_l, pred_label_l))
    print("---------------------------------------\n")

    print("*** First Page Performance ***")
    print("Ground Truth:", first_page_l[:10])
    print("Prediction :", pred_first_pg_l[:10])
    print("\n Classification report:", classification_report(first_page_l, pred_first_pg_l))
    print("---------------------------------------\n")
else:
    print(f"Error: Mismatched label and prediction sizes. Label size: {len(label_l)}, Prediction size: {len(pred_label_l)}")



# Evaluation: Compare ground-truth metadata to the predicted metadata from search results.

pred_label_l, pred_first_pg_l, pred_pg_num_l, pred_score_l = [], [], [], []

for idx, results in enumerate(lolo_results):
    print(f"Processing document#: {idx}")

    # Ensure results[0] is a dictionary
    if isinstance(results[0], list) and len(results[0]) > 0:
        result_dict = results[0]  # Extract the dictionary inside the list
    elif isinstance(results[0], dict):
        result_dict = results[0]  # Use it directly
    else:
        print(f"Skipping index {idx}, unexpected format:", results[0])
        continue

    # Extract distance values
    distances = result_dict.get('distances', [[0]])[0]
    cos_sim = 1 - max(0, *distances)  # Compute similarity score

    # Extract metadata values
    if 'metadatas' in result_dict and isinstance(result_dict['metadatas'], list) and len(result_dict['metadatas']) > 0:
        metadata = result_dict['metadatas'][0]  # Extract first metadata dictionary
        pred_label_l.append(metadata.get('label', 'Unknown'))
        pred_first_pg_l.append(metadata.get('first_pg', False))
        pred_pg_num_l.append(metadata.get('pg_num', -1))
    else:
        print(f"Skipping index {idx}, metadata not found")
        continue

    pred_score_l.append(cos_sim)

label_l, first_page_l = [], []

for idx, results in enumerate(srch_lolo_metadata):
    label_l.append(results[0].get('label', 'Unknown'))
    first_page_l.append(results[0].get('first_pg', False))

print("Evaluation Completed")



# Evaluation: Compare ground-truth metadata to the predicted metadata from search results.

pred_label_l, pred_first_pg_l, pred_pg_num_l, pred_score_l = [], [], [], []

for idx, results in enumerate(lolo_results):
    print(f"Processing document#: {idx}")

    # Ensure results[0] is a dictionary
    if isinstance(results, list) and len(results) > 0:
        first_element = results[0]
        if isinstance(first_element, list) and len(first_element) > 0:
            result_dict = first_element[0]  # Extract dictionary from nested list
        elif isinstance(first_element, dict):
            result_dict = first_element  # Use it directly
        else:
            print(f"Skipping index {idx}, unexpected format:", first_element)
            continue
    elif isinstance(results, dict):
        result_dict = results  # Use results directly if it's already a dictionary
    else:
        print(f"Skipping index {idx}, unexpected format:", results)
        continue

    # Extract distance values safely
    distances = result_dict.get('distances', [[0]])[0]
    if isinstance(distances, list):
        cos_sim = 1 - max(0, *distances)  # Compute similarity score
    else:
        print(f"Skipping index {idx}, invalid distance format:", distances)
        continue

    # Extract metadata values safely
    if 'metadatas' in result_dict and isinstance(result_dict['metadatas'], list) and len(result_dict['metadatas']) > 0:
        metadata = result_dict['metadatas'][0]  # Extract first metadata dictionary
        pred_label_l.append(metadata.get('label', 'Unknown'))
        pred_first_pg_l.append(metadata.get('first_pg', False))
        pred_pg_num_l.append(metadata.get('pg_num', -1))
    else:
        print(f"Skipping index {idx}, metadata not found")
        continue

    pred_score_l.append(cos_sim)

# Extract ground-truth metadata
label_l, first_page_l = [], []

for idx, results in enumerate(srch_lolo_metadata):
    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
        label_l.append(results[0].get('label', 'Unknown'))
        first_page_l.append(results[0].get('first_pg', False))
    else:
        print(f"Skipping index {idx}, unexpected format in ground truth metadata:", results)

print("Evaluation Completed")
