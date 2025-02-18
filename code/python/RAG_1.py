pip install torch transformers sentence-transformers chromadb flask

def chunk_text(text, max_words=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # move `overlap` words back for overlap
        if start < 0 or start >= len(words):
            break
    return chunks

# Example usage:
document_text = open("my_long_document.txt").read()
chunks = chunk_text(document_text)
print(f"Created {len(chunks)} chunks from the document.")

from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model for embeddings
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# Encode the chunks into embeddings
embeddings = embed_model.encode(chunks, show_progress_bar=True)
print(f"Generated {len(embeddings)} embedding vectors of dimension {len(embeddings[0])}.")

import chromadb
from chromadb.config import Settings

# Initialize ChromaDB in-memory (you can specify a persist_directory in Settings for persistence)
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=None))
# Create a collection (like a table) for our documents
collection = client.create_collection(name="knowledge_base")

# Add documents (chunks) to the collection with their embeddings
# We also provide unique IDs for each and optional metadata (e.g., source or document name)
ids = [f"doc_{i}" for i in range(len(chunks))]
metadatas = [{"source": "my_document"} for _ in chunks]  # example metadata
collection.add(
    documents=chunks, 
    embeddings=embeddings.tolist(),  # ensure embeddings are in list form (if using NumPy array)
    ids=ids,
    metadatas=metadatas
)
print(f"Added {collection.count()} documents to ChromaDB collection.")

def embed_query(query):
    return embed_model.encode(query)


import nltk
from nltk.corpus import wordnet

def expand_query(query):
    words = query.split()
    expanded = []
    for w in words:
        expanded.append(w)
        # add first synonym of each word if available
        synsets = wordnet.synsets(w)
        if synsets:
            # take the first synonym lemma that is not the word itself
            for lemma in synsets[0].lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != w.lower():
                    expanded.append(synonym)
                    break
    return " ".join(expanded)

# Example usage:
user_query = "What is the capital of France?"
enhanced_query = expand_query(user_query)
print("Expanded query:", enhanced_query)


user_query = "What is the capital of France?"
# Optionally enhance the query
enhanced_query = expand_query(user_query)
query_embedding = embed_query(enhanced_query)


# Perform similarity search in ChromaDB
results = collection.query(
    query_embeddings=[query_embedding], 
    n_results=3, 
    include=["documents", "distances"]
)
top_docs = results["documents"][0]  # list of top 3 document strings
top_scores = results["distances"][0]  # distances (lower is more similar)
for i, doc in enumerate(top_docs):
    print(f"Result {i+1}: {doc[:100]}... (distance: {top_scores[i]:.4f})")


from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.1-8B-Instruct"
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# Load the model in half-precision and map it to available devices (GPU if available)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",       # Automatically distribute across GPU(s) or CPU
    torch_dtype=torch.float16  # Load model weights in FP16 for efficiency
)
model.eval()  # set model to evaluation mode (disable training layers like dropout)



[Context]
Question: [user question]
Answer:


# Join top retrieved docs into one context string
context_text = "\n\n".join(top_docs)
# Construct the final prompt for the LLM
prompt = f"{context_text}\n\nQuestion: {user_query}\nAnswer:"
inputs = tokenizer(prompt, return_tensors='pt')
# Move input to model device (e.g., GPU)
inputs = {k: v.to(model.device) for k,v in inputs.items()}


# Generate answer from the model
output = model.generate(
    **inputs, 
    max_new_tokens=200,      # limit the length of the generated answer
    temperature=0.7,         # a mild randomness in generation; lower for more deterministic output
    top_p=0.95,              # top-p sampling for more diverse results
    eos_token_id=tokenizer.eos_token_id
)
# Decode the generated tokens to text
answer = tokenizer.decode(output[0], skip_special_tokens=True)
# Extract the answer portion (everything after "Answer:")
answer = answer.split("Answer:")[-1].strip()
print("Answer:", answer)


