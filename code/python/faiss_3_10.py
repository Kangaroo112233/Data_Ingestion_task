import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =============================================================================
# Part A: FAISS Vector DB and Helper Functions (Optional for retrieval)
# =============================================================================

def load_model_on_device(embedding_model_name, device='cuda', gpu_id='0'):
    """
    Load a SentenceTransformer embedding model onto the specified device.
    """
    if device == 'cuda' and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        device_with_id = device + ':' + gpu_id
    elif device == 'cpu':
        device_with_id = device
    else:
        raise ValueError("Unknown device: " + device)
    
    # Load the SentenceTransformer model.
    embedding_model = SentenceTransformer(embedding_model_name, device=device_with_id)
    embedding_model.to(device_with_id)
    print("Loaded embedding model '{}' on device: {}".format(embedding_model_name, embedding_model.device))
    return embedding_model

class MyEmbeddingFunction:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def __call__(self, input_texts):
        # Encode a list of texts into embeddings.
        embeddings = self.embedding_model.encode(sentences=input_texts, show_progress_bar=False)
        return embeddings.tolist()

class FAISSVectorDB:
    def __init__(self, dimension, similarity_algo='cosine', custom_embedding_function=None, db_name="document_classification"):
        self.dimension = dimension
        self.similarity_algo = similarity_algo.lower()
        self.custom_embedding_function = custom_embedding_function
        self.db_name = db_name

        # Choose the FAISS index based on the similarity measure.
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

        # Dictionaries to store additional data.
        self.id_to_doc = {}
        self.id_to_meta = {}
        self.id_counter = 0

    def add(self, ids, embeddings=None, metadatas=None, documents=None):
        num_items = len(ids)
        for i in range(num_items):
            if embeddings is None:
                if self.custom_embedding_function is not None:
                    emb = self.custom_embedding_function([documents[i]])
                    embedding = np.array(emb[0])
                else:
                    raise ValueError("No embeddings provided and no custom embedding function available")
            else:
                embedding = np.array(embeddings[i])
            if self.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            embedding = np.expand_dims(embedding.astype('float32'), axis=0)
            self.index.add(embedding)
            self.id_to_doc[self.id_counter] = documents[i] if documents is not None else ""
            self.id_to_meta[self.id_counter] = metadatas[i] if metadatas is not None else {}
            self.id_counter += 1

    def query(self, query_embeddings=None, query_texts=None, n_results=1, where_document=None):
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

def initialize_FAISS(embedding_model, db_name="document_classification", similarity_algo='cosine', custom_embedding_function=None):
    dimension = embedding_model.get_sentence_embedding_dimension()
    vector_db = FAISSVectorDB(dimension=dimension,
                              similarity_algo=similarity_algo,
                              custom_embedding_function=custom_embedding_function,
                              db_name=db_name)
    print(f'FAISS Vector DB: {vector_db.db_name} created successfully')
    if custom_embedding_function:
        new_name = vector_db.db_name + '_emb'
        vector_db.db_name = new_name
        print('Vector DB name updated to:', new_name)
    return vector_db

# =============================================================================
# Part B: Classification Pipeline Using a BERT-Based Embedding Model
# =============================================================================

# 1. Dataset that precomputes embeddings.
class EmbeddingDataset(Dataset):
    def __init__(self, texts, labels, embedder):
        """
        texts: list of document texts.
        labels: list of integer labels.
        embedder: a SentenceTransformer model instance.
        """
        self.texts = texts
        self.labels = labels
        self.embedder = embedder
        print("Computing embeddings for dataset...")
        self.embeddings = self.embed_texts(self.texts)
        
    def embed_texts(self, texts):
        # Compute embeddings and return as tensor.
        embeddings = self.embedder.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        return embeddings
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# 2. Simple MLP Classification Head.
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# 3. Training and Evaluation Functions.
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for embeddings, labels in dataloader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * embeddings.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            outputs = model(embeddings)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

# =============================================================================
# Part C: Main Execution
# =============================================================================

def main():
    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Use a BERT-based SentenceTransformer model.
    # Here we use "all-mpnet-base-v2" (a high-performing Sentence-BERT variant).
    embedding_model_name = "all-mpnet-base-v2"
    embedder = load_model_on_device(embedding_model_name, device=device, gpu_id='0')
    
    # (Optional) Bind the embedding model to a FAISS Vector DB for search tasks.
    my_embedding_function = MyEmbeddingFunction(embedder)
    vector_db = initialize_FAISS(embedder, db_name="document_classification", similarity_algo='cosine', custom_embedding_function=my_embedding_function)
    
    # Load dataset.
    # Assumes a CSV file with at least "text" and "label" columns.
    data_file = "clf_200_per_doc_type_0102.csv"
    df = pd.read_csv(data_file)
    df = df.dropna(subset=["text", "label"])
    
    # For demonstration, sample a subset.
    df = df.sample(100, random_state=42)
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    
    # Create label mappings.
    unique_labels = sorted(list(set(labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    print("Unique labels:", unique_labels)
    labels_idx = [label_to_idx[label] for label in labels]
    
    # Split into training and test sets.
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts, labels_idx, test_size=0.2, random_state=42, stratify=labels_idx
    )
    
    # Create PyTorch Datasets and DataLoaders.
    train_dataset = EmbeddingDataset(texts_train, labels_train, embedder)
    test_dataset = EmbeddingDataset(texts_test, labels_test, embedder)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Define and initialize the classifier.
    input_dim = embedder.get_sentence_embedding_dimension()  # e.g. 768 for many BERT variants
    hidden_dim = 128
    num_classes = len(unique_labels)
    classifier = MLPClassifier(input_dim, hidden_dim, num_classes).to(device)
    
    # Define loss and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    
    # Train the classifier.
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train_model(classifier, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    # Evaluate the classifier.
    true_labels, pred_labels = evaluate_model(classifier, test_loader, device)
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=[idx_to_label[i] for i in range(num_classes)]))
    
    # (Optional) Store document embeddings into FAISS for retrieval.
    # Here we add all training documents.
    for idx, text in enumerate(texts_train):
        metadata = {"label": labels_train[idx]}
        vector_db.add(ids=[str(idx)], embeddings=None, metadatas=[metadata], documents=[text])
    print("Total vectors in FAISS DB:", vector_db.count())
    
    # (Optional) Example FAISS query.
    query_text = texts_train[0]
    results = vector_db.query(query_texts=[query_text], n_results=3)
    print("\nFAISS Query Results for example text:")
    print(results)
    
    # Save the trained classifier.
    torch.save(classifier.state_dict(), "mlp_classifier_bert.pth")
    print("Classifier saved to mlp_classifier_bert.pth")

if __name__ == '__main__':
    main()
