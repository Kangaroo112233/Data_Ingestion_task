import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###########################################
# Dataset Class for Combined Classification
###########################################

class CombinedEmbeddingDataset(Dataset):
    """
    Dataset for combined document type and first page classification
    
    texts: list of document texts
    doc_labels: list of document type labels (e.g., 'Bank Statement', 'Paystub')
    is_first_page: list of boolean labels (1 for first page, 0 for non-first page)
    embedder: a SentenceTransformer model instance
    """
    def __init__(self, texts, doc_labels, is_first_page, embedder):
        self.texts = texts
        self.doc_labels = doc_labels
        self.is_first_page = is_first_page
        self.embedder = embedder
        
        print("Computing embeddings for dataset...")
        self.embeddings = self.embed_texts(self.texts)
        
    def embed_texts(self, texts):
        # Compute embeddings and return as tensor
        embeddings = self.embedder.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        return embeddings
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.doc_labels[idx], self.is_first_page[idx]


###########################################
# Model Architecture for Combined Classification
###########################################

class CombinedClassifier(nn.Module):
    """
    Neural network for both document type and first page classification
    """
    def __init__(self, input_dim, hidden_dim, num_doc_classes):
        super(CombinedClassifier, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Document type specific layers
        self.doc_classifier = nn.Linear(hidden_dim, num_doc_classes)
        
        # First page specific layers
        self.fp_classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        shared_features = self.shared(x)
        
        # Document type classification output
        doc_output = self.doc_classifier(shared_features)
        
        # First page classification output
        fp_output = self.fp_classifier(shared_features)
        
        return doc_output, fp_output


###########################################
# Training and Evaluation Functions
###########################################

def train_combined_model(model, dataloader, doc_criterion, fp_criterion, optimizer, device):
    """
    Training function for combined classifier
    """
    model.train()
    running_doc_loss = 0.0
    running_fp_loss = 0.0
    
    for embeddings, doc_labels, fp_labels in dataloader:
        embeddings = embeddings.to(device)
        doc_labels = doc_labels.to(device)
        fp_labels = fp_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        doc_outputs, fp_outputs = model(embeddings)
        
        # Calculate losses
        doc_loss = doc_criterion(doc_outputs, doc_labels)
        fp_loss = fp_criterion(fp_outputs, fp_labels)
        
        # Total loss is the sum of document and first page losses
        total_loss = doc_loss + fp_loss
        
        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()
        
        # Update running losses
        running_doc_loss += doc_loss.item() * embeddings.size(0)
        running_fp_loss += fp_loss.item() * embeddings.size(0)
    
    # Calculate epoch losses
    epoch_doc_loss = running_doc_loss / len(dataloader.dataset)
    epoch_fp_loss = running_fp_loss / len(dataloader.dataset)
    epoch_total_loss = epoch_doc_loss + epoch_fp_loss
    
    return epoch_doc_loss, epoch_fp_loss, epoch_total_loss


def evaluate_combined_model(model, dataloader, device, doc_idx_to_label):
    """
    Evaluation function for combined classifier that produces combined metrics
    """
    model.eval()
    all_doc_preds = []
    all_doc_labels = []
    all_fp_preds = []
    all_fp_labels = []
    
    with torch.no_grad():
        for embeddings, doc_labels, fp_labels in dataloader:
            embeddings = embeddings.to(device)
            doc_labels = doc_labels.to(device)
            fp_labels = fp_labels.to(device)
            
            # Forward pass
            doc_outputs, fp_outputs = model(embeddings)
            
            # Get predictions
            doc_preds = torch.argmax(doc_outputs, dim=1)
            fp_preds = torch.argmax(fp_outputs, dim=1)
            
            # Store predictions and labels
            all_doc_preds.extend(doc_preds.cpu().numpy())
            all_doc_labels.extend(doc_labels.cpu().numpy())
            all_fp_preds.extend(fp_preds.cpu().numpy())
            all_fp_labels.extend(fp_labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_doc_preds = np.array(all_doc_preds)
    all_doc_labels = np.array(all_doc_labels)
    all_fp_preds = np.array(all_fp_preds)
    all_fp_labels = np.array(all_fp_labels)
    
    # Evaluate document classification
    print("\nDocument Classification Report:")
    doc_target_names = [doc_idx_to_label[i] for i in range(len(doc_idx_to_label))]
    print(classification_report(all_doc_labels, all_doc_preds, target_names=doc_target_names))
    
    # Evaluate first page classification
    print("\nFirst Page Classification Report:")
    print(classification_report(all_fp_labels, all_fp_preds, target_names=["Not First Page", "First Page"]))
    
    # Create combined labels and predictions
    # Format: "<document_type>:<is_first_page>"
    combined_true_labels = []
    combined_pred_labels = []
    
    for i in range(len(all_doc_labels)):
        doc_true = doc_idx_to_label[all_doc_labels[i]]
        doc_pred = doc_idx_to_label[all_doc_preds[i]]
        
        fp_true = "True" if all_fp_labels[i] == 1 else "False"
        fp_pred = "True" if all_fp_preds[i] == 1 else "False"
        
        combined_true = f"{doc_true}:{fp_true}"
        combined_pred = f"{doc_pred}:{fp_pred}"
        
        combined_true_labels.append(combined_true)
        combined_pred_labels.append(combined_pred)
    
    # Get unique combined labels
    unique_combined_labels = sorted(list(set(combined_true_labels + combined_pred_labels)))
    
    # Create classification report for combined predictions
    print("\nCombined Classification Report:")
    combined_report = classification_report(
        combined_true_labels, 
        combined_pred_labels, 
        labels=unique_combined_labels,
        zero_division=0
    )
    print(combined_report)
    
    # Calculate custom binary metrics for each combined class
    print("\nDetailed Binary Metrics:")
    print(f"{'Class':30} {'precision':10} {'recall':10} {'f1-score':10} {'support':10}")
    
    for label in unique_combined_labels:
        # Convert to binary classification problem for this specific label
        binary_true = np.array([1 if l == label else 0 for l in combined_true_labels])
        binary_pred = np.array([1 if l == label else 0 for l in combined_pred_labels])
        
        # Check if this label exists in true labels
        if np.sum(binary_true) > 0:
            precision = precision_score(binary_true, binary_pred, zero_division=0)
            recall = recall_score(binary_true, binary_pred, zero_division=0)
            f1 = f1_score(binary_true, binary_pred, zero_division=0)
            support = np.sum(binary_true)
            
            print(f"{label:30} {precision:.2f}{'':<8} {recall:.2f}{'':<8} {f1:.2f}{'':<8} {support}")
    
    # Calculate overall accuracy
    accuracy = np.mean(np.array(combined_true_labels) == np.array(combined_pred_labels))
    print(f"\nOverall accuracy: {accuracy:.2f}")
    
    return all_doc_labels, all_doc_preds, all_fp_labels, all_fp_preds, combined_true_labels, combined_pred_labels


###########################################
# Main Function
###########################################

def main():
    # Load data
    # This assumes you have a CSV with 'text', 'label', and 'is_first_page' columns
    df = pd.read_csv('/path/to/your/combined_dataset.csv')
    
    # Check dataframe
    print("DataFrame columns:", df.columns)
    print("Total samples:", len(df))
    
    # Ensure 'is_first_page' is integer type
    df['is_first_page'] = df['is_first_page'].astype(int)
    
    # Get document label distribution
    print("\nDocument type distribution:")
    print(df['label'].value_counts())
    
    # Get first page distribution
    print("\nFirst page distribution:")
    print(df['is_first_page'].value_counts())
    
    # Create label mappings for document types
    unique_doc_labels = sorted(df['label'].unique())
    doc_label_to_idx = {label: idx for idx, label in enumerate(unique_doc_labels)}
    doc_idx_to_label = {idx: label for label, idx in doc_label_to_idx.items()}
    
    # Convert document labels to indices
    df['doc_label_idx'] = df['label'].map(doc_label_to_idx)
    
    # Prepare data
    texts = df["text"].tolist()
    doc_labels = df["doc_label_idx"].tolist()
    is_first_page = df["is_first_page"].tolist()
    
    # Split data
    X_train, X_test, doc_train, doc_test, fp_train, fp_test = train_test_split(
        texts, doc_labels, is_first_page, 
        test_size=0.3, random_state=42, 
        stratify=df[['doc_label_idx', 'is_first_page']]  # Stratify by both labels
    )
    
    # Load embedding model
    embedder = SentenceTransformer('all-mpnet-base-v2')
    
    # Create datasets and dataloaders
    train_dataset = CombinedEmbeddingDataset(X_train, doc_train, fp_train, embedder)
    test_dataset = CombinedEmbeddingDataset(X_test, doc_test, fp_test, embedder)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize the combined model
    input_dim = embedder.get_sentence_embedding_dimension()
    hidden_dim = 128
    num_doc_classes = len(unique_doc_labels)
    
    combined_model = CombinedClassifier(input_dim, hidden_dim, num_doc_classes).to(device)
    
    # Define loss and optimizer
    doc_criterion = nn.CrossEntropyLoss()
    fp_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(combined_model.parameters(), lr=1e-3)
    
    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        doc_loss, fp_loss, total_loss = train_combined_model(
            combined_model, train_loader, doc_criterion, fp_criterion, optimizer, device
        )
        print(f"Epoch {epoch+1}/{num_epochs}, Doc Loss: {doc_loss:.4f}, FP Loss: {fp_loss:.4f}, Total Loss: {total_loss:.4f}")
    
    # Evaluate the model
    print("\nEvaluation on test set:")
    doc_labels, doc_preds, fp_labels, fp_preds, combined_true, combined_pred = evaluate_combined_model(
        combined_model, test_loader, device, doc_idx_to_label
    )
    
    # Save the model
    torch.save(combined_model.state_dict(), 'combined_classifier.pth')
    print("\nModel saved as 'combined_classifier.pth'")


###########################################
# Inference Function
###########################################

def predict_document(text, model, embedder, doc_idx_to_label, device):
    """
    Predict document type and first page status for a single document
    """
    # Encode the text
    embedding = embedder.encode(text, convert_to_tensor=True)
    embedding = embedding.unsqueeze(0).to(device)  # Add batch dimension
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        doc_output, fp_output = model(embedding)
        
        # Get document type prediction
        doc_probs = torch.softmax(doc_output, dim=1)
        doc_pred = torch.argmax(doc_probs, dim=1).item()
        doc_confidence = doc_probs[0][doc_pred].item()
        
        # Get first page prediction
        fp_probs = torch.softmax(fp_output, dim=1)
        fp_pred = torch.argmax(fp_probs, dim=1).item()
        fp_confidence = fp_probs[0][fp_pred].item()
    
    # Convert predictions to human-readable form
    document_type = doc_idx_to_label[doc_pred]
    is_first_page = bool(fp_pred)
    
    # Create combined label
    combined_label = f"{document_type}:{is_first_page}"
    
    return {
        'document_type': document_type,
        'document_type_confidence': doc_confidence,
        'is_first_page': is_first_page,
        'first_page_confidence': fp_confidence,
        'combined_label': combined_label
    }


###########################################
# Entry Point
###########################################

if __name__ == "__main__":
    main()
    
    # Example of loading and using the model for inference
    """
    # Load model and make predictions
    embedder = SentenceTransformer('all-mpnet-base-v2')
    
    input_dim = embedder.get_sentence_embedding_dimension()
    hidden_dim = 128
    num_doc_classes = 4  # Update based on your unique document types
    
    # Load the model
    combined_model = CombinedClassifier(input_dim, hidden_dim, num_doc_classes).to(device)
    combined_model.load_state_dict(torch.load('combined_classifier.pth'))
    
    # Define label mapping
    doc_idx_to_label = {0: 'Bank Statement', 1: 'Paystub', 2: 'W2', 3: 'other'}
    
    # Example prediction
    sample_text = "This is a bank statement for account #12345..."
    result = predict_document(sample_text, combined_model, embedder, doc_idx_to_label, device)
    
    print(f"Document Type: {result['document_type']} (Confidence: {result['document_type_confidence']:.2f})")
    print(f"Is First Page: {result['is_first_page']} (Confidence: {result['first_page_confidence']:.2f})")
    print(f"Combined Label: {result['combined_label']}")
    """
