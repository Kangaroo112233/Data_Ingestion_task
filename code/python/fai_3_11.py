def evaluate_model_binary(model, dataloader, device):
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
    
    # Convert to numpy arrays for easier manipulation
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Get multiclass metrics
    print("\nMulticlass Classification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=[idx_to_label[i] for i in range(len(unique_labels))]))
    
    # Binary metrics for each class
    print("\nBinary Classification Reports:")
    
    for class_idx, class_name in idx_to_label.items():
        print(f"\n{class_name} vs. Rest:")
        
        # Create binary labels and predictions
        binary_true = (all_labels == class_idx).astype(int)
        binary_pred = (all_preds == class_idx).astype(int)
        
        # Calculate metrics
        precision_pos = precision_score(binary_true, binary_pred, pos_label=1)
        recall_pos = recall_score(binary_true, binary_pred, pos_label=1)
        f1_pos = f1_score(binary_true, binary_pred, pos_label=1)
        support_pos = np.sum(binary_true == 1)
        
        precision_neg = precision_score(binary_true, binary_pred, pos_label=0)
        recall_neg = recall_score(binary_true, binary_pred, pos_label=0)
        f1_neg = f1_score(binary_true, binary_pred, pos_label=0)
        support_neg = np.sum(binary_true == 0)
        
        # Print report
        print(f"{'':15} {'precision':10} {'recall':10} {'f1-score':10} {'support':10}")
        print(f"{class_name+' (True)':15} {precision_pos:.2f}{'':<8} {recall_pos:.2f}{'':<8} {f1_pos:.2f}{'':<8} {support_pos}")
        print(f"{class_name+' (False)':15} {precision_neg:.2f}{'':<8} {recall_neg:.2f}{'':<8} {f1_neg:.2f}{'':<8} {support_neg}")


from sklearn.metrics import precision_score, recall_score, f1_score


true_labels, pred_labels = evaluate_model_binary(classifier, test_loader, device)




def evaluate_model_binary(model, dataloader, device):
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
    
    # Convert to numpy arrays for easier manipulation
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Get multiclass metrics
    print("\nMulticlass Classification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=[idx_to_label[i] for i in range(len(unique_labels))]))
    
    # Binary metrics for each class
    print("\nBinary Classification Reports:")
    
    for class_idx, class_name in idx_to_label.items():
        print(f"\n{class_name} vs. Rest:")
        
        # Create binary labels and predictions
        binary_true = (all_labels == class_idx).astype(int)
        binary_pred = (all_preds == class_idx).astype(int)
        
        # Calculate metrics
        precision_pos = precision_score(binary_true, binary_pred, pos_label=1)
        recall_pos = recall_score(binary_true, binary_pred, pos_label=1)
        f1_pos = f1_score(binary_true, binary_pred, pos_label=1)
        support_pos = np.sum(binary_true == 1)
        
        precision_neg = precision_score(binary_true, binary_pred, pos_label=0)
        recall_neg = recall_score(binary_true, binary_pred, pos_label=0)
        f1_neg = f1_score(binary_true, binary_pred, pos_label=0)
        support_neg = np.sum(binary_true == 0)
        
        # Print report
        print(f"{'':15} {'precision':10} {'recall':10} {'f1-score':10} {'support':10}")
        print(f"{class_name+' (True)':15} {precision_pos:.2f}{'':<8} {recall_pos:.2f}{'':<8} {f1_pos:.2f}{'':<8} {support_pos}")
        print(f"{class_name+' (False)':15} {precision_neg:.2f}{'':<8} {recall_neg:.2f}{'':<8} {f1_neg:.2f}{'':<8} {support_neg}")
    
    # Return the labels and predictions
    return all_labels, all_preds

true_labels, pred_labels = evaluate_model_binary(classifier, test_loader, device)



import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# First Page Dataset class
class FirstPageEmbeddingDataset(Dataset):
    """
    texts: list of document texts.
    is_first_page: list of boolean labels (1 for first page, 0 for non-first page).
    embedder: a SentenceTransformer model instance.
    """
    def __init__(self, texts, is_first_page, embedder):
        self.texts = texts
        self.is_first_page = is_first_page
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
        return self.embeddings[idx], self.is_first_page[idx]

# First Page Classifier Network
class FirstPageClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FirstPageClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)  # Binary classification: first page or not
        )
    
    def forward(self, x):
        return self.model(x)

# Training function
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

# Evaluation function with binary metrics
def evaluate_model_binary(model, dataloader, device):
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
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Calculate standard classification report
    print("\nFirst Page Classification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=["Not First Page", "First Page"]))
    
    # Binary metrics for first page vs not first page
    print("\nBinary Classification Reports:")
    
    # First Page metrics
    print("\nFirst Page vs. Rest:")
    
    # Create binary true/false metrics for "First Page"
    binary_true = all_labels
    binary_pred = all_preds
    
    # Calculate metrics for First Page (class 1)
    precision_true = precision_score(binary_true, binary_pred, pos_label=1)
    recall_true = recall_score(binary_true, binary_pred, pos_label=1)
    f1_true = f1_score(binary_true, binary_pred, pos_label=1)
    support_true = np.sum(binary_true == 1)
    
    # Calculate metrics for Not First Page (class 0)
    precision_false = precision_score(binary_true, binary_pred, pos_label=0)
    recall_false = recall_score(binary_true, binary_pred, pos_label=0)
    f1_false = f1_score(binary_true, binary_pred, pos_label=0)
    support_false = np.sum(binary_true == 0)
    
    # Print report
    print(f"{'':20} {'precision':10} {'recall':10} {'f1-score':10} {'support':10}")
    print(f"{'First Page (True)':20} {precision_true:.2f}{'':<8} {recall_true:.2f}{'':<8} {f1_true:.2f}{'':<8} {support_true}")
    print(f"{'Not First Page (False)':20} {precision_false:.2f}{'':<8} {recall_false:.2f}{'':<8} {f1_false:.2f}{'':<8} {support_false}")
    
    return all_labels, all_preds

# Main execution function
def main():
    # Load and prepare data
    # Assuming df has columns: 'text', 'is_first_page' (1 for first page, 0 for non-first page)
    df = pd.read_csv('/path/to/your/dataset.csv')
    
    # Check dataframe columns
    print("DataFrame columns:", df.columns)
    print("Total samples:", len(df))
    
    # Ensure 'is_first_page' is integer type
    df['is_first_page'] = df['is_first_page'].astype(int)
    
    # Get distribution of classes
    print("\nClass distribution:")
    print(df['is_first_page'].value_counts())
    
    # Get texts and labels
    texts = df["text"].tolist()
    is_first_page = df["is_first_page"].tolist()
    
    # Split into training and test sets (stratified by first page label)
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts, is_first_page, test_size=0.3, random_state=42, stratify=is_first_page
    )
    
    # Load the sentence transformer model (use the same embedder as your document classifier)
    # This assumes you have a sentence transformer model already defined
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-mpnet-base-v2')  # Use the same model as before
    
    # Create PyTorch Datasets and DataLoaders
    train_dataset = FirstPageEmbeddingDataset(texts_train, labels_train, embedder)
    test_dataset = FirstPageEmbeddingDataset(texts_test, labels_test, embedder)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Define and initialize the classifier
    input_dim = embedder.get_sentence_embedding_dimension()  # e.g. 768 for BERT variants
    hidden_dim = 128
    first_page_classifier = FirstPageClassifier(input_dim, hidden_dim).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(first_page_classifier.parameters(), lr=1e-3)
    
    # Train the classifier
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train_model(first_page_classifier, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    # Evaluate the classifier
    print("\nEvaluation on test set:")
    true_labels, pred_labels = evaluate_model_binary(first_page_classifier, test_loader, device)
    
    # Save the model
    torch.save(first_page_classifier.state_dict(), 'first_page_classifier.pth')
    print("\nModel saved as 'first_page_classifier.pth'")

if __name__ == "__main__":
    main()
