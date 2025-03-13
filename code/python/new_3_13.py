import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader

class CombinedDataset(Dataset):
    def __init__(self, texts, doc_labels, is_first_page, embedder):
        self.texts = texts
        self.doc_labels = doc_labels
        self.is_first_page = is_first_page
        
        # Ensure all arrays have the same length
        assert len(texts) == len(doc_labels) == len(is_first_page), "All input arrays must have the same length"
        
        self.embedder = embedder
        print("Computing embeddings for dataset...")
        self.embeddings = self.embed_texts(self.texts)
        
        # Double-check that embeddings match the text count
        assert len(self.embeddings) == len(self.texts), "Embedding count doesn't match text count"
    
    def embed_texts(self, texts):
        # Compute embeddings and return as tensor
        embeddings = self.embedder.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        return embeddings
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Check index bounds to prevent the exact error we're seeing
        if idx >= len(self.texts) or idx < 0:
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.texts)}")
            
        return self.embeddings[idx], self.doc_labels[idx], self.is_first_page[idx]

def evaluate_combined_metrics(doc_classifier, fp_classifier, dataloader, device, label_to_idx):
    """
    Evaluate combined document type and first page metrics
    
    Args:
        doc_classifier: Document type classifier model
        fp_classifier: First page classifier model
        dataloader: DataLoader with combined dataset
        device: Device to run models on (cuda/cpu)
        label_to_idx: Dictionary mapping label names to indices
        
    Returns:
        Dictionary with combined metrics
    """
    doc_classifier.eval()
    fp_classifier.eval()
    
    all_doc_preds = []
    all_doc_labels = []
    all_fp_preds = []
    all_fp_labels = []
    all_texts = []
    
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    with torch.no_grad():
        for embeddings, doc_labels, fp_labels in dataloader:
            embeddings = embeddings.to(device)
            doc_labels = doc_labels.to(device)
            fp_labels = fp_labels.to(device)
            
            # Document type predictions
            doc_outputs = doc_classifier(embeddings)
            doc_preds = torch.argmax(doc_outputs, dim=1)
            
            # First page predictions
            fp_outputs = fp_classifier(embeddings)
            fp_preds = torch.argmax(fp_outputs, dim=1)
            
            all_doc_preds.extend(doc_preds.cpu().numpy())
            all_doc_labels.extend(doc_labels.cpu().numpy())
            all_fp_preds.extend(fp_preds.cpu().numpy())
            all_fp_labels.extend(fp_labels.cpu().numpy())
            
    # Convert to numpy arrays
    all_doc_preds = np.array(all_doc_preds)
    all_doc_labels = np.array(all_doc_labels)
    all_fp_preds = np.array(all_fp_preds)
    all_fp_labels = np.array(all_fp_labels)
    
    # Create combined metrics
    results = {}
    
    # For each document type
    for doc_idx, doc_label in idx_to_label.items():
        # True instances (first page)
        doc_fp_true_mask = (all_doc_labels == doc_idx) & (all_fp_labels == 1)
        if np.sum(doc_fp_true_mask) > 0:
            doc_fp_true_preds = (all_doc_preds == doc_idx) & (all_fp_preds == 1)
            
            # Calculate metrics
            true_positives = np.sum(doc_fp_true_preds & doc_fp_true_mask)
            predicted_positives = np.sum(doc_fp_true_preds)
            actual_positives = np.sum(doc_fp_true_mask)
            
            precision = true_positives / predicted_positives if predicted_positives > 0 else 0
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[f"{doc_label}:True"] = {
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": int(np.sum(doc_fp_true_mask))
            }
        else:
            results[f"{doc_label}:True"] = {
                "precision": 0,
                "recall": 0,
                "f1-score": 0,
                "support": 0
            }
        
        # False instances (not first page)
        doc_fp_false_mask = (all_doc_labels == doc_idx) & (all_fp_labels == 0)
        if np.sum(doc_fp_false_mask) > 0:
            doc_fp_false_preds = (all_doc_preds == doc_idx) & (all_fp_preds == 0)
            
            # Calculate metrics
            true_positives = np.sum(doc_fp_false_preds & doc_fp_false_mask)
            predicted_positives = np.sum(doc_fp_false_preds)
            actual_positives = np.sum(doc_fp_false_mask)
            
            precision = true_positives / predicted_positives if predicted_positives > 0 else 0
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[f"{doc_label}:False"] = {
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": int(np.sum(doc_fp_false_mask))
            }
        else:
            results[f"{doc_label}:False"] = {
                "precision": 0,
                "recall": 0,
                "f1-score": 0,
                "support": 0
            }
    
    return results

def print_combined_metrics(results):
    """Print the combined metrics in a formatted table"""
    print(f"{'Combined Label':50} {'Precision':10} {'Recall':10} {'F1-Score':10} {'Support':10}")
    print("-" * 90)
    
    for label, metrics in sorted(results.items()):
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1-score"]
        support = metrics["support"]
        
        print(f"{label:50} {precision:.2f}{'':<8} {recall:.2f}{'':<8} {f1:.2f}{'':<8} {support}")

# Example usage:
"""
# Load the trained models
doc_classifier = MLPClassifier(input_dim, hidden_dim, num_classes).to(device)
doc_classifier.load_state_dict(torch.load('document_classifier.pth'))

fp_classifier = FirstPageClassifier(input_dim, hidden_dim).to(device)
fp_classifier.load_state_dict(torch.load('first_page_classifier.pth'))

# Create a combined dataset
combined_dataset = CombinedDataset(texts, doc_labels, is_first_page, embedder)
combined_loader = DataLoader(combined_dataset, batch_size=16, shuffle=False)

# Calculate combined metrics
results = evaluate_combined_metrics(doc_classifier, fp_classifier, combined_loader, device, label_to_idx)

# Print the results
print_combined_metrics(results)
"""

# Alternative implementation if you already have the predictions
def calculate_combined_metrics_from_preds(doc_true, doc_pred, fp_true, fp_pred, idx_to_label):
    """
    Calculate combined metrics from existing predictions
    
    Args:
        doc_true: True document label indices
        doc_pred: Predicted document label indices
        fp_true: True first page labels (0 or 1)
        fp_pred: Predicted first page labels (0 or 1)
        idx_to_label: Dictionary mapping indices to label names
    
    Returns:
        Dictionary with combined metrics
    """
    # Ensure numpy arrays
    doc_true = np.array(doc_true)
    doc_pred = np.array(doc_pred)
    fp_true = np.array(fp_true)
    fp_pred = np.array(fp_pred)
    
    # Create combined metrics
    results = {}
    
    # For each document type
    for doc_idx, doc_label in idx_to_label.items():
        # True instances (first page)
        doc_fp_true_mask = (doc_true == doc_idx) & (fp_true == 1)
        if np.sum(doc_fp_true_mask) > 0:
            doc_fp_true_preds = (doc_pred == doc_idx) & (fp_pred == 1)
            
            # Calculate metrics
            true_positives = np.sum(doc_fp_true_preds & doc_fp_true_mask)
            predicted_positives = np.sum(doc_fp_true_preds)
            actual_positives = np.sum(doc_fp_true_mask)
            
            precision = true_positives / predicted_positives if predicted_positives > 0 else 0
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[f"{doc_label}:True"] = {
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": int(np.sum(doc_fp_true_mask))
            }
        else:
            results[f"{doc_label}:True"] = {
                "precision": 0,
                "recall": 0,
                "f1-score": 0,
                "support": 0
            }
        
        # False instances (not first page)
        doc_fp_false_mask = (doc_true == doc_idx) & (fp_true == 0)
        if np.sum(doc_fp_false_mask) > 0:
            doc_fp_false_preds = (doc_pred == doc_idx) & (fp_pred == 0)
            
            # Calculate metrics
            true_positives = np.sum(doc_fp_false_preds & doc_fp_false_mask)
            predicted_positives = np.sum(doc_fp_false_preds)
            actual_positives = np.sum(doc_fp_false_mask)
            
            precision = true_positives / predicted_positives if predicted_positives > 0 else 0
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[f"{doc_label}:False"] = {
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": int(np.sum(doc_fp_false_mask))
            }
        else:
            results[f"{doc_label}:False"] = {
                "precision": 0,
                "recall": 0,
                "f1-score": 0,
                "support": 0
            }
    
    return results
