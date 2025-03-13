import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report
import pandas as pd
import numpy as np

# Assuming you have both classifiers already trained and saved
# document_classifier: for document type classification
# first_page_classifier: for first page detection

def evaluate_combined_classification(document_classifier, first_page_classifier, 
                                    dataloader, device, unique_labels, idx_to_label):
    """
    Evaluate document type classification and first page detection together.
    
    Args:
        document_classifier: Trained document classifier model
        first_page_classifier: Trained first page classifier model
        dataloader: DataLoader with test data
        device: Device to run inference on
        unique_labels: List of unique document labels
        idx_to_label: Dictionary mapping indices to document labels
        
    Returns:
        DataFrame with combined metrics
    """
    # Set both models to evaluation mode
    document_classifier.eval()
    first_page_classifier.eval()
    
    # Initialize lists to store predictions and ground truth
    doc_true_labels = []
    doc_pred_labels = []
    fp_true_labels = []
    fp_pred_labels = []
    
    # Store combined labels
    combined_true = []
    combined_pred = []
    
    with torch.no_grad():
        for embeddings, doc_labels, is_first_page in dataloader:
            # Move data to device
            embeddings = embeddings.to(device)
            doc_labels = doc_labels.to(device)
            is_first_page = is_first_page.to(device)
            
            # Get document type predictions
            doc_outputs = document_classifier(embeddings)
            doc_preds = torch.argmax(doc_outputs, dim=1)
            
            # Get first page predictions
            fp_outputs = first_page_classifier(embeddings)
            fp_preds = torch.argmax(fp_outputs, dim=1)
            
            # Convert to numpy arrays for easier processing
            doc_labels_np = doc_labels.cpu().numpy()
            doc_preds_np = doc_preds.cpu().numpy()
            fp_labels_np = is_first_page.cpu().numpy()
            fp_preds_np = fp_preds.cpu().numpy()
            
            # Extend lists with batch data
            doc_true_labels.extend(doc_labels_np)
            doc_pred_labels.extend(doc_preds_np)
            fp_true_labels.extend(fp_labels_np)
            fp_pred_labels.extend(fp_preds_np)
            
            # Create combined labels (document_type:is_first_page)
            for i in range(len(doc_labels_np)):
                true_doc = idx_to_label[doc_labels_np[i]]
                pred_doc = idx_to_label[doc_preds_np[i]]
                
                true_fp = "True" if fp_labels_np[i] == 1 else "False"
                pred_fp = "True" if fp_preds_np[i] == 1 else "False"
                
                combined_true.append(f"{true_doc}:{true_fp}")
                combined_pred.append(f"{pred_doc}:{pred_fp}")
    
    # Get all unique combined classes
    all_combined_classes = []
    for doc_label in unique_labels:
        all_combined_classes.append(f"{doc_label}:True")
        all_combined_classes.append(f"{doc_label}:False")
    
    # Generate classification report
    report = classification_report(combined_true, combined_pred, 
                                  labels=all_combined_classes, 
                                  output_dict=True)
    
    # Convert to DataFrame for better display
    report_df = pd.DataFrame(report).transpose()
    
    # Print basic accuracy
    accuracy = np.mean(np.array(combined_true) == np.array(combined_pred))
    print(f"Combined Accuracy: {accuracy:.4f}")
    
    return report_df

# Example usage (you'll need to adapt this to your specific code):

def generate_combined_metrics(document_classifier_path, first_page_classifier_path, 
                              test_dataset, unique_labels, label_to_idx, device):
    """
    Load both models and generate combined metrics.
    
    Args:
        document_classifier_path: Path to saved document classifier
        first_page_classifier_path: Path to saved first page classifier
        test_dataset: Dataset with test samples
        unique_labels: List of unique document labels
        label_to_idx: Dictionary mapping labels to indices
        device: Device to run inference on
    """
    # Create index to label mapping
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    # Load document classifier
    input_dim = 768  # Adjust based on your embedder
    hidden_dim = 128
    num_classes = len(unique_labels)
    
    document_classifier = MLPClassifier(input_dim, hidden_dim, num_classes)
    document_classifier.load_state_dict(torch.load(document_classifier_path))
    document_classifier = document_classifier.to(device)
    
    # Load first page classifier
    first_page_classifier = FirstPageClassifier(input_dim, hidden_dim)
    first_page_classifier.load_state_dict(torch.load(first_page_classifier_path))
    first_page_classifier = first_page_classifier.to(device)
    
    # Create DataLoader that returns document labels and is_first_page
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Generate and display combined metrics
    combined_metrics = evaluate_combined_classification(
        document_classifier, 
        first_page_classifier,
        test_loader,
        device,
        unique_labels,
        idx_to_label
    )
    
    print("\nCombined Classification Report:")
    print(combined_metrics[['precision', 'recall', 'f1-score', 'support']])
    
    # Save metrics to CSV
    combined_metrics.to_csv('combined_classification_metrics.csv')
    print("Metrics saved to combined_classification_metrics.csv")
    
    return combined_metrics

# ---- Adapting your dataset to return both document label and is_first_page ----

class CombinedDataset(Dataset):
    """
    Dataset that returns document embeddings, document labels, and is_first_page flags.
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

# Example of how to use this in your main script:
"""
# Example usage:
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load your data
    # Assuming df has 'text', 'label', and 'is_first_page' columns
    texts = df["text"].tolist()
    doc_labels = [label_to_idx[label] for label in df["label"]]
    is_first_page = df["is_first_page"].tolist()
    
    # Initialize embedder
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-mpnet-base-v2')
    
    # Create dataset
    test_dataset = CombinedDataset(texts, doc_labels, is_first_page, embedder)
    
    # Generate combined metrics
    metrics = generate_combined_metrics(
        'document_classifier.pth',
        'first_page_classifier.pth',
        test_dataset,
        unique_labels,
        label_to_idx,
        device
    )
"""



class CombinedDataset(Dataset):
    def __init__(self, texts, doc_labels, is_first_page, embedder):
        self.texts = texts
        self.doc_labels = doc_labels
        self.is_first_page = is_first_page
        self.embedder = embedder
        
        print("Computing embeddings for dataset...")
        self.embeddings = self.embed_texts(self.texts)
        
        # Add length checking
        assert len(self.embeddings) == len(self.doc_labels) == len(self.is_first_page), \
               f"Length mismatch: embeddings {len(self.embeddings)}, doc_labels {len(self.doc_labels)}, is_first_page {len(self.is_first_page)}"
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        # Add bounds checking
        if idx >= len(self.embeddings) or idx >= len(self.doc_labels) or idx >= len(self.is_first_page):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.embeddings)} elements")
        return self.embeddings[idx], self.doc_labels[idx], self.is_first_page[idx]

# Before creating the dataset, limit all arrays to the length of the shortest one
min_length = min(len(texts), len(doc_labels), len(is_first_page))
texts = texts[:min_length]
doc_labels = doc_labels[:min_length]
is_first_page = is_first_page[:min_length]

test_dataset = CombinedDataset(texts, doc_labels, is_first_page, embedder)

# Check if there's an issue in how doc_labels is being created
print(f"Original texts length: {len(texts)}")
print(f"Original doc_labels length: {len(doc_labels)}")
print(f"Original is_first_page length: {len(is_first_page)}")

# Make sure doc_labels is created from the same source as the other arrays
doc_labels = [label_to_idx[label] for label in df['label'].tolist()]
