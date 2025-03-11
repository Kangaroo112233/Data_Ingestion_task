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
