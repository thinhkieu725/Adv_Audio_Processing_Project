import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from model import ClassifierModel


def audio_taxonomy_parent_map(class_id):
    """
    Maps audio subcategories to their parent categories based on the audio taxonomy.
    
    Parent Categories:
    0 - Music
    1 - Instrument Samples
    2 - Speech
    3 - Sound Effects
    4 - Soundscapes
    
    Subcategories:
    Music (0-2):
        0: Solo percussion
        1: Solo instrument
        2: Multiple instruments
    
    Instrument Samples (3-7):
        3: Percussion
        4: String
        5: Wind
        6: Piano/Keyboard instruments
        7: Synth/Electronic
    
    Speech (8-10):
        8: Solo speech
        9: Conversation/Crowd
        10: Processed/Synthetic
    
    Sound Effects (11-18):
        11: Objects/House appliances
        12: Vehicles
        13: Other mechanisms, engines, machines
        14: Animals
        15: Human sounds and actions
        16: Natural elements and explosions
        17: Experimental
        18: Electronic/Design
    
    Soundscapes (19-22):
        19: Nature
        20: Indoors
        21: Urban
        22: Synthetic/Artificial
    """
    class_to_parent = {
        # Music
        0: 0, 1: 0, 2: 0,
        # Instrument Samples
        3: 1, 4: 1, 5: 1, 6: 1, 7: 1,
        # Speech
        8: 2, 9: 2, 10: 2,
        # Sound Effects
        11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3,
        # Soundscapes
        19: 4, 20: 4, 21: 4, 22: 4,
    }
    
    return class_to_parent.get(int(class_id), int(class_id) // 5)


def evaluate_classification_model(
    y_true,
    y_pred,
    lambda_sibling,
    parent_map_func,
    top_n_confusions,
    top_n_recall_classes,
):
    """
    Comprehensive classification model evaluation with hierarchical metrics.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    lambda_sibling : float
        Weight for same-parent class pairs in hierarchical metrics
    parent_map_func : callable or None
        Function mapping class -> parent. Default: class // 5
    top_n_confusions : int
        Number of top confusion pairs to display
    top_n_recall_classes : int
        Number of best/worst recall classes to show
    
    Returns:
    --------
    dict with keys: 'accuracy', 'per_class_metrics', 'confusion_matrix', 
                    'hierarchical_metrics', 'hF_global'
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.array(sorted(set(y_true) | set(y_pred)))
    
    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    per_class = pd.DataFrame({
        "class": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }).sort_values("class").reset_index(drop=True)
    
    print("\nPer-class metrics:")
    print(per_class)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Plot confusion matrix (counts)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix (Counts)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, int(cm[i, j]),
                          ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontsize=7)
    
    plt.tight_layout()
    plt.show()
    
    # Normalized confusion matrix
    cm_row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, cm_row_sum, out=np.zeros_like(cm, dtype=float), where=cm_row_sum != 0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Oranges", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix (Row-normalized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                          ha="center", va="center",
                          color="white" if cm_norm[i, j] > 0.5 else "black",
                          fontsize=7)
    
    plt.tight_layout()
    plt.show()
    
    # Confusion analysis
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    conf_pairs = []
    for i_idx, i_cls in enumerate(labels):
        for j_idx, j_cls in enumerate(labels):
            count = off_diag[i_idx, j_idx]
            if count > 0:
                conf_pairs.append((int(count), int(i_cls), int(j_cls)))
    conf_pairs.sort(reverse=True)
    
    print(f"\nTop {top_n_confusions} confusion pairs (ground_truth -> predicted):")
    for count, i_cls, j_cls in conf_pairs[:top_n_confusions]:
        print(f"{i_cls} -> {j_cls}: {count}")
    
    worst_recall = per_class.sort_values("recall", ascending=True).head(top_n_recall_classes)
    best_recall = per_class.sort_values("recall", ascending=False).head(top_n_recall_classes)
    print(f"\nLowest-recall classes (top {top_n_recall_classes}):")
    print(worst_recall)
    print(f"Highest-recall classes (top {top_n_recall_classes}):")
    print(best_recall)
    
    # Hierarchical metrics
    if parent_map_func is None:
        parent_map_func = lambda c: int(c) // 5
    
    parent_map = {int(c): parent_map_func(int(c)) for c in labels}
    
    weights = np.zeros((len(labels), len(labels)), dtype=float)
    for i_idx, i_cls in enumerate(labels):
        for j_idx, j_cls in enumerate(labels):
            if i_cls == j_cls:
                weights[i_idx, j_idx] = 1.0
            elif parent_map[int(i_cls)] == parent_map[int(j_cls)]:
                weights[i_idx, j_idx] = lambda_sibling
    
    col_sums = cm.sum(axis=0)
    row_sums = cm.sum(axis=1)
    
    h_rows = []
    for i_idx, i_cls in enumerate(labels):
        num = 0.0
        den_p = 0.0
        den_r = 0.0
        for j_idx, _ in enumerate(labels):
            w_ij = weights[i_idx, j_idx]
            tp_ij = cm[i_idx, j_idx]
            fp_ij = col_sums[j_idx] - tp_ij
            fn_ij = row_sums[i_idx] - tp_ij
            num += w_ij * tp_ij
            den_p += w_ij * (tp_ij + fp_ij)
            den_r += w_ij * (tp_ij + fn_ij)
        h_precision = num / den_p if den_p > 0 else 0.0
        h_recall = num / den_r if den_r > 0 else 0.0
        h_f = (2 * h_precision * h_recall / (h_precision + h_recall)) if (h_precision + h_recall) > 0 else 0.0
        h_rows.append((int(i_cls), h_precision, h_recall, h_f))
    
    h_df = pd.DataFrame(h_rows, columns=["class", "hPrecision", "hRecall", "hF"])
    hF_global = h_df["hF"].mean()
    
    print(f"\nHierarchical setup: lambda_sibling={lambda_sibling}, parent_map={parent_map_func.__name__ if hasattr(parent_map_func, '__name__') else 'class//5'}")
    print(f"Global hierarchical F-score (hF_global): {hF_global:.4f}")
    print("Per-class hierarchical scores:")
    print(h_df.sort_values("class").reset_index(drop=True))
    
    return {
        "accuracy": acc,
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_norm,
        "hierarchical_metrics": h_df,
        "hF_global": hF_global,
    }


def main():
    """Main evaluation function."""
    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Process on {device}\n')

    # Initialize model
    model = ClassifierModel()
    checkpoint = torch.load("best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Get predictions and ground truth from model output
    with torch.no_grad():
        parent_pred, leaf_pred = model.predict()
    
    # Run evaluation
    results = evaluate_classification_model(
        y_true=parent_pred,
        y_pred=leaf_pred,
        lambda_sibling=0.5,
        parent_map_func=audio_taxonomy_parent_map,  
        top_n_confusions=10,
        top_n_recall_classes=5,
    )
    
    print(f"\n{'='*50}")
    print(f"Final Results:")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Global hF-score: {results['hF_global']:.4f}")


if __name__ == "__main__":
    main()
