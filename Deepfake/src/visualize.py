"""Visualization helpers extracted from `detection.py` training/evaluation sections.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from . import config


def plot_confusion_matrix(y_true, y_pred, class_names=('real', 'fake')):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={'size':16})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    return plt.gcf()  # Return figure for saving


def plot_confusion_matrix_and_save(y_true, y_pred, save_path=None, class_names=('real', 'fake')):
    """Plot confusion matrix and save to file."""
    if save_path is None:
        save_path = os.path.join(str(config.EVALUATION_DIR), 'confusion_matrix.png')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={'size':16})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Confusion matrix saved to {save_path}')
    plt.close()


def plot_roc(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    return plt.gcf(), roc_auc  # Return figure and AUC


def plot_roc_and_save(y_true, y_probs, save_path=None):
    """Plot ROC curve and save to file."""
    if save_path is None:
        save_path = os.path.join(str(config.EVALUATION_DIR), 'roc_curve.png')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'ROC curve saved to {save_path}')
    plt.close()
    
    return roc_auc


def show_sample_predictions(test_paths, y_true, y_probs, y_pred_classes, num_samples=10, cmap='viridis'):
    import numpy as _np
    plt.figure(figsize=(15,10))
    num_samples_to_show = min(num_samples, len(y_true))
    sample_indices = _np.random.choice(len(y_true), num_samples_to_show, replace=False)

    for i, idx in enumerate(sample_indices):
        path = test_paths[idx]
        spec = np.load(path)
        if spec.ndim == 3:
            spec_disp = spec[:, :, 0]
        else:
            spec_disp = spec

        plt.subplot(2, 5, i+1)
        plt.imshow(spec_disp, cmap=cmap)
        true_label = 'real' if int(y_true[idx]) == 0 else 'fake'
        pred_prob = float(y_probs[idx])
        pred_label = 'real' if int(y_pred_classes[idx]) == 0 else 'fake'
        title_color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label} ({pred_prob:.2f})", color=title_color, fontsize=10)
        plt.axis('off')

    plt.suptitle('Sample Test Predictions', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return plt.gcf()  # Return figure for saving


def show_sample_predictions_and_save(test_paths, y_true, y_probs, y_pred_classes, save_path=None, num_samples=10, cmap='viridis'):
    """Show sample predictions and save to file."""
    if save_path is None:
        save_path = os.path.join(str(config.EVALUATION_DIR), 'sample_predictions.png')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    import numpy as _np
    plt.figure(figsize=(15,10))
    num_samples_to_show = min(num_samples, len(y_true))
    sample_indices = _np.random.choice(len(y_true), num_samples_to_show, replace=False)

    for i, idx in enumerate(sample_indices):
        path = test_paths[idx]
        spec = np.load(path)
        if spec.ndim == 3:
            spec_disp = spec[:, :, 0]
        else:
            spec_disp = spec

        plt.subplot(2, 5, i+1)
        plt.imshow(spec_disp, cmap=cmap)
        true_label = 'real' if int(y_true[idx]) == 0 else 'fake'
        pred_prob = float(y_probs[idx])
        pred_label = 'real' if int(y_pred_classes[idx]) == 0 else 'fake'
        title_color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label} ({pred_prob:.2f})", color=title_color, fontsize=10)
        plt.axis('off')

    plt.suptitle('Sample Test Predictions', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Sample predictions saved to {save_path}')
    plt.close()


def save_classification_report(y_true, y_pred_classes, save_path=None, class_names=('real', 'fake')):
    """Generate and save classification report as JSON and text file."""
    if save_path is None:
        save_path = os.path.join(str(config.EVALUATION_DIR), 'classification_report.json')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    report_dict = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    
    # Save as JSON
    json_path = save_path if save_path.endswith('.json') else save_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f'Classification report (JSON) saved to {json_path}')
    
    # Save as text
    text_path = save_path if save_path.endswith('.txt') else json_path.replace('.json', '.txt')
    report_text = classification_report(y_true, y_pred_classes, target_names=class_names)
    with open(text_path, 'w') as f:
        f.write(report_text)
    print(f'Classification report (text) saved to {text_path}')
    
    return report_dict


def save_evaluation_summary(y_true, y_probs, y_pred_classes, loss, accuracy, roc_auc, save_path=None):
    """Save comprehensive evaluation summary as JSON."""
    if save_path is None:
        save_path = os.path.join(str(config.EVALUATION_DIR), 'evaluation_summary.json')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    summary = {
        'test_accuracy': float(accuracy),
        'test_loss': float(loss),
        'roc_auc': float(roc_auc),
        'total_samples': int(len(y_true)),
        'true_positives': int(np.sum((y_true == 1) & (y_pred_classes == 1))),
        'true_negatives': int(np.sum((y_true == 0) & (y_pred_classes == 0))),
        'false_positives': int(np.sum((y_true == 0) & (y_pred_classes == 1))),
        'false_negatives': int(np.sum((y_true == 1) & (y_pred_classes == 0))),
    }
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Evaluation summary saved to {save_path}')
    
    return summary
