import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from Model import ResNet18FineTune
from datasetfor2d73 import get_all_dataloaders

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ResNet18FineTune(num_classes=2)
model.load_state_dict(torch.load("best_model_val_loss.pth"))  # Load best model weights
model.to(device)
model.eval()

# Load test dataset
root_folder = "/path/to/your/dataset"
train_loader, val_loader = get_all_dataloaders(root_folder, batch_size=32)

# Store test results
test_labels = []
test_preds = []
test_probs = []
patient_predictions = defaultdict(list)  # Store patient-level predictions
target_labels = {}  # Store true patient labels

# Perform testing
with torch.no_grad():
    for images, labels, filenames in val_loader:  # Ensure Dataset returns filenames
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1
        _, predicted = torch.max(outputs, 1)

        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(predicted.cpu().numpy())
        test_probs.extend(probs.cpu().numpy())

        # Record patient-level predictions
        for filename, true_label, pred_prob in zip(filenames, labels.cpu().numpy(), probs.cpu().numpy()):
            patient_id = filename.split('_')[0]  # Extract patient ID
            patient_predictions[patient_id].append(pred_prob)  # Record all slice probabilities
            target_labels[patient_id] = true_label  # True label (same for all slices of the patient)

# Compute patient-level predictions
final_preds = {}
final_probs = {}  # Store final patient-level probability
for patient_id, prob_list in patient_predictions.items():
    avg_prob = np.mean(prob_list)  # Average slice probabilities
    final_probs[patient_id] = avg_prob
    final_preds[patient_id] = 1 if avg_prob >= 0.5 else 0  # Threshold = 0.5

# Extract patient-level true labels and predictions
targets = [target_labels[pid] for pid in final_preds.keys()]
preds = [final_preds[pid] for pid in final_preds.keys()]
probs = [final_probs[pid] for pid in final_probs.keys()]  # Patient-level probabilities

# Calculate evaluation metrics
accuracy = accuracy_score(targets, preds)
precision = precision_score(targets, preds, average='binary')
recall = recall_score(targets, preds, average='binary')
f1 = f1_score(targets, preds, average='binary')

# Calculate AUC
auc = roc_auc_score(targets, probs)

# Confusion matrix
conf_matrix = confusion_matrix(targets, preds)
tn, fp, fn, tp = conf_matrix.ravel()

# Sensitivity and specificity
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# Print results
print(f"Patient-Level Test Accuracy: {accuracy:.4f}")
print(f"Patient-Level Test Precision: {precision:.4f}")
print(f"Patient-Level Test Recall (Sensitivity): {recall:.4f}")
print(f"Patient-Level Test F1-score: {f1:.4f}")
print(f"Patient-Level Test Specificity: {specificity:.4f}")
print(f"Patient-Level Test AUC: {auc:.4f}")

# Plot confusion matrix
plt.figure(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["No Invasion", "Invasion"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Patient-Level Confusion Matrix")
plt.savefig("patient_confusion_matrix.png", dpi=300, bbox_inches='tight')
print("Patient-level confusion matrix saved as 'patient_confusion_matrix.png'")

# Plot ROC curve
fpr, tpr, _ = roc_curve(targets, probs)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Patient-Level ROC Curve')
plt.legend(loc='lower right')
plt.savefig("patient_roc_curve.png", dpi=300, bbox_inches='tight')
print("Patient-level ROC curve saved as 'patient_roc_curve.png'")
