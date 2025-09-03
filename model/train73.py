import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from Model import ResNet18FineTune
from datasetfor2d73 import get_all_dataloaders

def calculate_metrics(labels, preds, probs):
    """ Calculate evaluation metrics """
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')
    auc = roc_auc_score(labels, probs)

    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return acc, precision, recall, f1, auc, specificity

def train(model, train_loader, val_loader, num_epochs=100, lr=0.001):
    """ Train the model """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Optimizer
    best_val_loss = float('inf')  # Initialize best validation loss
    best_metrics = {}  # Store best metrics

    print(f"Device: {device}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # ========== Training Phase ==========
        model.train()
        train_loss, total = 0.0, 0
        train_labels, train_preds, train_probs = [], [], []

        for images, labels, _ in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(predicted.cpu().numpy())
            train_probs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())

            total += labels.size(0)

        train_loss /= total
        train_acc, train_precision, train_recall, train_f1, train_auc, train_specificity = calculate_metrics(
            train_labels, train_preds, train_probs)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Precision: {train_precision:.4f} | "
              f"Recall: {train_recall:.4f} | F1: {train_f1:.4f} | AUC: {train_auc:.4f} | "
              f"Specificity: {train_specificity:.4f}")

        # ========== Validation Phase ==========
        model.eval()
        val_loss, total = 0.0, 0
        val_labels, val_preds, val_probs = [], [], []

        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())
                val_probs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
                total += labels.size(0)

        val_loss /= total
        val_acc, val_precision, val_recall, val_f1, val_auc, val_specificity = calculate_metrics(
            val_labels, val_preds, val_probs)

        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Precision: {val_precision:.4f} | "
              f"Recall: {val_recall:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f} | "
              f"Specificity: {val_specificity:.4f}")

        # ========== Save Best Model (based on lowest validation loss) ==========
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "Loss": val_loss,
                "Accuracy": val_acc,
                "Precision": val_precision,
                "Recall": val_recall,
                "F1": val_f1,
                "AUC": val_auc,
                "Specificity": val_specificity
            }
            torch.save(model.state_dict(), "best_model_val_loss.pth")
            print("🔥 Best model saved! (based on lowest validation loss)")

    print(f"\nTraining complete! Best validation metrics (by lowest loss):")
    for k, v in best_metrics.items():
        print(f"{k}: {v:.4f}")


# Load dataset
root_folder = "/path/to/your/dataset"
train_loader, val_loader = get_all_dataloaders(root_folder, batch_size=32)

# Initialize model
model = ResNet18FineTune(num_classes=2)

# Start training
train(model, train_loader, val_loader, num_epochs=100)
