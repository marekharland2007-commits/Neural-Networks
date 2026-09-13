import time
import sys
import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

from textRecModels.torchModels.digitModel import UpgradedCNNNet

torch.backends.cudnn.benchmark = True


def add_weight_noise(model, std=1e-3):
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * std)

def load_cached_emnist_tensors(
    tensor_path="emnist_byclass_tensors.pt",
    batch_size=512,
    subset_size=None,
    valid_size = None,
    test_size = None
):
    data = torch.load(tensor_path)

    X_train, y_train = data["X_train"], data["y_train"]
    X_valid, y_valid = data["X_valid"], data["y_valid"]
    X_test,  y_test  = data["X_test"],  data["y_test"]

    # Optional subset on TRAIN ONLY
    if subset_size is not None and subset_size < X_train.shape[0]:
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        indices = indices[:subset_size]

        X_train = X_train[indices]
        y_train = y_train[indices]

    if valid_size is not None and valid_size < X_valid.shape[0]:
        indices1 = np.arange(X_valid.shape[0])
        np.random.shuffle(indices1)
        indices1 = indices1[:valid_size]

        X_valid = X_valid[indices1]
        y_valid = y_valid[indices1]

    if test_size is not None and test_size < X_test.shape[0]:
        indices2 = np.arange(X_test.shape[0])
        np.random.shuffle(indices2)
        indices2 = indices2[:test_size]

        X_test = X_test[indices2]
        y_test = y_test[indices2]

    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset  = TensorDataset(X_test,  y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print("Train samples:", len(train_dataset))
    print("Valid samples:", len(valid_dataset))
    print("Test samples:", len(test_dataset))
    print("Train batches:", len(train_loader))

    return train_loader, valid_loader, test_loader


def train(model, train_loader, valid_loader, initial_learning_rate, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=5e-5)

    best_val_accuracy = 0.0
    best_model_state = None
    epochswoimprovement = 0
    patience = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs, eta_min=5e-5)
    

    for epoch in range(epochs):
        t0 = time.time()

        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        # -------- Train over batches --------
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == batch_y).sum().item()
            running_total += batch_X.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}",
            end="",
        )

        # -------- Validation every epoch (or change condition) --------
        if (epoch + 1) % 1 == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_X, val_y in valid_loader:
                    val_X = val_X.to(device, non_blocking=True)
                    val_y = val_y.to(device, non_blocking=True)

                    logits = model(val_X)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == val_y).sum().item()
                    val_total += val_X.size(0)

            val_acc = val_correct / val_total

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                epochswoimprovement = 0
            elif val_acc <= best_val_accuracy:
                epochswoimprovement += 1
                if epochswoimprovement >= patience:
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    return model
            
            print(f", Val Acc: {val_acc:.4f}", end="")

        scheduler.step()

        add_weight_noise(model, std=5e-3)

        t1 = time.time()

        print(
            f"  |  Epoch time: {t1 - t0:.2f} s "
            f"({len(train_loader)} train batches)"
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    return correct / total


def compute_confusion_matrix(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load cached EMNIST tensors and split into loaders
    train_loader, valid_loader, test_loader = load_cached_emnist_tensors(
        tensor_path="emnist_byclass_tensors.pt",
        batch_size=2048,
        subset_size=200000,
        valid_size=20000,
        test_size=20000
    )

    model = UpgradedCNNNet().to(device)

    learning_rate = 0.001
    epochs = 50

    t0 = time.time()
    model = train(model, train_loader, valid_loader, learning_rate, epochs, device)
    t1 = time.time()
    print(f"Training Time: {t1 - t0:.2f}")

    train_accuracy = evaluate(model, train_loader, device)
    valid_accuracy = evaluate(model, valid_loader, device)
    test_accuracy = evaluate(model, test_loader, device)
    with open("trainingLog.txt", "w") as f:
        print(f"\nFinal Train Accuracy: {train_accuracy:.4f}", file = f)
        print(f"Final Valid Accuracy: {valid_accuracy:.4f}", file = f)
        print(f"Final Test Accuracy: {test_accuracy:.4f}", file = f)

    cm = compute_confusion_matrix(model, test_loader, device)

    labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.to_csv("confusion_matrix_labeled.csv")

    torch.save(model.state_dict(), "digit+letter_model_pytorch.pth")
    print("Saved model to digit+letter_model_pytorch.pth")