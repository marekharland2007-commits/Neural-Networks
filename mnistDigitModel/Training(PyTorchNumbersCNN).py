from xml.parsers.expat import model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import gzip
import pickle
import numpy as np
from digitModel import CNNNet
from sklearn.metrics import confusion_matrix

def load_mnist():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    return train_set, valid_set, test_set

def train(model, X_train, y_train, X_valid, y_valid, initial_learning_rate, epochs, device):
    """
    Train the model with your learning rate decay schedule and best-weight reload.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Convert numpy arrays to PyTorch tensors
    X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    X_valid_t = torch.as_tensor(X_valid, dtype=torch.float32, device=device)
    y_valid_t = torch.as_tensor(y_valid, dtype=torch.long, device=device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    valid_ds = TensorDataset(X_valid_t, y_valid_t)

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    best_val_accuracy = 0.0
    best_model_state = None

    for epoch in range(epochs):

        # -------- Train over batches --------
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)  # your CNN handles reshape internally
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == batch_y).sum().item()
            running_total += batch_X.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # -------- Validation over batches --------
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_X, val_y in valid_loader:
                val_X = val_X.to(device)
                val_y = val_y.to(device)

                logits = model(val_X)
                preds = logits.argmax(dim=1)
                val_correct += (preds == val_y).sum().item()
                val_total += val_X.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

def evaluate(model, X, y, device):
    model.eval()
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(dim=1)
        acc = (preds == y_t).float().mean().item()
    return acc

def compute_confusion_matrix(model, X_test, y_test, device):
    model.eval()
    X_t = torch.as_tensor(X_test, dtype=torch.float32, device=device).view(-1, 1, 28, 28)
    y_t = torch.as_tensor(y_test, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(dim=1).cpu().numpy()

    cm = confusion_matrix(y_test, preds)
    return cm

if __name__ == "__main__":
    # Select device (GPU if available)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load MNIST
    train_set, valid_set, test_set = load_mnist()
    X_train, y_train = train_set
    X_valid, y_valid = valid_set
    X_test, y_test = test_set

    # No one-hot encoding needed: y_* should be integer labels 0–9
    input_size = X_train.shape[1]
    output_size = 10

    model = CNNNet().to(device)

    learning_rate = 0.01
    epochs = 50

    model = train(model, X_train, y_train, X_valid, y_valid,
                  learning_rate, epochs, device)

    train_accuracy = evaluate(model, X_train, y_train, device)
    valid_accuracy = evaluate(model, X_valid, y_valid, device)
    test_accuracy = evaluate(model, X_test, y_test, device)

    print(f"\nFinal Train Accuracy: {train_accuracy:.4f}")
    print(f"Final Valid Accuracy: {valid_accuracy:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

    cm = compute_confusion_matrix(model, X_test, y_test, device)
    print(cm)

    # Save PyTorch weights
    torch.save(model.state_dict(), "digit_model_pytorch.pth")
    print("Saved model to digit_model_pytorch.pth")