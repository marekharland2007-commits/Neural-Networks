import torch
import torch.nn as nn
import torch.optim as optim
import gzip
import pickle
import numpy as np
from digitModel import MNISTNet
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
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate)

    # Convert numpy arrays to PyTorch tensors
    X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    X_valid_t = torch.as_tensor(X_valid, dtype=torch.float32, device=device)
    y_valid_t = torch.as_tensor(y_valid, dtype=torch.long, device=device)

    best_val_accuracy = 0.0
    best_model_state = None
    noImprovement_epochs = 0
    epoch_reductions = 0
    learning_rate = initial_learning_rate

    for epoch in range(epochs):
        # Learning rate decay at 1/4, 1/2, 3/4 of total epochs
        if epoch == (epochs // 4) and epoch_reductions == 0:
            learning_rate *= 0.75
            print(f"Learning rate reduced to {learning_rate:.6f} at epoch {epoch+1}")
            for pg in optimizer.param_groups:
                pg['lr'] = learning_rate
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            epoch_reductions += 1

        if epoch == (epochs // 2) and epoch_reductions <= 1:
            learning_rate *= 0.75
            print(f"Learning rate reduced to {learning_rate:.6f} at epoch {epoch+1}")
            for pg in optimizer.param_groups:
                pg['lr'] = learning_rate
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            epoch_reductions += 1

        elif epoch == (epochs // 4 * 3) and epoch_reductions <= 2:
            learning_rate *= 0.75
            print(f"Learning rate reduced to {learning_rate:.6f} at epoch {epoch+1}")
            for pg in optimizer.param_groups:
                pg['lr'] = learning_rate
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            epoch_reductions += 1

        # ----- Training step -----
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        # ----- Evaluate train + val -----
        model.eval()
        with torch.no_grad():
            train_logits = model(X_train_t)
            train_preds = train_logits.argmax(dim=1)
            train_acc = (train_preds == y_train_t).float().mean().item()

            val_logits = model(X_valid_t)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_valid_t).float().mean().item()

        print(f'Epoch {epoch+1}/{epochs}, '
              f'Loss: {loss.item():.4f}, '
              f'Train Acc: {train_acc:.4f}, '
              f'Val Acc: {val_acc:.4f}')

        # Track best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            noImprovement_epochs = 0
        else:
            noImprovement_epochs += 1

    # Load best weights
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

def confusion_matrix(model, X_test, y_test, device):
    model.eval()
    X_t = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_test, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(dim=1).cpu().numpy()

    cm = confusion_matrix(y_test, preds)
    return cm

if __name__ == "__main__":
    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MNIST
    train_set, valid_set, test_set = load_mnist()
    X_train, y_train = train_set
    X_valid, y_valid = valid_set
    X_test, y_test = test_set

    # No one-hot encoding needed: y_* should be integer labels 0–9
    input_size = X_train.shape[1]
    output_size = 10

    model = MNISTNet(input_size, 256, 128, output_size).to(device)

    learning_rate = 0.5
    epochs = 1000

    model = train(model, X_train, y_train, X_valid, y_valid,
                  learning_rate, epochs, device)

    train_accuracy = evaluate(model, X_train, y_train, device)
    valid_accuracy = evaluate(model, X_valid, y_valid, device)
    test_accuracy = evaluate(model, X_test, y_test, device)

    print(f"\nFinal Train Accuracy: {train_accuracy:.4f}")
    print(f"Final Valid Accuracy: {valid_accuracy:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

    cm = confusion_matrix(model, X_test, y_test, device)
    print(cm)
    print("Errors for 7:", cm[7])

    # Save PyTorch weights
    torch.save(model.state_dict(), "digit_model_pytorch.pth")
    print("Saved model to digit_model_pytorch.pth")