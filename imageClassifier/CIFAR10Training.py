import time

import math

from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import Subset, DataLoader
from torch.amp import autocast, GradScaler

import torchvision
from torchvision import transforms

from copy import deepcopy

import pandas as pd

from Models import ResNet18

def add_weight_noise(model, std=1e-3):
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * std)

def load_data(
    batch_size=64,
    train_size=None,
    valid_size=None,
    test_size=None
    ):
    download = False

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load training set
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=download, 
        transform=train_transform
    )

    # Load test set
    fulltestset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=download, 
        transform=basic_transform
    )

    # Create validation set from the end of the test set
    testset_size = len(fulltestset)

    valset = Subset(fulltestset, indices=range(int(0.5 * testset_size), testset_size))
    testset = Subset(fulltestset, indices=range(int(0.5 * testset_size)))

    if train_size is not None and train_size < len(trainset):
        trainset = Subset(trainset, indices=range(train_size))
    if valid_size is not None and valid_size < len(valset):
        valset = Subset(valset, indices=range(valid_size))
    if test_size is not None and test_size < len(testset):
        testset = Subset(testset, indices=range(test_size))
    

    # Create DataLoaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    print(f"Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    return train_loader, val_loader, test_loader

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

def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup from 0 to 1.0 (will be multiplied by base lr=0.4)
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return cosine_decay * (1 - eta_min / 0.4) + eta_min / 0.4
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def training(model,train_loader, valid_loader, device, epochs=50, warmup_epochs=10, learning_rate=0.001, weightDecay=1e-4, momentum=0.9):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weightDecay, momentum=momentum)
    scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs, epochs)
    scaler = GradScaler()

    best_val_accuracy = 0.0
    best_model_state = None
    epochswoimprovement = 0
    patience = 15

    for epoch in range(epochs):
        t0 = time.time()

        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f},", end=" ")

        train_accuracy = evaluate(model, train_loader, device)
        print(f"Train Accuracy: {train_accuracy:.4f},", end=" ")

        val_accuracy = evaluate(model, valid_loader, device)
        print(f"Validation Accuracy: {val_accuracy:.4f},", end=" ")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = deepcopy(model.state_dict())
            epochswoimprovement = 0
        else:
            epochswoimprovement += 1
        
        if epochswoimprovement >= patience:
            print("Early stopping triggered")
            break

        scheduler.step()

        # add_weight_noise(model, std=1e-4)

        t1 = time.time()
        print(f"Epoch time: {t1 - t0:.2f} seconds")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_model_state

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

    model = ResNet18(num_classes=10).to(device)
    
    # Edit Hyperparameters here:
    lr = 0.3
    batch_size = 1024
    epochs = 75
    weightDecay = 5e-4
    momentum = 0.9
    warmup_epochs = 5

    t0 = time.time()
    train_loader, valid_loader, test_loader = load_data(batch_size=batch_size)
    best_model_state = training(model, train_loader, valid_loader, device, epochs=epochs, warmup_epochs=warmup_epochs, learning_rate=lr, weightDecay=weightDecay, momentum=momentum)
    t1 = time.time()
    print(f"Total training time: {t1 - t0:.2f} seconds")

    train_accuracy = evaluate(model, train_loader, device)
    valid_accuracy = evaluate(model, valid_loader, device)
    test_accuracy = evaluate(model, test_loader, device)

    with open("trainingLog.txt", "w") as f:
        print(f"Epochs: {epochs}", file = f)
        print(f"Warmup Epochs: {warmup_epochs}", file = f)
        print(f"Batch size: {batch_size}", file = f)
        print(f"Learning rate: {lr}", file = f)
        print(f"Dropout: {model.dropout.p}", file = f)
        print(f"Weight Decay: {weightDecay}", file = f)
        print(f"Momentum: {momentum}", file = f)

        print(f"\nTraining time: {t1 - t0:.2f} seconds", file = f)

        print(f"\nFinal Train Accuracy: {train_accuracy:.4f}", file = f)
        print(f"Final Valid Accuracy: {valid_accuracy:.4f}", file = f)
        print(f"Final Test Accuracy: {test_accuracy:.4f}", file = f)

    print(f"\nFinal Train Accuracy: {train_accuracy:.4f}")
    print(f"Final Valid Accuracy: {valid_accuracy:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

    cm = compute_confusion_matrix(model, test_loader, device)

    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.to_csv("CIFAR10_confusion_matrix.csv")

    torch.save(model.state_dict(), "CIFAR10Model.pth")
    print("Saved model to CIFAR10Model.pth")
