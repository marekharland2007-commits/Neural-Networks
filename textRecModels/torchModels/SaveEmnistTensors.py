import os
import numpy as np
import torch
from torchvision import datasets, transforms


class FixEMNISTOrientation:
    def __call__(self, img):
        _img = transforms.functional.rotate(img, -90)
        _img = transforms.functional.hflip(_img)
        return _img


# Deterministic transform: orientation + tensor + normalize
base_transform = transforms.Compose([
    FixEMNISTOrientation(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Random affine to be applied ONLY to training tensors (offline once)
random_affine = transforms.RandomAffine(
    degrees=10,
    translate=(0.1, 0.1),
    scale=(0.9, 1.1),
)


def dataset_to_tensors(dataset, name):
    xs = []
    ys = []
    for i, (img, label) in enumerate(dataset):
        xs.append(img)
        ys.append(label)
        if (i + 1) % 50000 == 0:
            print(f"Processed {i + 1} samples for {name}")
    X = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y


def main(
    data_dir="data",
    out_path="emnist_byclass_tensors.pt",
    valid_ratio=0.1,
    subset_size=None,
    seed=42,
):
    # Load full EMNIST train/test with base transform (no random affine here)
    train_dataset = datasets.EMNIST(
        root=data_dir,
        split="bymerge",
        train=True,
        download=True,
        transform=base_transform,
    )

    test_dataset = datasets.EMNIST(
        root=data_dir,
        split="bymerge",
        train=False,
        download=True,
        transform=base_transform,
    )

    # Optionally limit train size before splitting
    num_train_total = len(train_dataset)
    indices = np.arange(num_train_total)
    if subset_size is not None and subset_size < num_train_total:
        indices = indices[:subset_size]

    # Train/valid split indices
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_valid = int(valid_ratio * len(indices))
    valid_indices = indices[:n_valid]
    train_indices = indices[n_valid:]

    print(f"Total train samples (after subset, before split): {len(indices)}")
    print(f"Train split: {len(train_indices)}, Valid split: {len(valid_indices)}")

    # Build small datasets via index lists
    train_split = torch.utils.data.Subset(train_dataset, train_indices)
    valid_split = torch.utils.data.Subset(train_dataset, valid_indices)

    print("Converting train split to tensors...")
    X_train, y_train = dataset_to_tensors(train_split, "train")

    print("Converting valid split to tensors...")
    X_valid, y_valid = dataset_to_tensors(valid_split, "valid")

    print("Converting test set to tensors...")
    X_test, y_test = dataset_to_tensors(test_dataset, "test")

    # Apply RandomAffine ONLY to training tensors (offline once)
    # print("Applying RandomAffine to training tensors...")
    # for i in range(X_train.shape[0]):
    #     X_train[i] = random_affine(X_train[i])

    # Save everything
    torch.save(
        {
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "X_test": X_test,
            "y_test": y_test,
        },
        out_path,
    )

    print(f"Saved tensors to {out_path}")
    print(f"X_train shape: {tuple(X_train.shape)}")
    print(f"y_train shape: {tuple(y_train.shape)}")
    print(f"X_valid shape: {tuple(X_valid.shape)}")
    print(f"y_valid shape: {tuple(y_valid.shape)}")
    print(f"X_test shape:  {tuple(X_test.shape)}")
    print(f"y_test shape:  {tuple(y_test.shape)}")


if __name__ == "__main__":
    main()