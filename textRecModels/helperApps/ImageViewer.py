import matplotlib.pyplot as plt
from torchvision.datasets import EMNIST
from torchvision import transforms

dataset = EMNIST(
    root="data",
    split="bymerge",
    train=True,
    download=False,
    transform=None
)

classes = dataset.classes

def fix1(img):
    return transforms.functional.rotate(img, -90)

def fix2(img):
    img = transforms.functional.rotate(img, -90)
    img = transforms.functional.hflip(img)
    return img

fig, axes = plt.subplots(3, 5, figsize=(12, 7))

for i in range(5):
    img, label = dataset[i]
    char = classes[label]

    axes[0, i].imshow(img, cmap="gray")
    axes[0, i].set_title(f"Raw: {char}")
    axes[0, i].axis("off")

    axes[1, i].imshow(fix1(img), cmap="gray")
    axes[1, i].set_title(f"Rot -90: {char}")
    axes[1, i].axis("off")

    axes[2, i].imshow(fix2(img), cmap="gray")
    axes[2, i].set_title(f"Rot -90 + Flip: {char}")
    axes[2, i].axis("off")

plt.tight_layout()
plt.show()