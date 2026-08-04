import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Path to your confusion matrix CSV
csv_path = "confusion_matrix_labeled.csv"

# Read CSV:
# First column = actual labels
# Header row = predicted labels
cm = pd.read_csv(csv_path, index_col=0)

# Make the plot larger for readability
plt.figure(figsize=(10, 8))

# Draw color-coded heatmap
ax = sns.heatmap(
    cm,
    annot=True,
    fmt="g",
    cmap="Blues",
    linewidths=0.5,
    linecolor="white",
    xticklabels=cm.columns,
    yticklabels=cm.index,
)

# Force one tick per row/column
ax.set_xticks(np.arange(len(cm.columns)) + 0.5)
ax.set_yticks(np.arange(len(cm.index)) + 0.5)

ax.set_xticklabels(cm.columns, rotation=45, ha="right")
ax.set_yticklabels(cm.index, rotation=0)

ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("Actual label")

plt.tight_layout()
plt.show()