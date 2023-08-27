"""
experiment functions
"""

import io

from torchmetrics import Accuracy, ConfusionMatrix
import torch
from matplotlib import pyplot as plt
from mlxtend.plotting import heatmap, plot_confusion_matrix
from PIL import Image


def update_accuracy_confusion_matrix(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    accuracy: Accuracy,
    confusion_matrix: ConfusionMatrix,
) -> None:
    """
    self accuracy & confusion_matrix update

    Args:
        outputs (torch.Tensor): _description_
        labels (torch.Tensor): _description_
        accuracy (Accuracy): _description_
        confusion_matrix (ConfusionMatrix): _description_
    """
    accuracy(outputs, labels)
    confusion_matrix(outputs, labels)


# Attention Matrix update: mat method = 'contain'
def update_attention_matrix(batch_attention_matrix, value_attention_matrix, count_attention_matrix, labels):
    for idx_i, label_i in enumerate(labels):
        for idx_j, label_j in enumerate(labels):
            if idx_i == idx_j:
                continue
            value_attention_matrix[label_i][label_j] += batch_attention_matrix[idx_i][idx_j]
            count_attention_matrix[label_i][label_j] += 1


# make confusion matrix plot
# use when appear new best measure
def get_confusion_matrix_plot(confusion_matrix, expression_labels):
    figure, _ = plot_confusion_matrix(
        conf_mat=confusion_matrix,
        colorbar=True,
        show_absolute=False,
        show_normed=True,
        class_names=expression_labels,
    )
    return figure


# make attention matrix plot
# use when appear new best attention
def get_attention_matrix_plot(attention_matrix, expression_labels):
    figure, _ = heatmap(
        attention_matrix,
        row_names=expression_labels,
        column_names=expression_labels,
    )
    return figure


# convert plt figure to PIL Image
def img_from_buffer(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer)
    buffer.seek(0)
    del figure
    plt.close()
    return Image.open(buffer)
