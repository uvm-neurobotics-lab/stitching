"""
Various metrics and loss functions. These will generally be functions of one of the following signatures:
    - (output: Tensor, labels: Tensor) -> loss_value: float
    - (output: Tensor, labels: Tensor) -> dict[metric_name: str, metric_value: float]
"""
from typing import Union

import numpy as np
import torch


def accuracy(preds: Union[torch.Tensor, np.ndarray], labels: Union[torch.Tensor, np.ndarray]):
    """
    Computes accuracy assuming labels are a list of indices of the ground-truth classes.
    """
    assert len(preds) == len(labels)
    if len(preds) == 0:
        return np.nan
    return {"Accuracy": (preds.argmax(axis=1) == labels).sum().item() / len(labels)}


def topk_accuracy(preds: Union[torch.Tensor, np.ndarray], labels: Union[torch.Tensor, np.ndarray],
                  topk: Union[int, tuple, list] = 1):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    assert len(preds) == len(labels)
    if not isinstance(topk, (tuple, list)):
        topk = [topk]

    # Get largest K label indices in each row.
    maxk = max(topk)
    _, largest_indices = preds.topk(maxk, dim=1)
    largest_indices = largest_indices.t()
    # Mark as True any place where one of the top K indices matches the label index.
    # Now columns are examples and rows are 1 thru K.
    correct = largest_indices.eq(labels.view(1, -1).expand_as(largest_indices))

    res = {}
    for k in topk:
        # Take only the first K rows (corresponding to the top K answers for each sample), and sum.
        # There cannot be more than one true value per sample, so the total sum is the total # correct.
        # If we want a vector that would tell us which samples were correct, we could use `.sum(0, keepdim=True)`.
        num_correct = correct[:k].sum().item()
        res[f"Top-{k} Accuracy"] = num_correct / len(labels)
    return res
