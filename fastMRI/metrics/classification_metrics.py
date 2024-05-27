from typing import Dict
from sklearn import metrics
import torch
from torchmetrics import functional
import numpy as np


def compute_auc(preds, labels):
    return functional.auroc(preds=preds, target=labels, num_classes=2)


def compute_accuracy(preds, labels):
    return functional.accuracy(preds=preds, target=labels)


def get_operating_point(preds, labels, operating_point=None, threshold=0.1):
    # print(preds)
    preds = preds.cpu()
    preds_positive = preds[:, 1].numpy()

    labels = labels.cpu()

    if operating_point is None:
        fpr, tpr, thresholds = metrics.roc_curve(labels.numpy(), preds_positive)
        operating_point = thresholds[fpr > 0.25][0]

        try:
            fnr = 1 - tpr
            operating_point = thresholds[fnr < threshold][0]
        except IndexError:
            operating_point = thresholds[fpr > 0.25][0]


    test_predictions = (preds_positive > operating_point).astype(int)

    sensitivity = metrics.recall_score(labels, test_predictions)
    specificity = metrics.recall_score(
        np.abs(1 - labels.numpy()), np.abs(1 - test_predictions)
    )
    balanced_acc = metrics.balanced_accuracy_score(labels.numpy(), test_predictions)

    return balanced_acc, specificity, sensitivity, operating_point


def evaluate_classifier(
    preds: torch.Tensor, labels: torch.Tensor, operating_point: float = None
) -> Dict:
    try:
        auc = compute_auc(preds=preds[:, 1], labels=labels).item()
        balanced_acc, specificity, sensitivity, operating_point = get_operating_point(
            preds, labels, operating_point=operating_point
        )
    except ValueError:
        auc = np.nan
        specificity = np.nan
        sensitivity = np.nan
        balanced_acc = np.nan
        operating_point = np.nan
        print(
            "No negative samples in targets, false positive value should be meaningless"
        )

    return dict(
        auc=auc,
        sensitivity=sensitivity,
        specificity=specificity,
        balanced_accuracy=balanced_acc,
        operating_point=operating_point,
    )


def get_bootstrap_estimates(
    preds: torch.Tensor,
    labels: torch.Tensor,
    operating_point: float,
    n_bootstrap_samples: int,
) -> Dict:
    N = len(preds)

    arr_auc = []
    arr_sensitivity = []
    arr_specificity = []
    arr_balanced_accuracy = []

    metrics = evaluate_classifier(
        preds=preds, labels=labels, operating_point=operating_point,
    )
    auc = metrics["auc"]
    sensitivity = metrics["sensitivity"]
    specificity = metrics["specificity"]
    balanced_accuracy = metrics["balanced_accuracy"]

    for i in range(n_bootstrap_samples):
        bootstrap_index = np.random.choice(
            np.arange(N), size=N, replace=True, p=np.ones(N) / N
        )
        bootstrap_preds = preds[bootstrap_index]
        bootstrap_labels = labels[bootstrap_index]

        if bootstrap_labels.sum() == 0:
            bootstrap_index = np.random.choice(
                np.arange(N), size=N, replace=True, p=np.ones(N) / N
            )

        bootstrap_metrics = evaluate_classifier(
            preds=bootstrap_preds,
            labels=bootstrap_labels,
            operating_point=operating_point,
        )
        arr_auc.append(bootstrap_metrics["auc"])
        arr_sensitivity.append(bootstrap_metrics["sensitivity"])
        arr_specificity.append(bootstrap_metrics["specificity"])
        arr_balanced_accuracy.append(bootstrap_metrics["balanced_accuracy"])

    std_auc = np.array(arr_auc).std()
    std_sensitivity = np.array(arr_sensitivity).std()
    std_specificity = np.array(arr_specificity).std()
    std_balanced_accuracy = np.array(arr_balanced_accuracy).std()

    print(operating_point)

    return dict(
        auc=auc,
        std_auc=std_auc,
        sensitivity=sensitivity,
        std_sensitivity=std_sensitivity,
        specificity=specificity,
        std_specificity=std_specificity,
        balanced_accuracy=balanced_accuracy,
        std_balanced_accuracy=std_balanced_accuracy,
        operating_point=operating_point,
    )


def classifier_metrics(
    val_preds: torch.Tensor,
    val_labels: torch.Tensor,
    test_preds: torch.Tensor,
    test_labels: torch.Tensor,
    threshold: float,
    n_bootstrap_samples: int,
):
    _, _, _, operating_point = get_operating_point(
        preds=val_preds, labels=val_labels, operating_point=None, threshold=threshold
    )

    return (
        operating_point,
        get_bootstrap_estimates(
            preds=test_preds, labels=test_labels, operating_point=operating_point
        ),
    )
