import numpy as np


def classcify_performance(y, hat_y, true_label=1, false_label=0):
    accuracy = np.sum(y == hat_y) / len(y)  # 准确率
    errorRate = 1 - accuracy  # 错误率

    true_pos = len(y[y == true_label][hat_y == true_label])
    true_neg = len(y[y == true_label][hat_y == false_label])
    flase_neg = len(y[y == false_label][hat_y == false_label])

    precision = true_pos / (true_pos + true_neg)
    recall = true_pos / (true_pos + flase_neg)
    return {
        "Accuracy": accuracy,
        "ErrorRate": errorRate,
        "Precision": precision,
        "Recall": recall,
        "F1": 2 * precision * recall / (precision + recall),
    }
