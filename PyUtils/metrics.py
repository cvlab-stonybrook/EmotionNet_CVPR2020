import sklearn.metrics as metrics
import numpy as np

def compute_mAP(y_labels, y_predicts):
    # n by d matrix, where n is the number of instances, d is the number of classes
    assert y_labels.shape == y_predicts.shape, 'Check'
    n_classes = y_labels.shape[1]
    total_aps = np.zeros(n_classes)
    for i in range(n_classes):
        label_slice = y_labels[:, i]
        predicts_slice = y_predicts[:, i]
        s_ap = metrics.average_precision_score(np.array(label_slice), predicts_slice)
        total_aps[i] = s_ap
    return total_aps