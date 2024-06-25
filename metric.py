import torch
import numpy as np


def get_metrics(predict, target, threshold=0.5, predict_b=None):
    predict = torch.sigmoid(predict).cpu().detach().numpy().flatten()
    if predict_b is not None:
        predict_b = predict_b.flatten()
    else:
        predict_b = np.where(predict >= threshold, 1, 0)
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy().flatten()
    else:
        target = target.flatten()
    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()

    # Precision
    pre_denominator = tp + fp
    pre = tp / pre_denominator if pre_denominator > 0 else 0

    # Sensitivity
    sen_denominator = tp + fn
    sen = tp / sen_denominator if sen_denominator > 0 else 0

    # IoU
    iou_denominator = tp + fp + fn
    iou = tp / iou_denominator if iou_denominator > 0 else 0

    # F1-score
    f1_denominator = pre + sen
    f1 = 2 * pre * sen / f1_denominator if f1_denominator > 0 else 0

    numerator = (tp * tn) - (fp * fn)

    log_denominator = np.log1p(tp + fp) + np.log1p(tp + fn) + np.log1p(tn + fp) + np.log1p(tn + fn)
    denominator = np.exp(log_denominator / 2)  # Taking square root in the log space

    if denominator == 0:
        mcc = 0
    else:
        mcc = numerator / denominator

    return {
        "F1": np.round(f1, 4),
        "Sen": np.round(sen, 4),
        "IOU": np.round(iou, 4),
        "MCC": np.round(mcc, 4)
    }