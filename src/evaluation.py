#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2025/09/18
@author yhj

"""

import math
import numpy as np
from pathlib import Path

__all__ = ['CUTOFF', 'acc_and_auc', 'get_perfomance']

CUTOFF = 1.0 - math.log(500, 50000)


def acc_and_auc(ouputs, targets): 
    y_pred = []
    y_true = []
    ouputs_int = []
    for tmp in range(len(ouputs)):
        y_pred.append(ouputs[tmp].item())
        y_true.append(int(targets[tmp]))
        if ouputs[tmp] >= CUTOFF:
            ouputs_int.append(1)
        else:
            ouputs_int.append(0)
    num_correct = 0
    for tmp in range(len(ouputs_int)):
        if ouputs_int[tmp] == targets[tmp]:
            num_correct += 1
    return y_pred, y_true, num_correct, ouputs_int

def get_perfomance(labelArr, predictArr):
    TP = 0.0001
    TN = 0.0001
    FP = 0.0001
    FN = 0.0001
    for i in range(len(labelArr)):
        if labelArr[i] == 1 and predictArr[i] == 1:
            TP += 1.
        if labelArr[i] == 1 and predictArr[i] == 0:
            FN += 1.
        if labelArr[i] == 0 and predictArr[i] == 1:
            FP += 1.
        if labelArr[i] == 0 and predictArr[i] == 0:
            TN += 1.
    SN = TP / (TP + FN)
    SP = TN / (FP + TN)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    ACC = (TP+TN) / len(labelArr)
    F1 = (2 * TP) / (2 * TP + FP + FN)
    precision = TP / (TP + FP)
    recall = SN
    bacc = (precision + recall) / 2
    return SN, SP, MCC, ACC, F1, precision, recall, bacc


def output_res(mhc_names, targets, scores, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, scores, fmt='%.6f')
