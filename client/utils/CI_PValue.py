#  Copyright (c) 2021. Jiefeng, Ziwei and Hanchen
#  jiefenggan@gmail.com, ziwei@hust.edu.cn, hc.wang96@gmail.com


import math
import pickle
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.nn.functional import binary_cross_entropy


def bootstrap(idx_lst, n_sample=1000):
    sampled_id_lst = []
    for _ in range(n_sample):
        temp_lst = []
        for _ in range(len(idx_lst)):
            temp_lst.append(random.choice(idx_lst))
        sampled_id_lst.append(temp_lst)
    return sampled_id_lst


def cal_CI(binary_label_file='binary_label_FL.pkl',
           pred_probs_file='pred_probs_FL.pkl'):
    with open(binary_label_file, 'rb') as f:
        binary_label = pickle.load(f)
    with open(pred_probs_file, 'rb') as f:
        pred_probs = pickle.load(f)
    pred, label = pred_probs[:, 1], binary_label[:, 1]
    idx_lst = [i for i in range(pred.shape[0])]
    sampled_id_lst = bootstrap(idx_lst, n_sample=1000)

    auc_lst = []
    for sample in sampled_id_lst:
        sample_pred = pred[sample]
        sample_label = label[sample]
        auc = roc_auc_score(sample_label, sample_pred)
        auc_lst.append(auc)
    ci = np.percentile(auc_lst, (2.5, 97.5))
    return ci


def cal_pvalue(binary_label_file='binary_label_FL.pkl',
               pred_probs_file='pred_probs_FL.pkl'):
    with open(binary_label_file, 'rb') as f:
        binary_label = pickle.load(f)
    with open(pred_probs_file, 'rb') as f:
        pred_probs = pickle.load(f)
    pred, label = pred_probs[:, 1], binary_label[:, 1]
    unsampled_auc = roc_auc_score(label, pred)
    idx_lst = [i for i in range(pred.shape[0])]
    sampled_id_lst = bootstrap(idx_lst, n_sample=1000)

    diff_auc_lst = []
    for sample in sampled_id_lst:
        sample_pred = pred[sample]
        sample_label = label[sample]
        auc = roc_auc_score(sample_label, sample_pred)
        diff_auc_lst.append(np.abs(auc - unsampled_auc))
    pvalue = np.mean(np.array(diff_auc_lst))
    return pvalue


if __name__ == '__main__':
    binary_label_file = 'binary_label_cambridge.pkl'
    pred_probs_file = 'pred_probs_cambridge.pkl'
    ci = cal_CI(binary_label_file, pred_probs_file)
    print(ci)
    pvalue = cal_pvalue(binary_label_file, pred_probs_file)
    print(pvalue)
