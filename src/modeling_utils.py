import os
import sys
import math
import logging
from pathlib import Path

# to handle datasets
import pandas as pd
import numpy as np

# to build the models
from sklearn.linear_model import LogisticRegression

# to evaluate the models

from sklearn.metrics import (classification_report, plot_confusion_matrix, plot_precision_recall_curve, 
                             plot_roc_curve, precision_score, recall_score, f1_score, accuracy_score,
                             roc_curve, auc)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

from yellowbrick.model_selection import RFECV
import matplotlib.pyplot as plt

from tqdm import tqdm


import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context("poster")
sns.set(rc={'figure.figsize': (12,8)})
sns.set_style("whitegrid")

   

def plot_train_test_metrics(model, X_train, y_train, X_test, y_test, threshold=0.5):
    """
    plot a comparision of the metrics precision, recal, f1, accuracy and roc auc
    between train and test metrics

    :param model: model estimator
    :param X_train: independent var for train
    :param y_train: target for train
    :param X_test: independent var for test
    :param y_test: target for test
    :param threshold: base score prediction
    :return:
    """
    def eval_metrics(y_true, y_score, y_pred):
        """
        compute the metrics
        :y_true: actual label
        :y_pred: predicted lable
        :return: precision, recall, f1, score, roc_auc
        """

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accurace = accuracy_score(y_true, y_pred)

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        return precision, recall, f1, accurace, roc_auc

    def get_metrics(model, X_train, y_train, X_test, y_test):
        y_score_train = model.predict_proba(X_train)[:, 1]
        y_pred_train = np.where(y_score_train > threshold, 1, 0)

        y_score_test = model.predict_proba(X_test)[:, 1]
        y_pred_test = np.where(y_score_test > threshold, 1, 0)

        train_metrics = eval_metrics(y_train, y_score_train, y_pred_train)
        test_metrics = eval_metrics(y_test, y_score_test, y_pred_test)

        metric = ["precision", "recall", "f1", "accurace", "roc_auc"]
        data = {"train": train_metrics, "test": test_metrics, "metric": metric}

        return pd.DataFrame(data)


    fig, ax = plt.subplots(figsize=(10, 5))

    df_metric = get_metrics(model, X_train, y_train, X_test, y_test)

    x = np.arange(len(df_metric.metric.unique()))

    bar_width = 0.4

    b1 = ax.bar(x, df_metric.train,
                width=bar_width, label='Train')
    b2 = ax.bar(x + bar_width, df_metric.test,
                width=bar_width, label='Test')

    # Fix the x-axes.
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(df_metric.metric.values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()



























