from ml_model.utils.preprocess import *

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
import scikitplot as skplt


def acceptance(matrix, n):
    i = n-1
    if n == 1:
        score = matrix[i][i] + matrix[i+1][i]
    elif n == len(matrix):
        score = matrix[i-1][i] + matrix[i][i]
    else:
        score = matrix[i-1][i] + matrix[i][i] + matrix[i+1][i]
    return score


def nan_to_zero(n):
    if np.isnan(n):
        return 0
    else:
        return n


def eval_predict(array):
    temp = np.asmatrix(array).transpose()
    matrix = np.array(temp)
    for i in range(len(matrix)):
        class_num = i+1
        print("Predict {} ({})".format(class_num, matrix[i].sum()))
        for n in range(len(matrix[i])):
            pred_num = n+1
            score = matrix[i][n]
            # print(" - Actual Class {}: {}".format(pred_num, score))
            if i == n:
                print(" - Accuracy:\t{}\t({:.0%})".format(score,
                      nan_to_zero(score/matrix[i].sum())))
        acp = acceptance(array, class_num)
        print(" - Acceptable:\t{}\t({:.0%})".format(nan_to_zero(acp),
              nan_to_zero(acp/matrix[i].sum())))
        print(" - Missed:\t{}\t({:.0%})".format(
            matrix[i].sum() - acp, 1 - nan_to_zero(acp/matrix[i].sum())))


def kfold_eval(n_splits, model, X, y_class):
    model_score = []
    underEst_score = []
    pred_all = np.array(())
    cv = KFold(n_splits)

    for train_index, test_index in cv.split(X):
        X_train_k, y_train_k = X[train_index], y_class[train_index]
        X_test_k, y_test_k = X[test_index], y_class[test_index]

        # print("test index: {}-{}".format(test_index[0], test_index[-1]))

        model.fit(X_train_k, y_train_k)
        pred = model.predict(X_test_k)
        accur = accuracy_score(y_test_k, pred)
        underEst = underEstimate_cal(y_test_k, pred)

        model_score.append(accur)
        underEst_score.append(underEst)
        pred_all = np.concatenate((pred_all, pred), axis=None)
        # skplt.metrics.plot_confusion_matrix(y_test_k, pred)
    show_score(model_score)
    show_score(underEst_score)
    # pred_all = pred_all.astype(int)
    eval_predict(confusion_matrix(y_class, pred_all))
    skplt.metrics.plot_confusion_matrix(y_class, pred_all)


def show_score(score):
    print("--------------------")
    print("Mean:\t", np.mean(score))
    print("Max:\t", np.max(score))
    print("Min:\t", np.min(score))
    print("S.D.:\t", np.std(score))
    print(score, "\n")

# Return only model score


def underEstimate_cal(y_true, y_pred):
    count_score = 0
    for i in range(len(y_true)):
        if y_pred[i]+1 == y_true[i]:
            count_score += 1
    return count_score/len(y_true)

def overEstimate_cal(y_true, y_pred):
    count_score = 0
    for i in range(len(y_true)):
        if y_pred[i]-1 == y_true[i]:
            count_score += 1
    return count_score/len(y_true)
    

def kfold_score(n_splits, model, X, y_class):
    model_score = []
    underEst_score = []
    overEst_score = []
    
    pred_all = np.array(())
    cv = KFold(n_splits)

    for train_index, test_index in cv.split(X):
        X_train_k, y_train_k = X[train_index], y_class[train_index]
        X_test_k, y_test_k = X[test_index], y_class[test_index]

        model.fit(X_train_k, y_train_k)
        pred = model.predict(X_test_k)
        accur = accuracy_score(y_test_k, pred)
        underEst = underEstimate_cal(y_test_k, pred)
        overEst = overEstimate_cal(y_test_k, pred)

        model_score.append(accur)
        underEst_score.append(underEst)
        overEst_score.append(overEst)
        
        pred_all = np.concatenate((pred_all, pred), axis=None)
    return np.mean(model_score), np.mean(underEst_score), np.mean(overEst_score)

def kfold_score_k3(n_splits, model, X, y_class):
    model_score_k3 = []
    
    numerator_0 = []
    numerator_1 = []
            
    denominator_0 = []
    denominator_1 = []
    
    for i in range(3):
        model_score = []
        
        cv = KFold(n_splits, random_state=i, shuffle=True)

        for train_index, test_index in cv.split(X):
            X_train_k, y_train_k = X[train_index], y_class[train_index]
            X_test_k, y_test_k = X[test_index], y_class[test_index]

            model.fit(X_train_k, y_train_k)
            pred = model.predict(X_test_k)
            accur = accuracy_score(y_test_k, pred)
            
            confusion = confusion_matrix(y_test_k, pred)
            # print(confusion)
            
            numerator_0.append(confusion[0][0])
            denominator_0.append(confusion[0][0] + confusion[1][0])
            
            numerator_1.append(confusion[1][1])
            denominator_1.append(confusion[0][1] + confusion[1][1])
            
            model_score.append(accur)
        
            
        model_score_k3.append(model_score)
        
    # print(numerator_0)
    # print(denominator_0)
    # print(numerator_1)
    # print(denominator_1)
        
    mean_accuracy_0 = sum(numerator_0)/sum(denominator_0)
    mean_accuracy_1 = sum(numerator_1)/sum(denominator_1)
    mean_accuracy_all = np.mean(model_score_k3)
    
    # print(model_score_k3)
    # print(mean_accuracy_0)
    # print(mean_accuracy_1)
            
    return [mean_accuracy_all, mean_accuracy_0, mean_accuracy_1]