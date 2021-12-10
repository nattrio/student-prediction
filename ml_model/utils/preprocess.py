import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ml_model.utils.scaler import *


def dropRecords(dataset, VAR_NUM):
    df = pd.DataFrame.copy(dataset)

    # print("Before:\t", df['Yr'].unique())
    # print(df.shape)
    # print(df.columns)

    # Drop null subject
    for v in VAR_NUM:
        df[v].replace('', np.nan, inplace=True)

    df.dropna(subset=VAR_NUM, inplace=True)

    # Drop Yr <= 51
    # df.drop(df[df['Yr'] <= 51].index, axis=0, inplace=True)

    # หลังจาก drop แล้ว ให้เรียง index ใหม่
    df.index = range(len(df))
    # print("After:\t", df['Yr'].unique())
    # print(df.shape)
    return df


def toClass(df, VAR_NUM, IN_CLASS, OUT_CLASS, IN_SCALE, OUT_SCALE):
    col_list = VAR_NUM
    if IN_CLASS == 0:
        for col in col_list:
            pass
    else:
        for col in col_list:
            scale_temp = scaleList(df[col], IN_CLASS, IN_SCALE)
            # print(col, scale_temp)
            applyClass(df[col], col, scale_temp)

    scale_sit = scaleList(df["SIT_GPAX"], OUT_CLASS, OUT_SCALE)
    # print(scale_sit)

    df["SIT_class"] = df["SIT_GPAX"]
    for i in range(len(df)):
        df["SIT_class"][i] = funcClass(df["SIT_GPAX"][i], scale_sit)
        df["SIT_class"] = pd.to_numeric(df["SIT_class"])

    return df


def scaleList(df_list, n_class, scale):
    sorted_df_list = sorted(df_list)
    sample = len(sorted_df_list)
    score_range = sorted_df_list[-1]-sorted_df_list[0]

    if scale == 'custom_in':
        scale_class = customScaleIn()
        return scale_class
    
    if scale == 'custom_out':
        scale_class = customScaleOut()
        return scale_class

    elif scale == 'score_range':
        scale_class = scoreRangeScale(sorted_df_list, score_range, n_class)
        return scale_class

    elif scale == 'sample_size':
        scale_class = sampleSizeScale(sorted_df_list, sample, n_class)
        return scale_class

    elif scale > 0:
        scale_class = [scale, 4]
        return scale_class


def applyClass(df_col, col, scale_list):

    for i in range(len(df_col)):
        df_col[i] = funcClass(df_col[i], scale_list)
    df_col = pd.to_numeric(df_col)


def funcClass(score, scale_list):
    for i in range(len(scale_list)):
        if i == len(scale_list)-1:
            return len(scale_list)
        elif score < scale_list[i]:
            return i+1


def assignXY(df, VAR_NUM):
    X = np.array(df[VAR_NUM])  # .reshape(1,-1)
    y = np.array(df['SIT_GPAX'])
    y_class = np.array(df['SIT_class'])
    return X, y, y_class


def splitData(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8)  # train size 0.8 คือ แบ่งไว้เทรน 80%
    # print("All:", len(X))
    # print("Train: {}, Test: {}".format(len(X_train), len(X_test)))
    return X_train, X_test, y_train, y_test
