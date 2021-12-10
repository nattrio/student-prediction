import numpy as np


def customScaleIn():
    return [2.5, 3, 4]

def customScaleOut():
    return [2.5, 4]


def scoreRangeScale(sorted_df_list, score_range, n_class):
    temp_list = []
    range_per_class = score_range/n_class
    # print("{}/{} = {}".format(score_range, n_class, range_per_class))
    for i in range(n_class-1):
        class_point = sorted_df_list[0]+(range_per_class*(i+1))
        temp_list.append(np.round(class_point, 2))
    temp_list.append(sorted_df_list[-1])
    return temp_list


def sampleSizeScale(sorted_df_list, sample, n_class):
    temp_list = []
    sample_per_class = sample//n_class
    # print("{}//{} = {}".format(sample, n_class, sample_per_class))
    for i in range(n_class-1):
        temp_list.append(sorted_df_list[(i+1)*sample_per_class-1])
    temp_list.append(sorted_df_list[-1])
    return temp_list

