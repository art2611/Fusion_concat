import numpy as np
import math

def minmax_norm(data):
    min = np.amin(data, axis=1)
    max = np.amax(data, axis=1)

    for k in range(data.shape[0]):
        for i in range(data.shape[1]):
            data[k][i] = (data[k][i] - min[k]) / (max[k] - min[k])
    return(data)

def Z_score(data):
    std = np.std(data, axis=1)
    mean = np.mean(data, axis=1)

    for k in range(data.shape[0]):
        for i in range(data.shape[1]):
            data[k][i] = (data[k][i] - mean[k])/ std[k]
    return(data)

def tanh_norm(data):
    std = np.std(data, axis=1)
    mean = np.mean(data, axis=1)

    for k in range(data.shape[0]):
        for i in range(data.shape[1]):
            data[k][i] = 0.5*math.tanh(0.01*(data[k][i] - mean[k])/ std[k])
    return(data)

def l2_norm(data):

    norm_l2 = np.linalg.norm(data, ord=2, axis=1)
    for k in range(data.shape[0]):
        for i in range(data.shape[1]):
            data[k][i] = data[k][i] / norm_l2[k]
    return(data)

def Normalize(query, gallery, norm) :
    print(f" NORM WHICH IS USED : {norm}")
    if norm == "minmax" :
        return(minmax_norm(query), minmax_norm(gallery))
    elif norm == "tanh" :
        return(tanh_norm(query), tanh_norm(gallery))
    elif norm == "l2" :
        return(l2_norm(query), l2_norm(gallery))
    elif norm == "zscore" :
        return(Z_score(query), Z_score(gallery))
    elif norm == "none" :
        return(query, gallery)

