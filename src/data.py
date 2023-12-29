import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import clone 


def rul_calculator(x_len):
    """calculate RUL

    Args:
        x_len (int): lifetime of a bearing

    Returns:
        np.array: normalized RUL of a given bearing 
    """
    rul = x_len - np.arange(x_len) - 1
    rul = rul/x_len
    return rul 

def my_train_test_split(X, y, train_bearings, test_bearing):
    """create train and test data for ML model

    Args:
        X (dict): a dictionary containing features for all bearings
        y (dict): a dictionary containing labels for all bearings
        train_bearings (list): a list of training bearings
        test_bearing (str): test bearing

    Returns:
        4 arrays repressenting train and test, features and labels, respectively. 
    """
    X_train, y_train = np.vstack(list(map(X.get, train_bearings))), np.hstack(list(map(y.get, train_bearings)))
    X_test, y_test = X[test_bearing], y[test_bearing]
    return X_train, X_test, y_train, y_test


def my_scaler(X, train_bearings, scaler=MinMaxScaler()):
    """scale data  

    Args:
        X (dict): a dictionary containing features for all bearings
        train_bearings (list): a list of training bearings
        scaler (sklearn scaler, optional): Defaults to MinMaxScaler().

    Returns:
        dict: a dictionary containing scaled features for all bearings
    """
    X_train = np.vstack(list(map(X.get, train_bearings)))
    scaler.fit(X_train)
    X_scaled = {}
    for b in X.keys():
        X_scaled[b] = scaler.transform(X[b])
    return X_scaled


def my_cluster(X, train_bearings, random_state):
    """cluster data

    Args:
        X (dict): a dictionary containing features for all bearings
        train_bearings (list): a list of training bearings
        random_state (_type_): _description_

    Returns:
        dict: a dictionary containing clusters for all bearings
    """
    X_train = np.vstack(list(map(X.get, train_bearings)))
    kmeans = KMeans(n_clusters=2, algorithm="elkan", random_state=random_state)
    kmeans.fit(X_train[:,:10])
    X_clusters = {}
    for b in X.keys():
        X_clusters[b] = kmeans.predict(X[b][:,:10])
    return X_clusters

def find_transition_time(cluster, th=150):
    """given the clusters, find the transition_time

    Args:
        cluster (np.array): an array of clusters for a given bearing
        th (int, optional): A threshold untill which we ignore any changes in clusters. Defaults to 150.

    Returns:
        int: index between th and cluster.shape[0] indicating the transition point from healthy to unhealthy
    """
    idx = np.where(cluster[th:] != cluster[0])[0][0]
    return idx + th 

def data_cutter(X, y, transition_times):
    """cut data into healthy and unhealthy partiotions

    Args:
        X (dict): a dictionary containing features for all bearings
        y (dict): a dictionary containing labels for all bearings
        transition_times (dict): a dictionary containing transition_times for all bearings

    Returns:
        X_unhealthy (dict): a dictionary containing features for all bearings during their unhealthy state
        y_unhealthy (dict): a dictionary containing labels for all bearings during their unhealthy state
    """
    X_healthy = {}
    X_unhealthy = {}
    y_healthy = {}
    y_unhealthy = {}
    for b in X.keys():
        X_healthy[b] = X[b][:transition_times[b]]
        y_healthy[b] = y[b][:transition_times[b]]
        X_unhealthy[b] = X[b][transition_times[b]:]
        # y_unhealthy[b] = y[b][transition_times[b]:]
        y_unhealthy[b] = rul_calculator(X_unhealthy[b].shape[0]) #redefining RUL based on the transition point
    return X_healthy, X_unhealthy, y_healthy, y_unhealthy



def op_scaler(X, op, train_bearings, n_bins=20, random_state=2023):
    """scale data wrt operating condition

    Args:
        X (dict): a dictionary containing features for all bearings
        op (dict): a dictionary containing operating conditions for all bearings
        train_bearings (list): a list of training bearings
        n_bins (int, optional): nuber of bins for dicretizing 10dimensional op. Defaults to 20.
        random_state (int, optional): _description_. Defaults to 2023.

    Returns:
        dict: a dictionary containing scaled features for all bearings
    """
    X_train = np.vstack(list(map(X.get, train_bearings)))
    op_train = np.vstack(list(map(op.get, train_bearings)))
    pca = PCA(n_components=1, random_state=random_state)
    op_train_lowd = pca.fit_transform(op_train)
    dis = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None, random_state=random_state)
    op_train_lowd_dis = dis.fit_transform(op_train_lowd)    
    scalers = []
    for i in range(n_bins):
        scaler = clone(MinMaxScaler())
        idx = np.where(op_train_lowd_dis==i)[0]
        scaler.fit(X_train[idx])
        scalers.append(scaler)
    X_scaled = {}
    for b in X.keys():
        op_lowd = pca.transform(op[b])
        op_lowd_dis = dis.transform(op_lowd)  
        X_scaled[b] = X[b]
        for i in range(n_bins):
            idx = np.where(op_lowd_dis==i)[0]
            if len(idx) > 0:
                X_scaled[b][idx] = scalers[i].transform(X[b][idx])
    return X_scaled