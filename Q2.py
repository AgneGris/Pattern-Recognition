#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 21:54:29 2019

@author: agnes and edocchipi97
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AgglomerativeClustering
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors


face = sio.loadmat('face.mat')
faces = np.array(face['X'])

index_faces = np.array(face['l'])
index_training = index_faces[:,0:320]
index_testing = index_faces[:,320:]
training_set = np.array([])
testing_set = np.array([])

B = np.zeros([2576,520])
for i in range(520):
    B[:,i] = faces[:,i]/np.linalg.norm(faces[:,i])
    
B_training = B[:,0:320]
B_test = B[:,320:]
    

def k_means_rank1(input_training_set, true_labels, class_images):
    
    kmeans = KMeans(n_clusters=32).fit(input_training_set.T)
    labels_k_means = kmeans.labels_
    
    cm = confusion_matrix(true_labels, labels_k_means+1)

    max_confusion_matrix = np.max(cm)
    cost_function = -cm + max_confusion_matrix

    row_ind, col_ind = linear_sum_assignment(cost_function)


    idx_new = np.zeros(320)

    for parse in range(320):
        value = labels_k_means[parse]
        index = np.where(col_ind == value)
        idx_new[parse] = index[0]

    idx_new = idx_new + 1
    idx_new = idx_new.astype(int)

    counter = 0
    for i in range(320):
        if (class_images[0,i] == idx_new[i]):
            counter = counter+1
              
    accuracy_kmeans = counter/320*100 
    
    return accuracy_kmeans




def agglomerative_clustering_rank1(training_set, true_labels, class_images):
    
    agglomerative = AgglomerativeClustering(n_clusters=32).fit(training_set.T)
    labels_agglo = agglomerative.labels_
    
    cm = confusion_matrix(true_labels, labels_agglo+1)

    max_confusion_matrix = np.max(cm)
    cost_function = -cm + max_confusion_matrix

    row_ind, col_ind = linear_sum_assignment(cost_function)


    idx_new = np.zeros(320)

    for parse in range(320):
        value = labels_agglo[parse]
        index = np.where(col_ind == value)
        idx_new[parse] = index[0]

    idx_new = idx_new + 1
    idx_new = idx_new.astype(int)

    counter = 0
    for i in range(320):
        if (class_images[0,i] == idx_new[i]):
            counter = counter+1
              
    accuracy_agglo = counter/320*100 
    
    return agglomerative, idx_new, accuracy_agglo

def compute_centers(classes_labels, X):
    centers = []
    for label in range(32):
        label_index = label+1
        temp = X[:,classes_labels == label_index].mean(1)
        centers.append(temp)

    return np.array(centers)

def compute_distance(centers, X):
    distances = np.zeros([200,32])
    
    for i in range(200):
        image_test = X[:,i];
        for cluster in range(32):
            distance = np.linalg.norm(image_test-centers[cluster,:])
            distances[i,cluster] = distance
    
    return distances

def KNN(feature_vectors, index_testing, metric_type):
    knn = NearestNeighbors(n_neighbors=200, metric = metric_type).fit(feature_vectors)
    distances, indices =knn.kneighbors(feature_vectors)
    
    distances = np.delete(distances, obj = 0, axis = 1)  
    indices = np.delete(indices, obj = 0, axis = 1) 
    
    new_indices = np.zeros([200,199])
    
    for row in range(200):
        for column in range(199):
            image = indices[row,column]
            new_indices[row,column] = index_testing[image]
            
    
    ranks = np.arange(1,200)
    scores = np.zeros(len(ranks))
    
    b = 0
    for rank in ranks:
        count = 0
        
        for n in range(len(index_testing)):
            for c in range(rank):
                if(index_testing[n]  == new_indices[n,c]):
                    count = count+1
                    break
            
        scores[b] = count/len(index_testing)*100
        b = b+1
    
    rank_1 = scores[0]
    rank_10 = scores[9] 

    return new_indices, rank_1, rank_10

def mAP_calculation(NN_indices, index_testing):

    Pk = np.zeros([200,199])
    Rk = np.zeros([200,199])
    
    for n in range(len(index_testing)):
        count= 0
        for c in range(199):
            if index_testing[n] == NN_indices[n,c]:
                count = count + 1
            
            Pk[n,c] = count/(c+1)
            Rk[n,c] = count/9
    
    r_array = np.arange(0,1.1,0.1)
    
    p_inter = np.zeros([len(index_testing), len(r_array)])
    
    for k in range(len(index_testing)):
        Pk_1row = Pk[k,:]
        Rk_1row = Rk[k,:]
        
        r_index = 1
        for r in r_array:
            for rk_index in range(199):
                if Rk_1row[rk_index] > r:
                    p_inter[k,r_index] = max(Pk_1row[rk_index :])
                    break
                
            r_index = r_index+1
    
    AP = np.zeros([200,1])
    for row in range(200):
        AP[row,0] =  np.sum(p_inter[row,:])/11
    
    mean_AP = np.mean(AP)        
            
    return mean_AP
 
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
       
#QA - QB  
# training set 
training_set = faces[:, 0:320]
testing_set = faces[:, 320:]
index_testing = np.array(index_testing[0,:])

# unnormalised data
y_true = index_faces[:,0:320]
y_true = y_true.astype(int)
y_true_array = np.array(y_true[0,:])


accuracy_kmeans_A = k_means_rank1(training_set, y_true_array, index_training)
[agglomerative_A, classes_labels_A, accuracy_agglo_A] = agglomerative_clustering_rank1(training_set, y_true_array, index_training)

accuracy_kmeans_B = k_means_rank1(B_training, y_true_array, index_training)
[agglomerative_B, classes_labels_B, accuracy_agglo_B]= agglomerative_clustering_rank1(B_training, y_true_array, index_training)



#QCa
# find cluster centers
[agglomerative, classes_labels, accuracy_agglo_A] = agglomerative_clustering_rank1(training_set, y_true_array, index_training)
train_centers = compute_centers(classes_labels, training_set)
feature_vectors = compute_distance(train_centers, testing_set)

NN_indices_L2, rank1L2, rank10L2 = KNN(feature_vectors, index_testing, "euclidean")
mean_AP_L2 = mAP_calculation(NN_indices_L2, index_testing)

NN_indices_cos, rank1cos, rank10cos = KNN(feature_vectors, index_testing, "cosine")
mean_AP_cos = mAP_calculation(NN_indices_cos, index_testing)

NN_indices_L1, rank1L1, rank10L1 = KNN(feature_vectors, index_testing, "manhattan")
mean_AP_L1 = mAP_calculation(NN_indices_L1, index_testing)

NN_indices_min, rank1min, rank10min = KNN(feature_vectors, index_testing, "minkowski")
mean_AP_min = mAP_calculation(NN_indices_min, index_testing)

NN_indices_che, rank1che, rank10che = KNN(feature_vectors, index_testing, "chebyshev")
mean_AP_che = mAP_calculation(NN_indices_che, index_testing)

#QCb
inverse_distance = 1/feature_vectors
softmax_feature_vectors = softmax(inverse_distance)

soft_NN_indices_L2, soft_rank1L2, soft_rank10L2 = KNN(softmax_feature_vectors, index_testing, "euclidean")
soft_mean_AP_L2 = mAP_calculation(soft_NN_indices_L2, index_testing)

soft_NN_indices_cos, soft_rank1cos, soft_rank10cos = KNN(softmax_feature_vectors, index_testing, "cosine")
soft_mean_AP_cos = mAP_calculation(soft_NN_indices_cos, index_testing)

soft_NN_indices_L1, soft_rank1L1, soft_rank10L1 = KNN(softmax_feature_vectors, index_testing, "manhattan")
soft_mean_AP_L1 = mAP_calculation(soft_NN_indices_L1, index_testing)

soft_NN_indices_min, soft_rank1min, soft_rank10min = KNN(softmax_feature_vectors, index_testing, "minkowski")
soft_mean_AP_min = mAP_calculation(soft_NN_indices_min, index_testing)

soft_NN_indices_che, soft_rank1che, soft_rank10che = KNN(softmax_feature_vectors, index_testing, "chebyshev")
soft_mean_AP_che = mAP_calculation(soft_NN_indices_che, index_testing)





        



