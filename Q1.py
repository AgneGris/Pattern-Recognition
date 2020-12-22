#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:08:55 2019

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
from sklearn import mixture
from metric_learn import LMNN
from metric_learn import NCA


face = sio.loadmat('face.mat')
faces = np.array(face['X'])

index_faces = np.array(face['l'])
index_training = index_faces[:,0:320]
index_testing = index_faces[:,320:]
training_set = np.array([])
testing_set = np.array([])


# training set 
training_set = faces[:, 0:320]
# testing set
testing_set = faces[:, 320:]


B = np.zeros([2576,520])
for i in range(520):
    B[:,i] = faces[:,i]/np.linalg.norm(faces[:,i])
    
B_training = B[:,0:320]
B_test = B[:,320:]
    

def KNN(feature_vectors, index_testing, metric_type):
    
    n_neighbours = np.size(feature_vectors, 0)
    
    knn = NearestNeighbors(n_neighbors=n_neighbours, metric = metric_type).fit(feature_vectors)
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

def PCA_eigenvec(A):
    N = np.size(training_set,1)
    S_LD = (1/N)*np.matmul(A.T,A)
    eigvals_ld, eigvecs_ld = np.linalg.eig(S_LD)
    idx_ld = eigvals_ld.argsort()[::-1]
    eigenvalues_temp = eigvals_ld[idx_ld]
    eigenvectors_temp = eigvecs_ld[:,idx_ld]
    
    eigenvectors_ld = np.matmul(A, eigenvectors_temp)
    norm_eigenvec_ld = eigenvectors_ld/np.linalg.norm(eigenvectors_ld, axis=0)
    
    return norm_eigenvec_ld
    
# LMNN A_testing
#mean_training = training_set.mean(axis=1, keepdims=True)
#A = training_set - mean_training
#norm_eigenvec_ld = PCA_eigenvec(A)
#
#A_testing = testing_set - mean_training
#M_array = [16,32,64,128,256]
#
#scores_LMNN_A = np.zeros([5,3])
#
#row = 0
#for M in M_array:
#    
#    U = norm_eigenvec_ld[:, 0:M] 
#    W = np.matmul(U.T, A)
#    W_testing = np.matmul(U.T, A_testing)
#        
#    
#    index_training_array = np.array(index_training[0,:])
#    index_testing_array = np.array(index_testing[0,:])
#    
#    lmnn = LMNN()
#    lmnn.fit(W.T, index_training_array)
#    
#    new_indices, rank_1, rank_10 = KNN(W_testing.T, index_testing_array, lmnn.get_metric())
#    mAP = mAP_calculation(new_indices, index_testing_array)
#    
#    scores_LMNN_A[row,0] = rank_1
#    scores_LMNN_A[row,1] = rank_10
#    scores_LMNN_A[row,2] = mAP
#    
#    row = row+1
#
## LMNN B_testing
#mean_training_B = B_training.mean(axis=1, keepdims=True)
#B = B_training - mean_training_B
#norm_eigenvec_ld_B = PCA_eigenvec(B)
#
#B_testing = B_test - mean_training
#M_array = [16,32,64,128,256]
#
#scores_LMNN_B = np.zeros([5,3])
#
#row = 0
#for M in M_array:
#    
#    U = norm_eigenvec_ld[:, 0:M] 
#    W = np.matmul(U.T, B)
#    W_testing = np.matmul(U.T, B_testing)
#        
#    
#    index_training_array = np.array(index_training[0,:])
#    index_testing_array = np.array(index_testing[0,:])
#    
#    lmnn = LMNN()
#    lmnn.fit(W.T, index_training_array)
#    
#    new_indices, rank_1, rank_10 = KNN(W_testing.T, index_testing_array, lmnn.get_metric())
#    mAP = mAP_calculation(new_indices, index_testing_array)
#    
#    scores_LMNN_B[row,0] = rank_1
#    scores_LMNN_B[row,1] = rank_10
#    scores_LMNN_B[row,2] = mAP
#    
#    row = row+1

#NCA
    
# NCA A_testing
mean_training = training_set.mean(axis=1, keepdims=True)
A = training_set - mean_training
norm_eigenvec_ld = PCA_eigenvec(A)

A_testing = testing_set - mean_training
M_array = [16,32,64,128,256]

scores_NCA_A = np.zeros([5,3])

row = 0
for M in M_array:
    
    U = norm_eigenvec_ld[:, 0:M] 
    W = np.matmul(U.T, A)
    W_testing = np.matmul(U.T, A_testing)
        
    
    index_training_array = np.array(index_training[0,:])
    index_testing_array = np.array(index_testing[0,:])
    
    nca = NCA()
    nca.fit(W.T, index_training_array)
    
    new_indices, rank_1, rank_10 = KNN(W_testing.T, index_testing_array, nca.get_metric())
    mAP = mAP_calculation(new_indices, index_testing_array)
    
    scores_NCA_A[row,0] = rank_1
    scores_NCA_A[row,1] = rank_10
    scores_NCA_A[row,2] = mAP
    
    row = row+1

# NCA B_testing
mean_training_B = B_training.mean(axis=1, keepdims=True)
B = B_training - mean_training_B
norm_eigenvec_ld_B = PCA_eigenvec(B)

B_testing = B_test - mean_training
M_array = [16,32,64,128,256]

scores_NCA_B = np.zeros([5,3])

row = 0
for M in M_array:
    
    U = norm_eigenvec_ld[:, 0:M] 
    W = np.matmul(U.T, B)
    W_testing = np.matmul(U.T, B_testing)
        
    
    index_training_array = np.array(index_training[0,:])
    index_testing_array = np.array(index_testing[0,:])
    
    nca = NCA()
    nca.fit(W.T, index_training_array)
    
    new_indices, rank_1, rank_10 = KNN(W_testing.T, index_testing_array, nca.get_metric())
    mAP = mAP_calculation(new_indices, index_testing_array)
    
    scores_NCA_B[row,0] = rank_1
    scores_NCA_B[row,1] = rank_10
    scores_NCA_B[row,2] = mAP
    
    row = row+1

    
    