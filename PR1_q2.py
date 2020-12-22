# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:13:58 2019

@author: agnes
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import timeit
from scipy import ndimage
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D


def NN_class(norm_eigenvec, A, A_testing, range_nn):
    
    percentage_correctly_identified_images = np.array([])
    
    correctly_identified_images = 0
    U = norm_eigenvec
    W = np.matmul(U.T, A)
    W_testing = np.matmul(U.T, A_testing)
    
    
    for index_testing in range(104):
    
        W_test = W_testing[:,index_testing]
        W_difference_norm = np.array([])
        
        for n in range(range_nn):
            difference_vector = W_test - W[:,n]
            W_difference_norm = np.append(W_difference_norm, np.linalg.norm(difference_vector))
    
        
        index_training_tuple = np.where(W_difference_norm == min(W_difference_norm))
        index_training = int(index_training_tuple[0])
        
        class_number_training = class_training_set[index_training]
        class_number_testing = class_testing_set[index_testing]
        
        if (class_number_training == class_number_testing):
            correctly_identified_images = correctly_identified_images + 1
        
    percentage_correct = (correctly_identified_images/104)*100
    percentage_correctly_identified_images = np.append(percentage_correctly_identified_images, percentage_correct)
    
    return percentage_correctly_identified_images

# Find Phi
def find_p3(eigen1, eigen2, mean12, d1,d2):
    
    p_d1 = eigen1[:, 0:d1]
    p_d2 = eigen2[:, 0:d2]
    ph = np.concatenate((p_d1, p_d2), axis = 1)
    X = np.concatenate((ph, mean12), axis = 1)
    phi, R_grim = np.linalg.qr(X)
    
    matrix_product = np.real(np.matmul(np.matmul(phi.T, s_12), phi))
    # Find P3 = Phi*R
    
    R, eigenvalues_R, R_2 = np.linalg.svd(matrix_product)
    p_3 = np.matmul(phi, R)
    
    return p_3

# Question 1

face = sio.loadmat('face.mat')
faces = np.array(face['X'])
index_faces = np.array(face['l'])
training_set = np.array([])
testing_set = np.array([])


# splitting training set from testing set

for i in range(52):
    
    person = faces[:,10*i:10*i+10]
    training = np.array(person[:,0:8])
    testing = np.array(person[:,8:10])
    
    # 80% in training
    training_set = np.hstack([training_set, training]) if training_set.size else training
    
    # 20% in testing
    testing_set = np.hstack([testing_set, testing]) if testing_set.size else testing

#splitting the indeces of the training set and testing set to know the correct classes

class_training_set = np.array([])
class_testing_set = np.array([])
subset_training_index = np.array([])

training_1 = np.array([])
training_2 = np.array([])
training_3 = np.array([])
training_4 = np.array([])


training_index_1 = np.array([])
training_index_2 = np.array([])
training_index_3 = np.array([])
training_index_4 = np.array([])

subset_1 = np.array([])
subset_2 = np.array([])
subset_3 = np.array([])
subset_4 = np.array([])


for i in range(52):
    
    class_index = index_faces[0, 10*i:10*i+10]  
    
    class_training = np.array(class_index[0:8])
    class_testing = np.array(class_index[8:10])
    
    # 80% in training
    class_training_set = np.hstack([class_training_set, class_training]) if class_training_set.size else class_training
    
    # 20% in testing
    class_testing_set = np.hstack([class_testing_set, class_testing]) if class_testing_set.size else class_testing


for row in range(52):
    
    subset_training_index = class_training_set[8*row:8*row+8]
    subset_index = training_set[:,8*row:8*row+8]
    
    subset_training_1 = np.array(subset_training_index[0:2])
    subset_training_2 = np.array(subset_training_index[2:4])
    subset_training_3 = np.array(subset_training_index[4:6])
    subset_training_4 = np.array(subset_training_index[6:8])
    
    # training indexes of training images split in 4 subsets
    
    training_index_1 = np.hstack([training_index_1, subset_training_1]) if training_index_1.size else subset_training_1
    training_index_2 = np.hstack([training_index_2, subset_training_2]) if training_index_2.size else subset_training_2
    training_index_3 = np.hstack([training_index_3, subset_training_3]) if training_index_3.size else subset_training_3
    training_index_4 = np.hstack([training_index_4, subset_training_4]) if training_index_4.size else subset_training_4
    
    subset_1 = np.array(subset_index[:,0:2])
    subset_2 = np.array(subset_index[:,2:4])
    subset_3 = np.array(subset_index[:,4:6])
    subset_4 = np.array(subset_index[:,6:8])
    
    # four subsets with 104 images each
    
    training_1 = np.hstack([training_1, subset_1]) if training_1.size else subset_1
    training_2 = np.hstack([training_2, subset_2]) if training_2.size else subset_2
    training_3 = np.hstack([training_3, subset_3]) if training_3.size else subset_3
    training_4 = np.hstack([training_4, subset_4]) if training_4.size else subset_4
    
    
# PCA with first subset training_1
    
    
    

# Covariance of subsets
    
mean_training_1 = training_1.mean(axis=1, keepdims=True)
mean_training_2 = training_2.mean(axis=1, keepdims=True)
mean_training_3 = training_3.mean(axis=1, keepdims=True)
mean_training_4 = training_4.mean(axis=1, keepdims=True)
mean_training_12 = (np.concatenate((training_1,training_2), axis = 1)).mean(axis=1, keepdims=True)
sum_2sub = np.concatenate((training_1,training_2), axis=1)
mean_training_13 = (np.concatenate((sum_2sub, training_3), axis=1)).mean(axis=1, keepdims=True)

A_1 = training_1 - mean_training_1
A_2 = training_2 - mean_training_2
A_3 = training_3 - mean_training_3
A_4 = training_4 - mean_training_4

A_subset12 = np.concatenate((training_1,training_2), axis=1) - mean_training_12
#A = np.concatenate((training_1,training_2),axis=1) - mean_training_12

mean_training = training_set.mean(axis=1, keepdims=True)
A = training_set - mean_training

mean_testing = testing_set.mean(axis=1, keepdims=True)
A_testing1 = testing_set - mean_training_1

A_testing_subset12 = testing_set - mean_training_12
A_testing_subset13 = testing_set - mean_training_13

A_testing = testing_set - mean_training
N_testing = np.size(testing_set,1)

N = np.size(training_1,1)
S_1 = (1/N)*np.matmul(A_1,A_1.T)
S_2 = (1/N)*np.matmul(A_2,A_2.T)
S_3 = (1/N)*np.matmul(A_3,A_3.T)
S_4 = (1/N)*np.matmul(A_4,A_4.T)

mean_12 = mean_training_1 - mean_training_2
s_12 = S_1/2 + S_2/2 + (N*N/((2*N)^2))*np.matmul(mean_12, mean_12.T)

mean_13 = mean_training_12 - mean_training_3
s_13 = 2/3*s_12 + 1/3*S_3 + 2/3*N*np.matmul(mean_13, mean_13.T)

mean_14 = mean_training_13 - mean_training_4
s_14 = 3/4*s_13 + 1/4*S_4 + 3/4*N*np.matmul(mean_14, mean_14.T)


# Low dimensional PCA on subsets
computational_time_PCA1 = np.array([])
computational_time_PCA2 = np.array([])
computational_time_PCA3 = np.array([])
computational_time_PCA4 = np.array([])

start_PCA1 = timeit.default_timer()

start_IPCA = timeit.default_timer()

S_1_low = (1/N)*np.matmul(A_1.T,A_1)

eigvals_1, eigvecs_1 = np.linalg.eig(S_1_low)
idx_ld = eigvals_1.argsort()[::-1]
eigenvalues_temp = eigvals_1[idx_ld]
eigenvectors_temp = eigvecs_1[:,idx_ld]

eigenvectors_1 = np.matmul(A_1, eigenvectors_temp)
norm_eigenvec_1 = eigenvectors_1/np.linalg.norm(eigenvectors_1, axis=0)
end_PCA1 = timeit.default_timer()
computational_time_PCA1 = np.append(computational_time_PCA1, (end_PCA1 - start_PCA1))

print('Time to calculate eigenvectors and eigenvalues of subset 1 is:', computational_time_PCA1)


#print('Time to calculate eigenvalues and eigenvectors of matrix S_LD is:', end_ELD - start_ELD)

# Low Dimensional PCA on subset 2

start_PCA2 = timeit.default_timer()
S_2_low = (1/N)*np.matmul(A_2.T,A_2)
eigvals_2, eigvecs_2 = np.linalg.eig(S_2_low)
idx_ld = eigvals_2.argsort()[::-1]
eigenvalues_temp_2 = eigvals_2[idx_ld]
eigenvectors_temp_2 = eigvecs_2[:,idx_ld]

eigenvectors_2 = np.matmul(A_2, eigenvectors_temp_2)
norm_eigenvec_2 = eigenvectors_2/np.linalg.norm(eigenvectors_2, axis=0)

end_PCA2 = timeit.default_timer()
computational_time_PCA2 = np.append(computational_time_PCA2, (end_PCA2 - start_PCA2))

print('Time to calculate eigenvectors and eigenvalues of subset 1+2 is:', computational_time_PCA2)

# Low Dimensional PCA on subset 3
start_PCA3 = timeit.default_timer()
S_3_low = (1/N)*np.matmul(A_3.T,A_3)
eigvals_3, eigvecs_3 = np.linalg.eig(S_3_low)
idx_ld = eigvals_3.argsort()[::-1]
eigenvalues_temp_3 = eigvals_3[idx_ld]
eigenvectors_temp_3 = eigvecs_3[:,idx_ld]

eigenvectors_3 = np.matmul(A_3, eigenvectors_temp_3)
norm_eigenvec_3 = eigenvectors_3/np.linalg.norm(eigenvectors_3, axis=0)

end_PCA3 = timeit.default_timer()
computational_time_PCA3 = np.append(computational_time_PCA3, (end_PCA3 - start_PCA3))

print('Time to calculate eigenvectors and eigenvalues of subset 1+2+3 is:', computational_time_PCA3)

# Low Dimensional PCA on subset 4

start_PCA4 = timeit.default_timer()
S_4_low = (1/N)*np.matmul(A_4.T,A_4)
eigvals_4, eigvecs_4 = np.linalg.eig(S_4_low)
idx_ld = eigvals_4.argsort()[::-1]
eigenvalues_temp_4 = eigvals_4[idx_ld]
eigenvectors_temp_4 = eigvecs_4[:,idx_ld]

eigenvectors_4 = np.matmul(A_4, eigenvectors_temp_4)
norm_eigenvec_4 = eigenvectors_4/np.linalg.norm(eigenvectors_4, axis=0)

end_PCA4 = timeit.default_timer()
computational_time_PCA4 = np.append(computational_time_PCA4, (end_PCA4 - start_PCA4))

print('Time to calculate eigenvectors and eigenvalues of subset 1+2+3+4 is:', computational_time_PCA4)



p_subset12 = find_p3(norm_eigenvec_1, norm_eigenvec_2, mean_12,20,20)
p_subset13 = find_p3(p_subset12, norm_eigenvec_3, mean_13, 20, 20)
p_subset14 = find_p3(p_subset13, norm_eigenvec_4, mean_14, 20, 20)

percentage_correctly_identified_incremental = NN_class(p_subset14, A, A_testing, 416)


end_IPCA = timeit.default_timer()
computational_time_IPCA = np.array([])
computational_time_IPCA = np.append(computational_time_IPCA, (end_IPCA - start_IPCA))

print('Time to calculate Incremental PCA is:', computational_time_IPCA)

# do training time
# do reconstruction error

# time needed to compute and then merge one eigenspace model to another: incremental time
# time needed to compute both eigenvalues and merge them



# Nearest Neighbour Classification with only first subset

percentage_correctly_identified_subset1 = NN_class(norm_eigenvec_1, A, A_testing1, 104)
percentage_correctly_identified_subset2 = NN_class(p_subset12, A, A_testing_subset12, 208)
percentage_correctly_identified_subset3 = NN_class(p_subset13, A, A_testing_subset13, 312)

# Find time to do Batch PCA
N_batch = np.size(training_set,1)
S_batch = (1/N)*np.matmul(A,A.T)

# Find eigenvectors and eigenvalues of S
start_E = timeit.default_timer()
eigvals, eigvecs = np.linalg.eig(S_batch)
idx = eigvals.argsort()[::-1]
eigenvalues = eigvals[idx]
eigenvectors = eigvecs[:,idx]
end_E = timeit.default_timer()
print('Time to calculate Batch PCA is:', end_E - start_E)
















