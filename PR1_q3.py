# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:33:29 2019

@author: agnes
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import timeit
from scipy import ndimage
import matplotlib.lines as mlines
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import random


# Question 3

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

#splitting the indeces of the training set and testing set to knoe the correct classes

class_training_set = np.array([])
class_testing_set = np.array([])#

for i in range(52):
    
    class_index = index_faces[0, 10*i:10*i+10]  
    
    class_training = np.array(class_index[0:8])
    class_testing = np.array(class_index[8:10])
    
    # 80% in training
    class_training_set = np.hstack([class_training_set, class_training]) if class_training_set.size else class_training
    
    # 20% in testing
    class_testing_set = np.hstack([class_testing_set, class_testing]) if class_testing_set.size else class_testing

# Perform LDA
# calculate mean of specific classes
    
mean_class_image = np.zeros((2576,52))

for classes in range(52):
    
    index_classes = np.array([])
    
    for index in range(len(class_training_set)):
        if (class_training_set[index] == classes + 1):
            index_classes = np.append(index_classes,index)
            index_classes = index_classes.astype(int)
    
    
    values = np.array([])
    
    for value in index_classes:
        values = np.append(values, training_set[:,value])
        
    class_values_split = np.array([])
        
    for i in range(8):
    
        class_values = values[2576*i:2576*i+2576]
        class_values_split = np.vstack((class_values_split, class_values)) if class_values_split.size else class_values
    
    class_values_reshape = class_values_split.T
    mean_class_reshape = class_values_reshape.mean(axis=1, keepdims=True)
    
    
    mean_class_image[:,classes] = mean_class_reshape[:,0]  

# Total mean    
mean_training = training_set.mean(axis=1, keepdims=True)

# Between class scatter matrix
mean_between = mean_class_image - mean_training
class_scatter_between = np.matmul(mean_between, mean_between.T)
rank_between = np.linalg.matrix_rank(class_scatter_between)

# Within class scatter matrix
training_fill = np.array([])

for classes in range(52):
    
    index_classes = np.array([])
    
    for index in range(len(class_training_set)):
        if (class_training_set[index] == classes + 1):
            index_classes = np.append(index_classes,index)
            index_classes = index_classes.astype(int)

    for column in index_classes:
        training_set_class = training_set[:, column] - mean_class_image[:, classes]
        training_fill = np.vstack((training_fill, training_set_class)) if training_fill.size else training_set_class
            
training_fill_reshape = training_fill.T
        
class_scatter_within = np.matmul(training_fill_reshape, training_fill_reshape.T)
rank_within = np.linalg.matrix_rank(class_scatter_within)


# Low dimensional PCA
mean_training_PCA = training_set.mean(axis=1,keepdims=True)
A = training_set - mean_training_PCA
N = np.size(training_set,1)
S_LD = (1/N)*np.matmul(A.T,A)

eigvals_ld, eigvecs_ld = np.linalg.eig(S_LD)
idx_ld = eigvals_ld.argsort()[::-1]
eigenvalues_temp = eigvals_ld[idx_ld]
eigenvectors_temp = eigvecs_ld[:,idx_ld]

eigenvectors_ld = np.matmul(A, eigenvectors_temp)
norm_eigenvec_ld = eigenvectors_ld/np.linalg.norm(eigenvectors_ld, axis=0)

# Eigenvectors non zero
eigenvec_ld_nonzero = norm_eigenvec_ld[:,0:415]
M = [50,100,150,200,250,300,350] 
M_lda = [5, 10, 15, 20, 30, 40, 50]

mean_testing = testing_set.mean(axis=1, keepdims=True)
A_testing = testing_set - mean_training
N_testing = np.size(testing_set,1)

percentage_correctly_identified_images = np.zeros([len(M), len(M_lda)])

for eig_index in range(len(M)):
    
    omega_pca = eigenvec_ld_nonzero[:,0:M[eig_index]]
    
    num = np.matmul(omega_pca.T, np.matmul(class_scatter_between, omega_pca))
    denum = np.matmul(omega_pca.T, np.matmul(class_scatter_within, omega_pca))
    J = np.matmul(np.linalg.inv(denum), num)
        
    gen_eigvals_ld, gen_eigvecs_ld = np.linalg.eig(J)
    gen_idx_ld = gen_eigvals_ld.argsort()[::-1]
    gen_eigenvalues_temp = gen_eigvals_ld[gen_idx_ld]
    gen_eigenvectors_temp = gen_eigvecs_ld[:,gen_idx_ld]
    
    for index_mlda in range(len(M_lda)):
        omega_lda = gen_eigenvectors_temp[:, 0:M_lda[index_mlda]]
        omega_opt = np.matmul(omega_pca, omega_lda)
        
        # Nearest Neighbour Classification 
        W = np.matmul(omega_opt.T,A) 
        W_testing = np.matmul(omega_opt.T, A_testing)
        correctly_identified_images = 0
        
        for index_testing in range(104):
            W_test = W_testing[:,index_testing]
            W_difference_norm = np.array([])
                
            for n in range(416):
                difference_vector = W_test - W[:,n]
                W_difference_norm = np.append(W_difference_norm, np.linalg.norm(difference_vector))

            index_training_tuple = np.where(W_difference_norm == min(W_difference_norm))
            index_training = int(index_training_tuple[0])
                
            class_number_training = class_training_set[index_training]
            class_number_testing = class_testing_set[index_testing]
                
            if (class_number_training == class_number_testing): 
                correctly_identified_images = correctly_identified_images + 1
                
        percentage_correct = (correctly_identified_images/104)*100
        percentage_correctly_identified_images[eig_index, index_mlda] = percentage_correct      


# do a heatmap?

# Confusion matrix
M_lda_best = 15
M_pca_best = 150

omega_pca = eigenvec_ld_nonzero[:,0:M_pca_best]
num = np.matmul(omega_pca.T, np.matmul(class_scatter_between, omega_pca))
denum = np.matmul(omega_pca.T, np.matmul(class_scatter_within, omega_pca))
J = np.matmul(np.linalg.inv(denum), num)
        
gen_eigvals_ld, gen_eigvecs_ld = np.linalg.eig(J)
gen_idx_ld = gen_eigvals_ld.argsort()[::-1]
gen_eigenvalues_temp = gen_eigvals_ld[gen_idx_ld]
gen_eigenvectors_temp = gen_eigvecs_ld[:,gen_idx_ld]   

omega_lda = gen_eigenvectors_temp[:, 0:M_lda_best]
omega_opt = np.matmul(omega_pca, omega_lda)

W = np.real(np.matmul(omega_opt.T,A)) 
W_testing = np.real(np.matmul(omega_opt.T, A_testing))


confusion_matrix = np.zeros([52,52])

for index_testing in range(104):
    
    W_test = W_testing[:,index_testing]
    W_difference_norm = np.array([])
        
    for n in range(416):
        difference_vector = W_test - W[:,n]
        W_difference_norm = np.append(W_difference_norm, np.linalg.norm(difference_vector))

        
    index_training_tuple = np.where(W_difference_norm == min(W_difference_norm))
    index_training = int(index_training_tuple[0])
        
    predicted_class = class_training_set[index_training]
    true_class = class_testing_set[index_testing]
    
    confusion_matrix[true_class - 1, predicted_class - 1] = confusion_matrix[true_class - 1, predicted_class - 1] + 1



figure1 = plt.figure(figsize = (12,8))
ax = figure1.add_subplot(1,1,1)
img = ax.matshow(confusion_matrix, cmap = 'YlGnBu')

alpha = np.arange(52)
ax.set_xticks(alpha + 0.5);
ax.set_xticklabels(alpha + 1, rotation = 'vertical');

ax.set_yticks(alpha + 0.5);
ax.set_yticklabels(alpha + 1);

ax.grid()   

# Nearest neighbour examples 

bases = 415
U = eigenvec_ld_nonzero[:, 0:bases]  
k = 1

# image classes 13, 18, 52
# classe 13 alays correct, class 18 one yes, one no, class 52 is always confused with class 38

index_testing = [24,25,34,35,102,103]
 
figure2 = plt.figure()

for i in range(len(index_testing)):
    
    W_test = W_testing[:,index_testing[i]]
    W_difference_norm = np.array([])    
        
    for n in range(416):
        difference_vector = W_test - W[:,n]
        W_difference_norm = np.append(W_difference_norm, np.linalg.norm(difference_vector))
    
    index_training_tuple = np.where(W_difference_norm == min(W_difference_norm))
    index_training = int(index_training_tuple[0])            
    
    testing_image26 = np.reshape(testing_set[:, index_testing[i]], (46,56))
    rotated_image26 = ndimage.rotate(testing_image26,270)
    plt.subplot(3,4,k)
    plt.imshow(rotated_image26, cmap = 'gray')
    
    training_image_NN = np.reshape(training_set[:, index_training], (46,56))
    rotated_image_training = ndimage.rotate(training_image_NN,270)
    plt.subplot(3,4,k+1)
    plt.imshow(rotated_image_training, cmap = 'gray')
    
    k = k+2
    
    test_image_class_number_training = class_training_set[index_training]
    test_image_class_number_testing = class_testing_set[index_testing[i]]
    


# 3 Dimensional Subspace
bases = 415
U = eigenvec_ld_nonzero[:, 0:bases]  

# 5 is correctly identified, 18 and 20 are confused

W_training_5 = W[0:3, 32:40]
W_training_18 = W[0:3, 136:144]
W_training_20 = W[0:3, 152:160]

W_testing_5 = W_testing[0:3, 8:10]
W_testing_18 = W_testing[0:3, 34:36]
W_testing_20 = W_testing[0:3, 38:40]

figure3 = plt.figure(figsize = (12,8))
ax = Axes3D(figure3)

for image_scatter in range(8):
    
    ax.scatter(W_training_5[1,image_scatter], W_training_5[0,image_scatter], W_training_5[2,image_scatter], c='g', marker='x', s=36)
    ax.scatter(W_training_18[1,image_scatter], W_training_18[0,image_scatter], W_training_18[2,image_scatter], c='b', marker='x', s=36)
    ax.scatter(W_training_20[1,image_scatter], W_training_20[0,image_scatter], W_training_20[2,image_scatter], c='r', marker='x', s=36)

for image_scatter_testing in range(2):
    
    ax.scatter(W_testing_5[1,image_scatter_testing], W_testing_5[0,image_scatter_testing], W_testing_5[2,image_scatter_testing], c='g', marker='o', s=36)
    ax.scatter(W_testing_18[1,image_scatter_testing], W_testing_18[0,image_scatter_testing], W_testing_18[2,image_scatter_testing], c='b', marker='o', s=36)
    ax.scatter(W_testing_20[1,image_scatter_testing], W_testing_20[0,image_scatter_testing], W_testing_20[2,image_scatter_testing], c='r', marker='o', s=36)

#ax.set_xlim3d(1000,-1000)
#ax.set_ylim3d(-1500,2000)
#ax.set_zlim3d(-2000,1000)

def rotate(angle):
    ax.view_init(azim=angle)
    
rot_animation = animation.FuncAnimation(figure3, rotate, frames=np.arange(0,362,2),interval=100)

training_class_5 = mlines.Line2D([], [], color='g', marker='x', linestyle='None',
                          markersize=8, label='Training Class 5')

training_class_18 = mlines.Line2D([], [], color='b', marker='x', linestyle='None',
                          markersize=8, label='Training Class 18')

training_class_20 = mlines.Line2D([], [], color='r', marker='x', linestyle='None',
                          markersize=8, label='Training Class 20')

testing_class_5 = mlines.Line2D([], [], color='g', marker='o', linestyle='None',
                          markersize=8, label='Training Class 5')

testing_class_18 = mlines.Line2D([], [], color='b', marker='o', linestyle='None',
                          markersize=8, label='Testing Class 18')

testing_class_20 = mlines.Line2D([], [], color='r', marker='o', linestyle='None',
                          markersize=8, label='Testing Class 20')


ax.legend(handles=[training_class_5, training_class_18, training_class_20, testing_class_5, testing_class_18, testing_class_20])


# PCA-LDA ensemble

# set T - number of subspaces
# create eig(S) = W for each subset
# take top first eigenvectors M0 and add M1 eigenctors
# the number M1 is fixed but the eigenvectors are taken randomly
# have different Wpca_t
# set M_lda to a constant (best performing)
# use SB and SW to calculate W_lda
# fusion
    
W_original = eigenvec_ld_nonzero

T = 4
M0 = 50
M1 = 100

# create a dictionary for the different W_pca_t
W_pca_t={}

W_pca = W_original[:, 0:M0]
W_pca_remaining = W_original[:, M0+1:-1]

for t in range(T):
    
    indices = random.sample(range(0,362), M1)
    M1_vectors = W_pca_remaining[:, indices]
    
    W_pca_t["W_pca_t{0}".format(t)] = np.concatenate((W_pca, M1_vectors), axis = 1)

# perform LDA on all subspaces
class_number_training = {}
class_number_testing = {}

for key in W_pca_t:
    
    class_number_training_array = np.array([])    
    class_number_testing_array = np.array([])   

    W_random = W_pca_t[key]
    num = np.matmul(W_random.T, np.matmul(class_scatter_between, W_random))
    denum = np.matmul(W_random.T, np.matmul(class_scatter_within, W_random))
    J = np.matmul(np.linalg.inv(denum), num)
    
    gen_eigvals_ld, gen_eigvecs_ld = np.linalg.eig(J)
    gen_idx_ld = gen_eigvals_ld.argsort()[::-1]
    gen_eigenvalues_temp = gen_eigvals_ld[gen_idx_ld]
    gen_eigenvectors_temp = gen_eigvecs_ld[:,gen_idx_ld]
    
    M_lda = 15
    omega_lda = gen_eigenvectors_temp[:, 0:M_lda]
    omega_opt = np.matmul(W_random, omega_lda)
   
    # Nearest Neighbour Classification 
    W = np.matmul(omega_opt.T,A) 
    W_testing = np.matmul(omega_opt.T, A_testing)
    correctly_identified_images = 0
    
    for index_testing in range(104):
        W_test = W_testing[:,index_testing]
        W_difference_norm = np.array([])
        
        for n in range(416):
            difference_vector = W_test - W[:,n]
            W_difference_norm = np.append(W_difference_norm, np.linalg.norm(difference_vector))

        index_training_tuple = np.where(W_difference_norm == min(W_difference_norm))
        index_training = int(index_training_tuple[0])
                
        class_number_training_array = np.append(class_number_training_array,class_training_set[index_training])      
        class_number_testing_array = np.append(class_number_testing_array,class_testing_set[index_testing])  
        
    class_number_training["class_number_training{0}".format(key)] = class_number_training_array
    class_number_testing["class_number_testing{0}".format(key)] = class_number_testing_array
    
