# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:28:18 2019

@author: agnes and edo
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import timeit
from scipy import ndimage
import matplotlib.lines as mlines
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


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



# Find mean image and remove it from the training set
mean_training = training_set.mean(axis=1, keepdims=True)
A = training_set - mean_training


# Find covariance
N = np.size(training_set,1)

start_S = timeit.default_timer()
S = (1/N)*np.matmul(A,A.T)
end_S = timeit.default_timer()
print('Time to calculate covariance matrix S is:', end_S- start_S)
np.linalg.matrix_rank(S)

# Find eigenvectors and eigenvalues of S
start_E = timeit.default_timer()
eigvals, eigvecs = np.linalg.eig(S)
idx = eigvals.argsort()[::-1]
eigenvalues = eigvals[idx]
eigenvectors = eigvecs[:,idx]
end_E = timeit.default_timer()
print('Time to calculate eigenvectors and eigenvalues of matrix S is:', end_E - start_E)

# mean training set

eigenface_training = np.real(mean_training[:,0])
test_eigenface_training = ndimage.rotate(np.reshape(eigenface_training, (46,56)), 270)
figure = plt.figure()
plt.imshow(test_eigenface_training, cmap = 'gray')


# plot eigenfaces 1 image PCA for different Ms
#eigenvectors1 = eigenvectors[:,0:1]
#eigenvectors10 = eigenvectors[:,0:10]
#eigenvectors50 = eigenvectors[:,0:50]
#eigenvectors100 = eigenvectors[:,0:100]
#eigenvectors200 = eigenvectors[:,0:200]
#eigenvectors300 = eigenvectors[:,0:300]

eigenface_50_1 = np.real(eigenvectors[:,0])
test_eigenface_50_1 = ndimage.rotate(np.reshape(eigenface_50_1, (46,56)), 270)
figure = plt.figure()
plt.subplot(231)
plt.title('M=1',fontweight="bold")
plt.imshow(test_eigenface_50_1, cmap = 'gray')

eigenface_50_10 = np.real(eigenvectors[:,9])
test_eigenface_50_10 = ndimage.rotate(np.reshape(eigenface_50_10, (46,56)), 270)
plt.subplot(232)
plt.title('M=10',fontweight="bold")
plt.imshow(test_eigenface_50_10, cmap = 'gray')

eigenface_50_50 = np.real(eigenvectors[:,49])
test_eigenface_50_50 = ndimage.rotate(np.reshape(eigenface_50_50, (46,56)), 270)
plt.subplot(233)
plt.title('M=50',fontweight="bold")
plt.imshow(test_eigenface_50_50, cmap = 'gray')

eigenface_50_100 = np.real(eigenvectors[:,99])
test_eigenface_50_100 = ndimage.rotate(np.reshape(eigenface_50_100, (46,56)), 270)
plt.subplot(234)
plt.title('M=100',fontweight="bold")
plt.imshow(test_eigenface_50_100, cmap = 'gray')

eigenface_50_200 = np.real(eigenvectors[:,199])
test_eigenface_50_200 = ndimage.rotate(np.reshape(eigenface_50_200, (46,56)), 270)
plt.subplot(235)
plt.title('M=200',fontweight="bold")
plt.imshow(test_eigenface_50_200, cmap = 'gray')

eigenface_50_300 = np.real(eigenvectors[:,299])
test_eigenface_50_300 = ndimage.rotate(np.reshape(eigenface_50_300, (46,56)), 270)
plt.subplot(236)
plt.title('M=300',fontweight="bold")
plt.imshow(test_eigenface_50_300, cmap = 'gray')



figure = plt.figure(figsize=(10,6))
plt.subplot(121)
plt.ticklabel_format(style = 'sci', axis ='y', scilimits=(0,0))
plt.xlim(-10,420)
plt.yticks(np.arange(0, 1000000, 100000))
plt.xlabel('Index of eigenvalue')
plt.ylabel('Magnitude of eigenvalue')
figure = plt.plot(eigenvalues, linewidth=2)
#plt.savefig('eigenvalues.png')

# zoom into the drop
plt.subplot(122)
plt.xlim(380, 420)
plt.ylim(45, 300)
plt.xlabel('Index of eigenvalue')
#plt.ylabel('Magnitude of eigenvalue')
figure2 = plt.plot(eigenvalues, linewidth=2)
#plt.savefig('zoomin.png')

variance1 = eigenvalues[0]/np.sum(eigenvalues)
variance2 = eigenvalues[1]/np.sum(eigenvalues)
variance3 = eigenvalues[2]/np.sum(eigenvalues)
# Low-dimensional computation
start_SLD = timeit.default_timer()
S_LD = (1/N)*np.matmul(A.T,A)
end_SLD = timeit.default_timer()
print('Time to calculate covariance matrix S_LD is:', end_SLD - start_SLD)

start_ELD = timeit.default_timer()
eigvals_ld, eigvecs_ld = np.linalg.eig(S_LD)
idx_ld = eigvals_ld.argsort()[::-1]
eigenvalues_temp = eigvals_ld[idx_ld]
eigenvectors_temp = eigvecs_ld[:,idx_ld]

eigenvectors_ld = np.matmul(A, eigenvectors_temp)
norm_eigenvec_ld = eigenvectors_ld/np.linalg.norm(eigenvectors_ld, axis=0)

end_ELD = timeit.default_timer()
print('Time to calculate eigenvalues and eigenvectors of matrix S_LD is:', end_ELD - start_ELD)


# plot eigenfaces PCA

eigenface50 = np.real(eigenvectors[:,0])
test_eigenface = ndimage.rotate(np.reshape(eigenface50, (46,56)), 270)
figure = plt.figure()
plt.subplot(245)
plt.imshow(test_eigenface, cmap = 'gray')

eigenface100 = np.real(eigenvectors[:,1])
test_eigenface2 = ndimage.rotate(np.reshape(eigenface100, (46,56)), 270)
plt.subplot(246)
plt.imshow(test_eigenface2, cmap = 'gray')

eigenface70 = np.real(eigenvectors[:,2])
test_eigenface3 = ndimage.rotate(np.reshape(eigenface70, (46,56)), 270)
plt.subplot(247)
plt.imshow(test_eigenface3, cmap = 'gray')

eigenface20 = np.real(eigenvectors[:,3])
test_eigenface4 = ndimage.rotate(np.reshape(eigenface20, (46,56)), 270)
plt.subplot(248)
plt.imshow(test_eigenface4, cmap = 'gray')

# plot eigenfaces LD PCA


eigenface50l = np.real(norm_eigenvec_ld[:,0])
test_eigenfacel = ndimage.rotate(np.reshape(eigenface50l, (46,56)), 270)
plt.subplot(241)
plt.imshow(test_eigenfacel, cmap = 'gray')

eigenface100l = np.real(norm_eigenvec_ld[:,1])
test_eigenface2l = ndimage.rotate(np.reshape(eigenface100l, (46,56)), 270)
plt.subplot(242)
plt.imshow(test_eigenface2l, cmap = 'gray')

eigenface70l = np.real(norm_eigenvec_ld[:,2])
test_eigenface3l = ndimage.rotate(np.reshape(eigenface70l, (46,56)), 270)
plt.subplot(243)
plt.imshow(test_eigenface3l, cmap = 'gray')

eigenface20l = np.real(norm_eigenvec_ld[:,3])
test_eigenface4l = ndimage.rotate(np.reshape(eigenface20l, (46,56)), 270)
plt.subplot(244)
plt.imshow(test_eigenface4l, cmap = 'gray')


# Compare the eigenvectors 

eigenvec_nonzero = eigenvectors[:,0:415]
eigenvec_ld_nonzero = norm_eigenvec_ld[:,0:415]
angle_array = np.array([])
dist_array = np.array([])

for k in range(415):
    vecs = np.dot(eigenvec_nonzero[:,k], eigenvec_ld_nonzero[:,k])/(np.linalg.norm(eigenvec_nonzero[:,k])*np.linalg.norm(eigenvec_ld_nonzero[:,k]))
    angle_j = np.arccos(vecs)
    angle_array = np.append(angle_array, np.angle(angle_j))
    
    dist = np.linalg.norm(eigenvec_ld_nonzero[:,k] - eigenvec_nonzero[:,k])
    dist_array = np.append(dist_array, dist)
    
mean_angle = np.mean(angle_array)
mean_dist = np.mean(dist_array)


# Compare the euclidean distance



# Use Low Dimensional PCA technique

# Face image recontruction

# projected data/weight matrix

training_reconstruction_error_array = np.array([])
testing_reconstruction_error_array = np.array([])


# Find mean testing image and remove it from the testing set
mean_testing = testing_set.mean(axis=1, keepdims=True)
A_testing = testing_set - mean_training
N_testing = np.size(testing_set,1)

for bases in range(416):

    U = eigenvec_ld_nonzero[:, 0:bases]
    
    W = np.matmul(U.T, A)
    UW = np.matmul(U, W)
    
    w_testing = np.matmul(U.T, A_testing)
    UW_testing = np.matmul(U, w_testing)
    
    training_reconstructed_image = mean_training + UW
    testing_reconstructed_image = mean_training + UW_testing
    
    training_error_array = np.array([])
    testing_error_array = np.array([])
    
    for training_images in range(416):    
        training_difference = training_set[:,training_images] - training_reconstructed_image[:,training_images]
        training_error_array = np.append(training_error_array, np.linalg.norm(training_difference))
    
    for testing_images in range(104):
        testing_difference = testing_set[:,testing_images] - testing_reconstructed_image[:,testing_images]
        testing_error_array = np.append(testing_error_array, np.linalg.norm(testing_difference))
        
    
    #squared_error_array = np.square(error_array)     
        
    training_reconstruction_error_array = np.append(training_reconstruction_error_array,(1/N)* np.sum(training_error_array))
    testing_reconstruction_error_array = np.append(testing_reconstruction_error_array,(1/N_testing)* np.sum(testing_error_array))


# reconstruction with Full dimenstional PCA

#
#U_recon_fd = eigenvectors
#W_testing_recon_fd = np.matmul(U_recon_fd.T, A_testing)
#UW_testing_recon_fd = np.matmul(U_recon_fd, W_testing_recon_fd)
#    
#testing_reconstructed_image_recon_fd = mean_training + UW_testing_recon_fd
#int_image_recon_fd = testing_reconstructed_image_recon_fd.astype(np.uint8)
#test_image_recon_fd = ndimage.rotate(np.reshape(int_image_recon_fd[:, 50], (46,56)), 270)
#
#plt.subplot(245)
#plt.imshow(test_image_recon_fd, cmap = 'gray')

#
#testing_reconstructed_image_recon_fd2 = mean_training + UW_testing_recon_fd
#int_image_recon_fd2 = testing_reconstructed_image_recon_fd2.astype(np.uint8)
#test_image_recon_fd2 = ndimage.rotate(np.reshape(int_image_recon_fd2[:, 100], (46,56)), 270)
#
#plt.subplot(246)
#plt.imshow(test_image_recon_fd2, cmap = 'gray')
    
    
# reconstruction with Low dimensional PCA

#figure11 = plt.figure()   
#U_recon = eigenvec_ld_nonzero
#W_testing_recon = np.matmul(U_recon.T, A_testing)
#UW_testing_recon = np.matmul(U_recon, W_testing_recon)
#    

#plt.subplot(245)
#plt.imshow(UW_testing_recon, cmap = 'gray')
##
#testing_reconstructed_image_recon2 = mean_training + UW_testing_recon
#int_image_recon2 = testing_reconstructed_image_recon2.astype(np.uint8)
#test_image_recon2 = ndimage.rotate(np.reshape(int_image_recon2[:, 100], (46,56)), 270)
#plt.subplot(246)
#plt.imshow(test_image_recon2, cmap = 'gray')
#
#
#testing_reconstructed_image_recon3 = mean_training + UW_testing_recon
#int_image_recon3 = testing_reconstructed_image_recon3.astype(np.uint8)
#test_image_recon3 = ndimage.rotate(np.reshape(int_image_recon3[:, 70], (46,56)), 270)
#plt.subplot(247)
#plt.imshow(test_image_recon3, cmap = 'gray')
#
#testing_reconstructed_image_recon4 = mean_training + UW_testing_recon
#int_image_recon4 = testing_reconstructed_image_recon4.astype(np.uint8)
#test_image_recon4 = ndimage.rotate(np.reshape(int_image_recon4[:, 20], (46,56)), 270)
#plt.subplot(248)
#plt.imshow(test_image_recon4, cmap = 'gray')
 


# plot the reconstruction errror 
figure3 = plt.figure()
plt.plot(training_reconstruction_error_array, label='training set')
plt.plot(testing_reconstruction_error_array, label ='testing set')
plt.legend()
plt.xlabel('Number of eigenbases used')
plt.ylabel('Reconstruction error')

# plot the reconstructed image for different bases

M_values = [10, 100, 300]

# Reconstructed_image M=10    
U_10 = eigenvec_ld_nonzero[:, 0:M_values[0]]
W_testing = np.matmul(U_10.T, A_testing)
UW_testing = np.matmul(U_10, W_testing)
    
testing_reconstructed_image_10 = mean_training + UW_testing

# Reconstructed_image M=100    
U_100 = eigenvec_ld_nonzero[:, 0:M_values[1]]
W_testing = np.matmul(U_100.T, A_testing)
UW_testing = np.matmul(U_100, W_testing)
    
testing_reconstructed_image_100 = mean_training + UW_testing
        
# Reconstructed_image M=300    
U_300 = eigenvec_ld_nonzero[:, 0:M_values[2]]
W_testing = np.matmul(U_300.T, A_testing)
UW_testing = np.matmul(U_300, W_testing)
    
testing_reconstructed_image_300 = mean_training + UW_testing  


figure5 = plt.figure()

int_image_10 = testing_reconstructed_image_10.astype(np.uint8)
int_image_100 = testing_reconstructed_image_100.astype(np.uint8)
int_image_300 = testing_reconstructed_image_300.astype(np.uint8)

# image 50 of testing set 

test_image_10 = ndimage.rotate(np.reshape(int_image_10[:, 50], (46,56)), 270)
test_image_100 = ndimage.rotate(np.reshape(int_image_100[:, 50], (46,56)), 270)
test_image_300 = ndimage.rotate(np.reshape(int_image_300[:, 50], (46,56)), 270)

testing_image = np.reshape(testing_set[:, 50], (46,56))
rotated_image_120 = ndimage.rotate(testing_image,270)

plt.subplot(341)
plt.title('M = 10', fontweight='bold')
plt.imshow(test_image_10, cmap = 'gray')

plt.subplot(342)
plt.title('M = 100', fontweight='bold')
plt.imshow(test_image_100, cmap = 'gray')

plt.subplot(343)
plt.title('M = 300', fontweight='bold')
plt.imshow(test_image_300, cmap = 'gray')

plt.subplot(344)
plt.title('Original', fontweight='bold')
plt.imshow(rotated_image_120, cmap = 'gray')

# image 10 of testing set

test_image10_10 = ndimage.rotate(np.reshape(int_image_10[:, 10], (46,56)), 270)
test_image10_100 = ndimage.rotate(np.reshape(int_image_100[:, 10], (46,56)), 270)
test_image10_300 = ndimage.rotate(np.reshape(int_image_300[:, 10], (46,56)), 270)

testing_image10 = np.reshape(testing_set[:, 10], (46,56))
rotated_image10 = ndimage.rotate(testing_image10,270)

plt.subplot(345)
plt.imshow(test_image10_10, cmap = 'gray')

plt.subplot(346)
plt.imshow(test_image10_100, cmap = 'gray')

plt.subplot(347)
plt.imshow(test_image10_300, cmap = 'gray')

plt.subplot(348)
plt.imshow(rotated_image10, cmap = 'gray')

# image 100 of testing set

test_image150_10 = ndimage.rotate(np.reshape(int_image_10[:, 100], (46,56)), 270)
test_image150_100 = ndimage.rotate(np.reshape(int_image_100[:, 100], (46,56)), 270)
test_image150_300 = ndimage.rotate(np.reshape(int_image_300[:, 100], (46,56)), 270)

testing_image150 = np.reshape(testing_set[:, 100], (46,56))
rotated_image150 = ndimage.rotate(testing_image150,270)

plt.subplot(349)
plt.imshow(test_image150_10, cmap = 'gray')

plt.subplot(3,4,10)
plt.imshow(test_image150_100, cmap = 'gray')

plt.subplot(3,4,11)
plt.imshow(test_image150_300, cmap = 'gray')

plt.subplot(3,4,12)
plt.imshow(rotated_image150, cmap = 'gray')

# Nearest Neighbour Classification 

percentage_correctly_identified_images = np.array([])
computational_time_NN_array = np.array([])


for bases in range(415):
    start_NN = timeit.default_timer()
    # cannnot be done with 0 bases
    bases = bases+1
    
    correctly_identified_images = 0
    U = eigenvec_ld_nonzero[:, 0:bases]  
    
    W = np.matmul(U.T, A)
    W_testing = np.matmul(U.T, A_testing)
    
    
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
    
    end_NN = timeit.default_timer()
    computational_time_NN_array = np.append(computational_time_NN_array, (end_NN - start_NN))
    
    percentage_correctly_identified_images = np.append(percentage_correctly_identified_images, percentage_correct)
       


figure6 = plt.figure()
plt.subplot(121)
plt.xlabel('# bases learnt')
plt.ylabel('Percentage of correctly identified testing images')
figure6 = plt.plot(percentage_correctly_identified_images, linewidth=2)

plt.subplot(122)
plt.xlabel('# bases learnt')
plt.ylabel('Computational_time(s)')
figure7 = plt.plot(computational_time_NN_array, linewidth=2)

#Nearest neighbour examples 

bases = 415
U = eigenvec_ld_nonzero[:, 0:bases]  
    
W = np.matmul(U.T, A)
W_testing = np.matmul(U.T, A_testing)
k = 1

# image classes 5, 13, 48
# classe 5 alays correct, class 13 one yes, one no, class 48 always confused with class 29
# 48 is always confused with 29

index_testing = [8,9,24,25,94,95]
 

figure8 = plt.figure()

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
    
  
# Confusion matrix 

bases = 415
U = eigenvec_ld_nonzero[:, 0:bases]  
    
W = np.matmul(U.T, A)
W_testing = np.matmul(U.T, A_testing)

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


figure9 = plt.figure(figsize = (12,8))
ax = figure9.add_subplot(1,1,1)
plt.xlabel('Predicted class')
plt.ylabel('True class')
img = ax.matshow(confusion_matrix, cmap = 'YlGnBu')

alpha = np.arange(52)
ax.set_xticks(alpha + 0.5);
ax.set_xticklabels(alpha + 1, rotation = 'vertical');

ax.set_yticks(alpha + 0.5);
ax.set_yticklabels(alpha + 1);

ax.grid()

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

figure10 = plt.figure(figsize = (12,8))
ax = Axes3D(figure10)

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
    
rot_animation = animation.FuncAnimation(figure10, rotate, frames=np.arange(0,362,2),interval=100)

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