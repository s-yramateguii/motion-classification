import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from numpy import linalg
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D

# Function for computing centrals
def compute_centroids(X_traink, truth_labels):
    centroids = {}
    for j in range(0,3):
        movement_data = X_traink[truth_labels == j]
        centroid = np.mean(movement_data, axis=0)
        centroids[j] = centroid 
    return centroids

# Function to get trained and test labels
def get_labels(X_traink, centroids):
    labels = []
    for sample in X_traink:
        distances = []
        distances.append(linalg.norm(sample - centroids[0]))
        distances.append(linalg.norm(sample - centroids[1]))
        distances.append(linalg.norm(sample - centroids[2]))
        label = np.argmin(distances)
        labels.append(label)
    return labels

plt.style.use('seaborn-v0_8-poster')

#filename and folder to plot
fname= "walking_1"
folder = "hw2data/train/"

vals = np.load(folder+fname+".npy")
xyz = np.reshape( vals[:,:], (38,3,-1) )

print(xyz.shape)

#define the root joint and scaling of the values
r = 1000
xroot, yroot, zroot = xyz[0,0,0], xyz[0,0,1], xyz[0,0,2]

#define the connections between the joints (skeleton) 
I = np.array(
        [1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14, 15, 16, 17, 18, 19, 16, 21, 22, 23, 25, 26, 24, 28, 16, 30, 31,
         32, 33, 34, 35, 33, 37]) - 1
J = np.array(
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32,
         33, 34, 35, 36, 37, 38]) - 1

'''
Task 1
'''
X_train = []
movements = ['walking_', 'jumping_', 'running_']
for m in movements:
    for j in range(1,6):
        fname = m+str(j)
        vals = np.load(folder+fname+".npy")
        X_train.append(vals)
X_train = np.hstack(X_train)
print("Shape of X_train:", X_train.shape)

pca = PCA()
X_train_pca = pca.fit_transform(X_train.T)
cenergy = np.cumsum(pca.explained_variance_ratio_)
thresholds = [0.70, 0.80, 0.90, 0.95]
modes = []
for t in thresholds:
    for j in range(len(cenergy)):
        if cenergy[j] >= t:
            modes.append(j + 1)
            break 

for j in range(len(thresholds)):
    print(str(thresholds[j] * 100) + "%: " + str(modes[j]))

plt.figure(figsize=(10, 7))
plt.plot(np.arange(1, len(cenergy[:15]) + 1), cenergy[:15], linestyle='-',
         color='#1f77b4', label='Cumulative Energy')
plt.axhline(y=0.70, color='#d62728', linestyle='--', label='70% Threshold')
plt.axhline(y=0.80, color='#2ca02c', linestyle='--', label='80% Threshold')
plt.axhline(y=0.90, color='#ff7f0e', linestyle='--', label='90% Threshold')
plt.axhline(y=0.95, color='#9467bd', linestyle='--', label='95% Threshold')
plt.xlabel('PCA Spatial Modes')
plt.ylabel('Cumulative Energy')
plt.title('Cumulative Energy vs PCA Spatial Modes')
plt.legend()
plt.grid(True)
# plt.savefig(f"cumulative_energy.png", dpi=300, bbox_inches='tight')
plt.show()

'''
Task 2
'''
X_train2 = X_train_pca[:, :2]
X_train3 = X_train_pca[:, :3]
num_samples =5
wdata = X_train2[0:num_samples*100]
jdata = X_train2[100*num_samples:200*num_samples]
rdata = X_train2[200*num_samples:300*num_samples]
plt.figure(figsize=(10, 7))
plt.scatter(wdata[:, 0], wdata[:, 1], marker='o', linestyle='-', color='r', label='Walking')
plt.scatter(jdata[:, 0], jdata[:, 1], marker='o', linestyle='-', color='g', label='Jumping')
plt.scatter(rdata[:, 0], rdata[:, 1], marker='o', linestyle='-', color='b', label='Running')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('2D PCA Space')
plt.legend()
plt.grid(True)
# plt.savefig(f"2D_PCA.png", dpi=300, bbox_inches='tight')
plt.show()

wdata = X_train3[0:num_samples*100]
jdata = X_train3[100*num_samples:200*num_samples]
rdata = X_train3[200*num_samples:300*num_samples]
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(wdata[:, 0], wdata[:, 1], wdata[:, 2], marker='o', 
           linestyle='-', color='r', label='Walking')
ax.scatter(jdata[:, 0], jdata[:, 1], jdata[:, 2], marker='o', 
           linestyle='-', color='g', label='Jumping')
ax.scatter(rdata[:, 0], rdata[:, 1], rdata[:, 2], marker='o', 
           linestyle='-', color='b', label='Running')
ax.set_xlabel('PC1', labelpad=20)
ax.set_ylabel('PC2', labelpad=20)
ax.set_zlabel('PC3', labelpad=20)
ax.set_title('3D PCA Space')
ax.legend()
# plt.savefig(f"3D_PCA.png", dpi=300)
plt.show()

'''
Task 3
'''
# Computing Centroids function is above
truth_labels = np.zeros(num_samples * 3 * 100)
truth_labels[num_samples*100: num_samples*200] = 1
truth_labels[num_samples*200: num_samples*300] = 2

'''
Task 4
'''
k_vals = np.arange(1, 16, 1)
accuracy_vals = []
for k in k_vals:
    X_traink = X_train_pca[:, :k]
    centroids = compute_centroids(X_traink, truth_labels)
    trained_labels = get_labels(X_traink, centroids)
    accuracy = accuracy_score(truth_labels, trained_labels)
    accuracy_vals.append(accuracy)

plt.figure(figsize=(10, 7))
plt.plot(k_vals, np.array(accuracy_vals) * 100, marker='o', linestyle='-', color='b')
plt.xlabel('k')
plt.ylabel('Classification Accuracy (%)')
plt.title('Classification Accuracy vs. k-PCA Modes')
plt.xticks(np.arange(2, 17, 2))
plt.grid(True)
# plt.savefig(f"train_classification.png", dpi=300, bbox_inches='tight')
plt.show()

'''
Task 5
'''
X_test = []
folder = "hw2data/test/"
movements = ['walking_', 'jumping_', 'running_']
for m in movements:
    fname = m+"1t"
    vals = np.load(folder+fname+".npy")
    X_test.append(vals)
X_test = np.hstack(X_test)
print("Shape of tests:", X_test.shape)

X_test_pca = pca.transform(X_test.T)
test_truth_labels = np.zeros(300)
test_truth_labels[100:200] = 1
test_truth_labels[200:300] = 2
accuracy_train = []
accuracy_test = []

for k in k_vals:
    X_traink = X_train_pca[:, :k]
    X_testk = X_test_pca[:, :k]
    
    centroids = compute_centroids(X_traink, truth_labels)
    trained_labels = get_labels(X_traink, centroids)
    test_labels = get_labels(X_testk, centroids)
    
    train_acc = accuracy_score(truth_labels, trained_labels) 
    test_acc = accuracy_score(test_truth_labels, test_labels) 
    accuracy_train.append(train_acc)
    accuracy_test.append(test_acc)
    print(f"Accuracy with k={k} PCA components, train: {train_acc * 100:.2f}%, test: {test_acc * 100:.2f}")
    
plt.figure(figsize=(10, 7))
plt.plot(k_vals, np.array(accuracy_train) * 100, marker='o', linestyle='-', 
         color='b', label='Training Accuracy')
plt.plot(k_vals, np.array(accuracy_test) * 100, marker='s', linestyle='-', 
         color='r', label='Test Accuracy')
plt.xlabel('k')
plt.ylabel('Classification Accuracy (%)')
plt.title('Training vs. Test Classification Accuracy')
plt.xticks(np.arange(2, 17, 2))
plt.grid(True)
plt.legend()
# plt.savefig(f"test_train.png", dpi=300, bbox_inches='tight')
plt.show()


'''
Task 6
'''
LR = LogisticRegression(max_iter=1000)
accuracy_lr_train = []
accuracy_lr_test = []

for k in k_vals:
    X_traink = X_train_pca[:, :k]
    X_testk = X_test_pca[:, :k]

    LR.fit(X_traink, truth_labels)
    
    train_acc = LR.score(X_traink, truth_labels)
    test_acc = LR.score(X_testk, test_truth_labels)
    
    accuracy_lr_train.append(train_acc)
    accuracy_lr_test.append(test_acc)
    
    print(f"Accuracy for Logistic Regression with k={k} PCA components, train: {train_acc * 100:.2f}%, test: {test_acc * 100:.2f}")

plt.figure(figsize=(10, 7))
plt.plot(k_vals, np.array(accuracy_lr_train) * 100, marker='o', linestyle='-', 
         color='b', label='Training Accuracy')
plt.plot(k_vals, np.array(accuracy_lr_test) * 100, marker='s', linestyle='-', 
         color='r', label='Test Accuracy')
plt.xlabel('k')
plt.ylabel('Classification Accuracy (%)')
plt.title('Logistic Regression Classification Accuracy')
plt.xticks(np.arange(2, 17, 2))
plt.grid(True)
plt.legend()
# plt.savefig(f"lr.png", dpi=300, bbox_inches='tight')
plt.show()


# plot the skeleton accroding to joints (each plot is png image in anim folder)

# for tind in range(1,xyz.shape[2]):
    
#     fig = plt.figure(figsize = (10,10))
#     ax = plt.axes(projection='3d')
#     for ijind in range(0,I.shape[0]):
#         xline = np.array([xyz[I[ijind],0,tind], xyz[J[ijind],0,tind]])
#         yline = np.array([xyz[I[ijind],1,tind], xyz[J[ijind],1,tind]])
#         zline = np.array([xyz[I[ijind],2,tind], xyz[J[ijind],2,tind]])
#         # use plot if you'd like to plot skeleton with lines
#         ax.plot(xline,yline,zline)
    
#     # use scatter if you'd like to plot all points without lines 
#     # ax.scatter(xyz[:,0,tind],xyz[:,1,tind],xyz[:,2,tind], c = 'r', s = 50)   

#     ax.set_xlim([-r+xroot, r+xroot])
#     ax.set_zlim([-r+zroot, r+zroot])
#     ax.set_ylim([-r+yroot, r+yroot])
    
#     plt.savefig('anim/'+f"{tind}.png")
#     plt.close()

# when plotting a single sample (the skeleton can simply be plotted without saving an image)
    # plt.draw()
    # plt.pause(.001)
    # plt.show()

 
    
# save the animated plot as a gif in anim folder 
# from PIL import Image

# images = [Image.open('anim/'+f"{n}.png") for n in range(1,xyz.shape[2])]
# images[0].save('anim/'+fname+'.gif', save_all=True, append_images=images[1:], duration=30, loop=0) 

# # remove ploted png images
# for n in range(1,xyz.shape[2]):
#     os.remove('anim/'+f"{n}.png")
