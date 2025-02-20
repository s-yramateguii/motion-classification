# Motion Classification

This project involves motion classification using Principal Component Analysis (PCA) to reduce the dimensionality of joint movement data. The dataset consists of motion samples representing three types of human movements: walking, jumping, and running. The goal is to analyze the data using PCA and perform classification based on reduced dimensionality. Various machine learning techniques, such as k-nearest neighbors and logistic regression, are applied to classify the movements.

## Steps

1. **Data Loading and Preprocessing:**
   - The data is loaded from `.npy` files, which contain joint position data for different movements.
   - The data is reshaped to separate the joint positions by frame and joint.

2. **PCA for Dimensionality Reduction:**
   - PCA is applied to reduce the dimensionality of the motion data. Cumulative explained variance is plotted for different thresholds (70%, 80%, 90%, and 95%) to visualize the number of principal components required to explain the desired amount of variance.

3. **Data Visualization:**
   - A 2D and 3D scatter plot of the PCA-transformed data is created to visualize how different movements (walking, jumping, and running) are separated in the reduced PCA space.

4. **Centroid Calculation and k-NN Classification:**
   - For each class (walking, jumping, running), centroids are computed based on the PCA-reduced training data. 
   - The classification accuracy is evaluated for different values of k (number of principal components).

5. **Logistic Regression Classification:**
   - A logistic regression classifier is applied to the PCA-reduced data, and classification accuracy is compared to the centroid classifier results.
