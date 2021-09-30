import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

np.set_printoptions(precision=2)

fruits = pd.read_table('/Users/georgeanton/Desktop/Applied_Data_Science_Python_UM/Applied_Machine_Learning/course3_downloads/fruit_data_with_colors.txt')

feature_names_fruits = fruits.columns[3:]

X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label'] #output/classification label
target_names_fruits = fruits['fruit_name'].unique()

X_fruits_2d = fruits[['height','width']]
y_fruits_2d = y_fruits

#split fruit data set 3:1 train-test
X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0)

from sklearn.preprocessing import MinMaxScaler


#Fit and transform the sets by min/max values
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

#Only transform the test set, fitting is to be avoided to prevent Data Leakage
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
# print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train_scaled, y_train)))
# print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test_scaled, y_test)))

#Outputs are 0.95 for training set and 1.00 for test set
#Perhaps some overfitting occurring ?

example_fruit = [[5.5, 2.2, 10, 0.70]]
example_fruit_scaled = scaler.transform(example_fruit)
knn.predict(example_fruit_scaled)[0] #predicts/classifies the instance
target_names_fruits[knn.predict(example_fruit_scaled)[0] -1] #the -1 is because the targets are indexed from 0->3

from sklearn.datasets import make_classification, make_blobs, load_breast_cancer
from matplotlib.colors import ListedColormap
from adspy_shared_utilities import load_crime_dataset

cmap_bold = ListedColormap(['#FFFF00','#00FF00','#0000FF','#000000'])

#synthetic dataset for simple regression

from sklearn.datasets import make_regression

# plt.figure()
# plt.title('Sample regression problem with one input variable')
# X_R1, y_R1 = make_regression(n_samples=100, n_features=1, n_informative=1, bias=150.0,
# noise=30, random_state=0)
# plt.scatter(X_R1, y_R1, marker='o', s=50)
# plt.show()

#synthetic dataset for more complex regression

from sklearn.datasets import make_friedman1
#
# plt.figure()
# plt.title('Complex regression problem with one input variable')
# X_F1, y_F1 = make_friedman1(n_samples=100, n_features=7, random_state=0)
#
# plt.scatter(X_F1[:,2], y_F1, marker='o', s=50)
# plt.show()

#synthetic dataset for classification (binary) weith classes that are not linearly separable

# X_D2, y_D2 = make_blobs(n_samples=100, n_features=2, centers=8, cluster_std=1.3, random_state=4)
#
# y_D2 = y_D2%2
#
# plt.figure()
# plt.title('Sample binary classification problem with non-linearly separable classes')
# plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2, marker='o', s=50, cmap=cmap_bold)
# plt.show()

#K-NN Classification

from adspy_shared_utilities import plot_two_class_knn


# synthetic dataset for classification (binary)
# plt.figure()
# plt.title('Sample binary classification problem with two informative features')
# X_C2, y_C2 = make_classification(n_samples = 100, n_features=2,
#                                 n_redundant=0, n_informative=2,
#                                 n_clusters_per_class=1, flip_y = 0.1,
#                                 class_sep = 0.5, random_state=0)
# plt.scatter(X_C2[:, 0], X_C2[:, 1], c=y_C2,
#            marker= 'o', s=50, cmap=cmap_bold)
# plt.show()
#
#
# X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state=0)
# #
# plot_two_class_knn(X_train, y_train, 1, 'uniform', X_test, y_test)
# plot_two_class_knn(X_train, y_train, 3, 'uniform', X_test, y_test)
# plot_two_class_knn(X_train, y_train, 11, 'uniform', X_test, y_test)

#K-NN Regression














#
