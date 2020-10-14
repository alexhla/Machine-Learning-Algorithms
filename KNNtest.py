import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

print('''\n##### Loading Iris Flower Data Set ######
This data sets consists of 3 different types of irises’ 
(Setosa, Versicolour, and Virginica) petal and sepal length, 
stored in a 150x1 (target) and 150x4 (data) numpy.ndarray\n''')

iris = datasets.load_iris()
X, y = iris.data, iris.target

print("iris.target[0] is the iris type")
print("iris.data[0] is the sepal length in cm")
print("iris.data[1] is the sepal width in cm")
print("iris.data[2] is the petal length in cm")
print("iris.data[3] is the petal width in cm")


print('''\n##### Splitting Data #####
Data set split into Train/Test data sets at a 80/20 ratio\n''')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

print("Training data (X_train) has {} elements\nX_train.shape = {} --> 120 arrays containing 4 elements e.g. 120 x {}\n". format(X_train.size, X_train.shape, X_train[0]))
print("Training target (y_train) has {} elements\ny_train.shape = {} --> A single array of 120 elements\n". format(y_train.size, y_train.shape))

print("Testing data (X_test) has {} elements\nX_test.shape = {} --> 30 arrays containing 4 elements e.g. 120 x {}\n". format(X_test.size, X_test.shape, X_test[0]))
print("Testing target (y_target) has {} elements\ny_test.shape = {} --> A single array of 30 elements\n". format(y_test.size, y_test.shape))

print("target_labels = {}".format(y_test))

### view data
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()

# a = [1, 1, 1, 2, 2, 3, 4, 5, 6]
# from collections import Counter
# most_common = Counter(a).most_common(1)
# print(most_common[0][0])

from KNN import Knn
clf = Knn(k=5)  # instantiate a Knn classifier (clf) passing in the number of neighbors (default is 3)
clf.fit(X_train, y_train)  # pass the training data to your Knn classifier
predictions = clf.predict(X_test)  

# predict() algorithm... 
# 1) calculate distance between X_test and every training data entry
# 2) find kth nearest training data entries i.e. smallest euclidean distance
# 3) match kth nearest data entries with flower class target array and select the most common

acc = np.sum(predictions == y_test) / len(y_test)
print("accuracy = {}".format(acc))