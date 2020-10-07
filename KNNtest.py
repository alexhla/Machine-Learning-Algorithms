import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

print('''Loading Iris Flower Data Set\n
This data sets consists of 3 different types of irises’ 
(Setosa, Versicolour, and Virginica) petal and sepal length, 
stored in a 150x4 numpy.ndarray \n''')

iris = datasets.load_iris()
X, y = iris.data, iris.target

print("iris.target[0] is iris type")
print("iris.data[0] is sepal length in cm")
print("iris.data[1] is sepal width in cm")
print("iris.data[2] is petal length in cm")
print("iris.data[3] is petal width in cm")

print("\nSplitting Iris Data Set Into Train/Test Data Sets at a 80/20 Ratio\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

print("X training data shape {} and size {}". format(X_train.shape, X_train.size))
print("y training target shape is {} and size {}\n". format(y_train.shape, y_train.size))

print("X testing data shape {} and size {}". format(X_test.shape, X_test.size))
print("y testing target shape is {} and size {}\n". format(y_test.shape, y_test.size))

# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()

# a = [1, 1, 1, 2, 2, 3, 4, 5, 6]
# from collections import Counter
# most_common = Counter(a).most_common(1)
# print(most_common[0][0])

from KNN import Knn
clf = Knn(k=5)  # number of neighbors
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

accuracy = np.sum(predictions == y_test) / len(y_test)
print(accuracy)