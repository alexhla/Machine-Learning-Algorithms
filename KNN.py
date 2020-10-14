### KNN, a sample is classified by a popularity vote of its nearest neighbors

import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):  # distance between two points
	# subtract one array from the other element by element and square the result, finally summing all elements for each array 
	# thus, the smaller result is a closer match
	return np.sqrt(np.sum((x1-x2)**2))  

class Knn:

		def __init__(self, k=3):
			self.k = k

		def fit(self, X, y):
			self.X_train = X  # data
			self.y_train = y  # target

		def predict(self,X):
			predicted_labels = [self._predict(x) for x in X]  # iterate over all the test data (X) passing in each element to _predict() storing the returned value
			print("predicted_labels = {}".format(predicted_labels))
			return np.array(predicted_labels)

		def _predict(self,x):
			# compute distances
			distances = [euclidean_distance(x, x_train) for x_train in self.X_train]  # calculate the distance between the test array (x) and every training sample (self.X_train)
			# print("distances = {}".format (distances))

			# get k nearest samples, lables
			k_indices = np.argsort(distances)[:self.k]  # sort the distances into a new array containing the indices of the sorted data, where min=0 and max=self.k 
			# print("k_indices = {}".format(k_indices))
			# print("k closest distances = {}".format([distances[x] for x in k_indices ]))

			k_nearest_labels = [self.y_train[i] for i in k_indices]  # get matching flower classes in target training data mapped from nearest (euclidean distance) indices
			# print("self.y_train = {}".format(self.y_train))
			# print("k_nearest_labels = {}".format(k_nearest_labels))

			# majority vote, most common class label
			most_common_label = Counter(k_nearest_labels).most_common(1)  # get most common flower class from nearest
			# print("most_common_label = {}".format(most_common_label))
			return most_common_label[0][0]
			