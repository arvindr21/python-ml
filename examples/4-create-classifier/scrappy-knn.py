# Video: https://www.youtube.com/watch?v=AoeEHqVSNOw&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=5
from scipy.spatial import distance
# compute Euclidean distance 
def euc(a, b):
	return distance.euclidean(a, b)

# Build the classifier
class ScrappyKNN():
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train


	def predict(self, X_test):
		predictions = []
		for row in X_test:
			label = self.closest(row)
			predictions.append(label)
		return predictions

	def closest(self, row):
		best_dist = euc(row, self.X_train[0])
		best_index = 0
		for i in range(1, len(self.X_train)):
			dist = euc(row, self.X_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.y_train[best_index]

# import Iris dataset
from sklearn import datasets
iris = datasets.load_iris()

# Features 
X = iris.data
# Labels
y = iris.target

# A classifier is an equation
# f(X) = y
# f(X) => Features
# y => Labels

# Partition data into train and test sets
# from sklearn.cross_validation import train_test_split => Depc
from sklearn.model_selection import train_test_split
# added random_state=42 so that the split is same always! not the % but values
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Use ScrappyKNN
clf = ScrappyKNN()
clf.fit(X_train, y_train)

# Predict based on testing data
predictions = clf.predict(X_test)
# print predictions

# Calculate Accuracy
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions, normalize=True, sample_weight=None)