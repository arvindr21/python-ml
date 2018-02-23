# Video: https://www.youtube.com/watch?v=cKxRvEZd3Mw&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=1
# Import decision tree
from sklearn import tree
# Define features
features = [[140,1],[130,1],[150,1],[170,1]]
# Define lables
labels = [0,0,1,1]
# Define classifier
clf = tree.DecisionTreeClassifier()
# Fit the feature and labels into the classifier
# to create a model
clf.fit(features, labels)
# Predict result
print clf.predict([[160, 0]])