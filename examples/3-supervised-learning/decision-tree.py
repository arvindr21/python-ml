# Video: https://www.youtube.com/watch?v=84gqSbLcBFE&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=4
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

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict based on testing data
predictions = clf.predict(X_test, check_input=True)
# print predictions

# Calculate Accuracy
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions, normalize=True, sample_weight=None)



