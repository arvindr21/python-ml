# Video: https://www.youtube.com/watch?v=tNa99PG8hR8&index=2&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

for i in range(len(iris.target)):
    print "Exampele %d: label %s, features %s" % (i, iris.target[i], iris.data[i])

print "Length of total data %d " % len(iris.target)

test_idx = [0, 50, 100]
# Training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

print "Length of training data %d " % len(train_target)

# Testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

print "Length of testing data %d " % len(test_target)

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print test_target
print clf.predict(test_data)

# Visualization code
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names, filled=True, rounded=True, impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
