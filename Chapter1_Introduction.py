import sys
import numpy as np
import matplotlib
import scipy as sp
import mglearn
import IPython
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# print("Python version: {}".format(sys.version))
# print("NumPy version: {}".format(np.version))
# print("matplotlib version: {}".format(matplotlib.__version__))
# print("SciPy version: {}".format(sp.__version__))
# print("IPython version: {}".format(IPython.__version__))
# print("scikit-learn: {}".format(sklearn.__version__))

iris_dataset = load_iris()

#print("Keys of iris_dataset: {}".format(iris_dataset.keys()) + "\n...")

#print(" {}".format(iris_dataset['target_names']) + "\n...")

#print("Shape of data: {}".format(iris_dataset['data'].shape))

#print("Shape of data: {}".format(iris_dataset['target'].shape))

#print("Shape of data: {}".format(iris_dataset['target']))

#train_test_split randomly mixes the data and splits it into two groups: the training data containing 75% of the cases and the testing data containing 25% of the cases

X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))