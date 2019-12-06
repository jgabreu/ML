import sys
import numpy as np
import matplotlib
import scipy as sp
import mglearn
import IPython
# import sklearn
from sklearn.datasets import load_iris


# print("Python version: {}".format(sys.version))
# print("NumPy version: {}".format(np.version))
# print("matplotlib version: {}".format(matplotlib.__version__))
# print("SciPy version: {}".format(sp.__version__))
# print("IPython version: {}".format(IPython.__version__))
# print("scikit-learn: {}".format(sklearn.__version__))

iris_dataset = load_iris()

print("Keys of iris_dataset: {}".format(iris_dataset.keys()) + "\n...")

print(" {}".format(iris_dataset['target_names']) + "\n...")

print("Shape of data: {}".format(iris_dataset['data'].shape))

print("Shape of data: {}".format(iris_dataset['target'].shape))

print("Shape of data: {}".format(iris_dataset['target']))

