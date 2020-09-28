import pandas as pd
from sklearn import datasets

iris_raw = datasets.load_iris()
iris = pd.DataFrame(iris_raw.data)
iris.columns = iris_raw.feature_names
iris["species"] = iris_raw.target
