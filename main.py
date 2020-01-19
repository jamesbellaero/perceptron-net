import numpy as np
import sklearn.datasets
import pandas as pd
import matplotlib as mpl
from sklearn.model_selection import train_test_split


breast_cancer_set = sklearn.datasets.load_breast_cancer()

#TODO: what is pandas?
data = pd.DataFrame(breast_cancer_set.data, columns = breast_cancer_set.feature_names)
data["class"] = breast_cancer.target
#TODO: what are these?
data.head()
data.describe()

#TODO: how does this indexing work?
data["class"].value.counts().plot(kind = "barh")

mpl.xlabel("Count")
mpl.ylabel("Classes")
mpl.show()

x_raw = data.drop("class", axis = 1)
y = data["class"]
#TODO: huh
mnscaler = MinMaxScaler()
x_scaled = mnscaler.fit_transform(x_raw)
#TODO: why is x this?
x = pd.DataFrame(x_scaled, columns = data.drop("class",axis = 1).columns)

#TODO: what are stratify and random_state?
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .1, stratify = y, random_state = 1)





