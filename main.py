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






class Perceptron:
    def __init__(self):
        self.w = none
        self.b = none
    def model(self, x):
        return 1 if (np.dot(self.w,x) >= self.b) else 0
    def predict(self, x):
        y = []
        for x_i in x:
            result = self.model(x)
            y.append(result)
        return np.array(y)
    def fit(self, x, y, epochs = 1, lr = 1):
        #TODO: what is x.shape?
        self.w  = np.ones(x.shape[1])
        self.b = 0
        accuracy = {}
        max_accuracy = 0
        wt_matrix = []
        for i in range(epochs):
            for x_i, y_i in zip(x,y):
                y_pred = self.model(x)
                if y == 1 and y_pred == 0:
                    # lr is learning rate
                    self.w = self.w + lr * x
                    self.b = self.b - lr 
                elif y==0 and y_pred == 1:
                    self.w = self.w - lr*x
                    self.b = self.b + lr

                wt_matrx.append(self.w)
                # TODO: where is accuracy_score from?
                accuracy[i] = accuracy_score(self.predict(x), y)
                if(accuracy[i] > max_accuracy):
                    max_accuracy = accuracy[i]
                    chkptw = self.w
                    chkptb = self.b
            # END INNER FOR LOOP    
            
            #save the values
            self.w = chkptw
            self.b = chkptb
            print(max_accuracy)
            mpl.plot(accuracy.values())
            mpl.xlabel("Epoch #")
            mpl.ylabel("Accuracy")
            mpl.ylim([0,1])
            mpml.show()
            
            #return the weight matrix for all epochs
            return np.array(wt_matrix)



