from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score


class LogisticReg_model:
    def __init__(self,data):
        self.X,self.y=df['X'].values.reshape(-1,1),df['Y'].values.reshape(-1,1)
        self.preprocess()
    def preprocess(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=42)
        self.scaler = MinMaxScaler().fit(self.X_train)
        self.X_train_prep=self.scaler.transform(self.X_train)
        self.X_test_prep=self.scaler.transform(self.X_test)
    def fit(self):
        self.model = LogisticRegression(random_state=0).fit(self.X_train_prep,self.y_train)
        self.evaluate()
    def evaluate(self):
        y_pred=self.model.predict(self.X_test_prep)
        print(f"Accuracy of the classifier is: {accuracy_score(self.y_test, y_pred)*100}")
        print(f"Precision Score of the classifier is: {precision_score(self.y_test, y_pred)}")
        print(f"Recall Score of the classifier is: {recall_score(self.y_test, y_pred)}")
    def predict(self,X_test):
        X_test_prep=self.scaler.transform(X_test)
        return self.model.predict(X_test_prep)
       
       
#demo:
#def crate_data():
#     x=np.random.randint(1000,3000,150)
#     y=np.array([1.0 if num>=0.5 else 0.0 for num in np.random.rand(150)])
#     df=pd.DataFrame({
#         'X':x,
#         'Y':y
#     })
#     return df
# df=crate_data()
# X,y=df['X'].values.reshape(-1,1),df['Y'].values.reshape(-1,1)
# model=LogisticReg_model(df)
# model.fit()
# model.predict(np.array([1500]).reshape(-1,1))