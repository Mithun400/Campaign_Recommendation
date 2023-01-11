from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error


class LinearReg_model:
    def __init__(self,data):
        self.X,self.y=df['X'].values.reshape(-1,1),df['Y'].values.reshape(-1,1)
        self.preprocess()
    def preprocess(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=42)
        self.scaler = MinMaxScaler().fit(self.X_train)
        self.X_train_prep=self.scaler.transform(self.X_train)
        self.X_test_prep=self.scaler.transform(self.X_test)
    def fit(self):
        self.model = LinearRegression().fit(self.X_train_prep,self.y_train)
        self.evaluate()
    def evaluate(self):
        y_pred=self.model.predict(self.X_test_prep)
        print(f"Accuracy of the classifier is: {mean_squared_error(self.y_test, y_pred)*100}")

    def predict(self,X_test):
        X_test_prep=self.scaler.transform(X_test)
        return self.model.predict(X_test_prep).round(0)
        
        
#demo:
# def crate_data():
#     x=np.random.randint(1000,1500,100).reshape(-1,)
#     y=np.random.randint(1,10,100).reshape(-1,)
#     df=pd.DataFrame({
#         'X':x,
#         'Y':y
#     })
#     return df
# df=crate_data()
# X,y=df['X'].values.reshape(-1,1),df['Y'].values.reshape(-1,1)
# model=LinearReg_model(df)
# model.fit()
# model.predict(np.array([1300]).reshape(-1,1))