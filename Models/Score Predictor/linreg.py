import numpy as np
import pandas as pd
import matplotlib.pyplot as plt,mpld3
import pickle
import os

df=pd.read_csv("student_scores.csv")
x=df.drop("Scores",axis=1)
y=df["Scores"]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
def train_model():

    from sklearn.linear_model import LinearRegression
    model=LinearRegression()
    model.fit(x_train,y_train)

    
    with open('model_lr.pkl',"wb") as file:
        pickle.dump(model, file)

def test_model():
    filename = 'model_lr.pkl'
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    from sklearn.model_selection import cross_val_score
    y_pred=model.predict(x_test)
    scores=cross_val_score(model,x_train,y_train,cv=5)
    print(f"accuracy={np.mean(scores)*100:.2f}%")

    from sklearn import metrics
    print(f"Mean Absolute Error={metrics.mean_absolute_error(y_test,y_pred)}")

train_model()

def predicts(h):

    filename = 'model_lr.pkl'
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    scores = model.predict([[h]])
    return scores[0]

# if __name__ == '__main__':
#     train_model()
