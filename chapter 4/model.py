# 1. Linear Regression
# Real-Life Uses,Predicting house prices from size, location,Predicting salary from experience,Predicting temperature
# y=mX+c
"""from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4]])  # experience
y = np.array([40, 45, 50, 55])      # salary in thousands

model = LinearRegression()
model.fit(X, y)

print(model.predict([[5]])) """  # predict salary for 5 years experience
# ----------------------------------------------------------------------------------------------------------------
# 2 Logistic Regression
# Spam vs not spam,Fraud detection,Diabetes prediction (0 = no, 1 = yes),Customer will buy or not , ogistic regression does NOT predict directly like a line.
# z=mX+c , passes it through the sigmoid functio p=σ(z)=1+e−z1​, p = probability of class 1 ,m and c are learned from data
from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([0, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)

print(model.predict([[3.5]])) 
print(model.predict_proba([[3.5]]))  # probability of class 1
