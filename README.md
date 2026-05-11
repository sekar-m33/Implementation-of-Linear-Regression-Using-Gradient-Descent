# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Import NumPy, Pandas, and Matplotlib libraries. Load the Startup.csv dataset and read R&D Spend as input X and Profit as output Y.

Step 2: Normalize the input data X using mean and standard deviation. Initialize slope m = 0, intercept b = 0, learning rate 0.1, epochs 1000, and sample size n.

Step 3: For each epoch, calculate predicted values using Y_predict = mX + b, find gradients dm and db, then update m and b using gradient descent formula.

Step 4: Print the final slope and intercept values. Plot the scatter graph of dataset points and regression line using Matplotlib. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SAKER M
RegisterNumber:  212225230257
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("50_Startups.csv")

A = data['R&D Spend'].values
b = data['Profit'].values

A = (A - A.mean()) / A.std()

c = 0
d = 0

learning_rate = 0.01
epochs = 1000
n = len(A)

for i in range(epochs):
    b_pred = c * A + d
    
    dm = (-2/n) * np.sum(A * (b - b_pred))
    db = (-2/n) * np.sum(b - b_pred)
    
    c = c - learning_rate * dm
    d = d - learning_rate * db

print("Slope (c):", c)
print("Intercept (d):", d)

b_pred = c * A + d

plt.scatter(A, b)
plt.plot(A, b_pred)

plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")

plt.show()
                                                                                                                                                                3
```

## Output:
<img width="716" height="189" alt="image" src="https://github.com/user-attachments/assets/1d2abb2d-a637-47f7-aae0-421d7ad4af35" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
