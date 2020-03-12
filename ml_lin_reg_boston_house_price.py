import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

boston = pd.read_csv('boston.csv')
print(boston.head())
#extract target column
y=boston['MEDV'].values
# Remove the target column from data set. Convert DF into NP array
X=boston.drop('MEDV', axis=1).values
print(X.shape)
print(y.shape)
X_rooms = X[:, 5]
print(X_rooms.shape)

y = y.reshape(-1,1)
X_rooms = X_rooms.reshape(-1,1)

print(y.shape)
print(X_rooms.shape)

plt.scatter(X_rooms, y)
plt.ylabel('Value of house/1000 ($)')
plt.xlabel('Number of rooms')
plt.show()

reg = LinearRegression()
reg.fit(X_rooms, y)

prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1,1)

plt.scatter(X_rooms, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space), color='black', linewidth=3)
plt.show()
