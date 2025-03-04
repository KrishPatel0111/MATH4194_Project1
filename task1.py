import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


noise_levels = [0.1,0.5,1.0,2.0]
for std in noise_levels:
  x = np.linspace(0,10,100).reshape(-1, 1)
  #print(x)
  e = np.random.normal(0,std,100)
  #print(e)
  y = 3*x.flatten()+e
  #print(y)

  x_train,x_test, y_train, y_test = train_test_split(x, y , test_size=0.2, random_state=2)
  

  model = LinearRegression()
  model.fit(x_train,y_train)
  y_test_predicted = model.predict(x_test)
  mse = mean_squared_error(y_test,y_test_predicted)
  print(f"MSE for noise level {std} is:" + str(mse))

  plt.scatter(x, y, label = "Data points")
  plt.plot(x_test, y_test_predicted, color='red', label='Linear regression')

  plt.xlabel('X-axis with noise level: ' + str(std))
  plt.ylabel('Y-axis with noise level: ' + str(std))
  plt.title('Data with Regression Line')
  plt.legend()

 
  plt.show()

